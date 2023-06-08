import os
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt
import ot

import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import logging

from utils.utils import flatten_tensor_list, get_summary_stats, dump_tensor_to_mat
from policies.pruners import GradualPruner
from utils.plot import analyze_new_weights
from common.io import _load, _dump
from common.timer import Timer
from copy import copy


class OptimalTransportPruner(GradualPruner):
    have_not_started_pruning = True

    def __init__(self, model, inp_args, **kwargs):
        super(OptimalTransportPruner, self).__init__(model, inp_args, **kwargs)
        print("In Optimal Transport Pruner")
        self._fisher_inv_diag = None
        self._prune_direction = inp_args.prune_direction
        self._zero_after_prune = inp_args.zero_after_prune
        self._inspect_inv = inp_args.inspect_inv
        self._fisher_mini_bsz = inp_args.fisher_mini_bsz
        if self._fisher_mini_bsz < 0:
            self._fisher_mini_bsz = 1
        if self.args.woodburry_joint_sparsify:
            self._param_stats = []
        if self.args.dump_fisher_inv_mat:
            self._all_grads = []
        if self.args.fisher_inv_path is None:
            N_samples = self.args.fisher_subsample_size * self.args.fisher_mini_bsz
            N_batches = self.args.fisher_subsample_size
            seed = self.args.seed
            self.args.fisher_inv_path = os.path.join(
                "./prob_regressor_data",
                f"{self.args.arch}_{self.args.dset}_{N_samples}samples_{N_batches}batches_{seed}seed.fisher_inv",
            )

    def _get_weights(self):
        weights = []
        masks = []

        for idx, module in enumerate(self._modules):
            assert self._weight_only
            weights.append(module.weight.data.flatten())
            masks.append(module.weight_mask.data.flatten())
        weights = torch.cat(weights).to(module.weight.device)
        masks = torch.cat(masks).to(module.weight_mask.device)
        return weights, masks

    def _compute_sample_fisher(self, loss, return_outer_product=False):
        """Inputs:
            loss: scalar or B, Bx1 tensor
        Outputs:
            grads_batch: BxD
            gTw: B (grads^T @ weights)
            params: D
            ff: 0.0 or DxD(sum of grads * grads^T)
        """
        ys = loss
        params = []
        for module in self._modules:
            for name, param in module.named_parameters():
                # print("name is {} and shape of param is {} \n".format(name, param.shape))
                if self._weight_only and "bias" in name:
                    continue
                else:
                    params.append(param)

        grads = torch.autograd.grad(ys, params)  # first order gradient

        # Do gradient_masking: mask out the parameters which have been pruned previously
        # to avoid rogue calculations for the hessian

        for idx, module in enumerate(self._modules):
            grads[idx].data.mul_(module.weight_mask)
            params[idx].data.mul_(module.weight_mask)

        grads = flatten_tensor_list(grads)
        params = flatten_tensor_list(params)

        if self.args.dump_grads_mat:
            self._all_grads.append(grads)

        self._num_params = len(grads)
        self._old_weights = params

        # gTw = params.T @ grads
        gTw = None

        if not return_outer_product:
            # return grads, grads, gTw, params
            return grads, None, gTw, params
        else:
            return grads, torch.ger(grads, grads), gTw, params

    def _get_pruned_wts_scaled_basis(self, pruned_params, flattened_params):
        # import pdb;pdb.set_trace()
        return -1 * torch.div(
            torch.mul(pruned_params, flattened_params), self._fisher_inv_diag
        )

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _compute_wgH(self, dset, subset_inds, device, num_workers, debug=False):
        st_time = time.perf_counter()
        self._model = self._model.to(device)

        print("in optimal transport pruning: len of subset_inds is ", len(subset_inds))

        goal = self.args.fisher_subsample_size

        assert len(subset_inds) == goal * self.args.fisher_mini_bsz

        dummy_loader = torch.utils.data.DataLoader(
            dset,
            batch_size=self._fisher_mini_bsz,
            num_workers=num_workers,
            sampler=SubsetRandomSampler(subset_inds),
        )
        ## get g and g^T * w, denoted as XX and yy respectively
        Gs = []
        GTWs = []

        if self.args.aux_gpu_id != -1:
            aux_device = torch.device("cuda:{}".format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device("cpu")

        if self.args.disable_log_soft:
            # set to true for resnet20 case
            # set to false for mlpnet as it then returns the log softmax and we go to NLL
            criterion = torch.nn.functional.cross_entropy
        else:
            criterion = F.nll_loss

        self._fisher_inv = None

        num_batches = 0
        num_samples = 0

        FF = 0.0
        for in_tensor, target in dummy_loader:
            self._release_grads()

            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            try:
                loss = criterion(output, target, reduction="mean")
            except:
                import pdb

                pdb.set_trace()
            # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
            # which is just a practical heuristic to utilize more datapoints

            ## compute grads, XX, yy
            g, _, gTw, w = self._compute_sample_fisher(loss, return_outer_product=False)
            Gs.append(torch.Tensor(g[None, :].detach().cpu().numpy()))
            # GTWs.append(torch.Tensor(gTw[None, None].detach().cpu().numpy()))
            w = w.detach().cpu().numpy()
            # FF += ff
            del g, gTw

            num_batches += 1
            num_samples += self._fisher_mini_bsz
            if num_samples == goal * self._fisher_mini_bsz:
                break

        ## save Gs and GTWs
        # grads = torch.cat(Gs, 0) * 1 / np.sqrt(self.args.fisher_subsample_size)
        # wTgs = torch.cat(GTWs, 0) * 1/np.sqrt(self.args.fisher_subsample_size)
        grads = torch.cat(tuple(Gs), 0)
        # wTgs = torch.cat(tuple(GTWs), 0)
        # FF = FF / self.args.fisher_subsample_size
        print(
            "# of examples done {} and the goal (#outer products) is {}".format(
                num_samples, goal
            )
        )
        print("# of batches done {}".format(num_batches))

        end_time = time.perf_counter()
        print(
            "Time taken to compute empirical fisher is {} seconds".format(
                str(end_time - st_time)
            )
        )

        return grads, GTWs, w, None

    def _hard_threshold(self, x, k):
        # Set all but the largest k elements in x to zero
        if k < 1:
            return torch.zeros(len(x))
        threshold = x.abs().flatten().kthvalue(int(x.numel() - k + 1), dim=-1)[0]
        x[x.abs() < threshold] = 0
        return x

    def _set_weights(self, weight_updates, module_param_indices_list):
        for idx, module in enumerate(self._modules):
            weight = weight_updates[
                module_param_indices_list[idx] : module_param_indices_list[idx + 1]
            ]
            weight = weight.view(module.weight.shape)

            mask = (weight != 0).float()
            module.weight_mask = mask

            with torch.no_grad():
                module.weight.data = module.weight.data * mask

    def _pruning_objective(self, X, y, PI, w, w_bar, lam, ot_dist, transport):
        n = X.shape[0]
        lam_torch = torch.tensor(lam, device=w.device)

        if not transport:
            Q = (
                (1/2) * torch.linalg.norm(y - X @ w) ** 2
                + (n/2) * lam_torch * torch.linalg.norm(w - w_bar) ** 2
            )
        else:
            Q = (
                (1/2) * ot_dist + (n/2) * lam_torch * torch.linalg.norm(w - w_bar) ** 2
            )
        return Q

    def _get_weight_update(
        self,
        grads,
        target_weights,
        lam,
        transport,
        dset,
        subset_inds,
        device,
        num_workers,
        module_param_indices_list,
        iter_num,
    ):
        n = len(grads)
        params_num = grads.shape[1]
        init_sparsity = 0
        # sparsity_levels = np.arange(
        #     init_sparsity,
        #     self._target_sparsity + 1e-10,
        #     1 / iter_num * (self._target_sparsity - init_sparsity),
        # )[1:]
        sparsity_levels = [self._target_sparsity for i in range(iter_num)]
        lam_torch = torch.tensor(lam, device=device)
        n_torch = torch.tensor(n, device=device)
        PI = torch.eye(n) * 1 / n
        PI_norm = torch.linalg.norm(PI, ord=2)
        PI = PI.to(device)
        w, _ = self._get_weights()
        w_bar = copy(target_weights)
        w = w.to(device)
        w_bar = w_bar.to(device)
        for i, sparsity in enumerate(sparsity_levels):
            print(f"Iteration {i}:")
            non_zero_params_num = int(params_num * (1 - sparsity))
            grads, _, _, _ = self._compute_wgH(
                dset, subset_inds, device, num_workers, debug=False
            )

            X = grads.to(device)
            y = X @ w_bar
            X_norm = torch.linalg.norm(X, ord=2)

            if not transport:
                print("\t Perform no transport update")
                L = n_torch * lam_torch + X_norm**2
                L = L.to(device)
                print(f"\t Step size tau is {1/L}")
                print("\t w", w)
                print("\t w_bar", w_bar)
                w_new = w - (1 / L) * (
                    X.T @ (X @ w - y) + n_torch * lam_torch * (w - w_bar)
                )
                print("\t First part sum:", (1 / L) * (X.T @ (X @ w - y)).abs().sum())
                print(
                    "\t Second part sum:",
                    (1 / L) * (n_torch * lam_torch * (w - w_bar)).abs().sum(),
                )
                np.savetxt(f"logs/X_{i+1}.csv", (X.T).cpu(), delimiter=",")
            else:
                print("\t Perform optimal transport update")
                L = lam_torch + X_norm * X_norm * PI_norm
                L = L.to(device)
                print(f"\t Step size tau is {1/L}")
                print("\t w", w)
                print("\t w_bar", w_bar)
                w_new = w - (1 / L) * (X.T @ PI @ (X @ w - y) + lam_torch * (w - w_bar))
                print(
                    "\t First part sum:", (1 / L) * (X.T @ PI @ (X @ w - y)).abs().sum()
                )
                print(
                    "\t Second part sum:",
                    (1 / L) * (lam_torch * (w - w_bar)).abs().sum(),
                )
                np.savetxt(f"logs/X_PI{i+1}.csv", (X.T @ PI).cpu(), delimiter=",")

            # w_bar = copy(w)
            w_old = copy(w)

            print(f"\t Non-zero value num of w_{i}:", (w != 0).sum())
            w = self._hard_threshold(w_new, non_zero_params_num)
            print(f"\t Non-zero value num of w_{i+1}", (w != 0).sum())

            self._set_weights(
                weight_updates=w, module_param_indices_list=module_param_indices_list
            )
            print("\t Model weights updated")

            PI, ot_dist = self._get_transportation_plan(
                grads=grads, w=w, w_target=w_bar, reg=0.05, transport=transport
            )
            PI = PI.to(device)
            ot_dist = ot_dist.to(device)

            Q = self._pruning_objective(X=X, y=y, PI=PI, w=w, w_bar=w_bar, lam=lam, ot_dist=ot_dist, transport=transport)
            print(f"\t Objective function value: {Q}")

            # weight_norm_change = torch.norm(w - w_old)
            # print(f"weight norm change: {weight_norm_change}")

        plt.imshow(PI.to("cpu"))
        plt.savefig("transportation_plan.pdf", format="pdf")

    def _get_transportation_plan(self, grads, w, w_target, reg, transport):
        n = len(grads)
        if not transport:
            return torch.eye(n) * 1 / n

        original_distr = grads.to("cpu") * w_target.to("cpu")
        embedded_distr = grads.to("cpu") * w.to("cpu")

        original_distr = original_distr.detach().numpy()
        embedded_distr = embedded_distr.detach().numpy()

        original_distr_mass = [1 / n for i in range(n)]
        embedded_distr_mass = [1 / n for i in range(n)]

        # Compute the cost matrix (squared Euclidean distance) between original_distr and embedded_distr
        M = ot.dist(original_distr, embedded_distr, metric="sqeuclidean")

        PI = ot.bregman.sinkhorn_knopp(
            original_distr_mass, embedded_distr_mass, M, reg, numItermax=5000
        )
        np.savetxt("PI.csv", PI, delimiter=",")
        # np.savetxt("M.csv", M, delimiter=",")
        return torch.from_numpy(PI).float().to(w.device), torch.tensor(np.sum(PI*M))

    def on_epoch_begin(
        self, dset, subset_inds, device, num_workers, epoch_num, **kwargs
    ):
        meta = {}
        if self._pruner_not_active(epoch_num):
            print("Pruner is not ACTIVEEEE yaa!")
            if OptimalTransportPruner.have_not_started_pruning:
                self._target_weights, _ = self._get_weights()
                print("Target weights updated")
            return False, {}

        OptimalTransportPruner.have_not_started_pruning = False
        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        # reinit params if they were deleted during gradual pruning
        if not hasattr(self, "_all_grads"):
            self._all_grads = []
        if not hasattr(self, "_param_stats"):
            self._param_stats = []

        self._param_idx = 0

        flat_pruned_weights_list = []
        flat_module_weights_list = []
        module_shapes_list = []
        module_param_indices_list = []

        for idx, module in enumerate(self._modules):
            module_param_indices_list.append(self._param_idx)
            assert self._weight_only
            module_shapes_list.append(module.weight.shape)

            self._param_idx += module.weight.numel()

        for idx, module in enumerate(self._modules):
            flat_pruned_weights_list.append(module.weight.flatten())
            flat_module_weights_list.append(module.weight.flatten())

        module_param_indices_list.append(self._param_idx)

        flat_pruned_weights_list = flatten_tensor_list(flat_pruned_weights_list)
        flat_module_weights_list = flatten_tensor_list(flat_module_weights_list)

        # Compute the weight updates using the custom function
        grads, _, _, _ = self._compute_wgH(
            dset, subset_inds, device, num_workers, debug=False
        )
        self._get_weight_update(
            grads=grads,
            target_weights=self._target_weights,
            lam=0,
            transport=self.args.ot,
            dset=dset,
            subset_inds=subset_inds,
            device=device,
            num_workers=num_workers,
            module_param_indices_list=module_param_indices_list,
            iter_num=20,
        )

        return True, meta

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
        backup_masks = []
        for idx, module in enumerate(self._modules):
            backup_masks.append(module.weight_mask)
            module.weight_mask = torch.ones_like(module.weight_mask)

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

        def add_noise(tensor, noise_factor=1):
            tensor = tensor.float()  # Convert tensor to float
            noise_factor = torch.tensor(noise_factor).to(tensor.device)
            noise = torch.randn_like(tensor) * noise_factor
            return tensor + noise
            # return torch.zeros_like(tensor)

        def randomize_labels(labels, fraction=0.5):
            num_random_labels = int(fraction * len(labels))
            random_indices = torch.randperm(len(labels))[:num_random_labels]
            random_labels = torch.randint(low=0, high=10, size=(num_random_labels,))
            labels[random_indices] = random_labels
            return labels

        def add_noise_to_grads(grads, noise_std_dev=1, prop=0.25):
            # Randomly select two row indices
            m = int(grads.size(0)*prop)
            indices = torch.randperm(grads.size(0))[:m]
            indices = indices.to(grads.device)

            # Generate Gaussian noise to add
            noise = torch.normal(0, noise_std_dev, size=(m, grads.size(1)))
            noise = noise.to(grads.device)

            # Add the noise to the randomly selected rows
            grads[indices] += noise

            return grads

        for in_tensor, target in dummy_loader:
            self._release_grads()

            # in_tensor = add_noise(in_tensor)
            # target = randomize_labels(target)

            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            loss = criterion(output, target, reduction="mean")

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

        grads = torch.cat(tuple(Gs), 0)
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
        for idx, module in enumerate(self._modules):
            module.weight_mask = backup_masks[idx]

        # if self.args.ot:
        #     print('Zero grads')
        #     grads = torch.zeros_like(grads).to('cuda')

        grads = add_noise_to_grads(grads=grads, noise_std_dev=1.0*grads.std(), prop=0.3)
        # grads = (torch.ones_like(grads) * grads.mean()).to('cuda') 

        return grads, GTWs, w, None

    def __top_k_indice(self, vector, k):
        """Function to get the indices of the k largest values of a vector"""
        _, indices = torch.topk(vector.abs(), k)
        return set(indices.tolist())

    def __same_support(self, a, b, k):
        """Check if two vectors have the same support"""
        return self.__top_k_indice(a, k) == self.__top_k_indice(b, k)

    def __find_tau_c(self, a, b, k, eps=1e-16):
        """Find the largest tau such that the support of a-tau*b is the same as a"""
        low, high = 0, min(torch.max(a / (b + eps)), 100)

        count = 0
        while (high - low > eps) and (count < 50):
            count += 1
            # print(f'low={low}, high={high}')
            mid = (low + high) / 2
            if self.__same_support(a, a - mid * b, k):
                low = mid
            else:
                high = mid

        return torch.tensor(low)

    def __find_minimizing_tau(
        self, a, b, tau_c, k, X, y, w_bar, lam, reg, transport, PI, lr=0.01, eps=1e-16
    ):
        """Find the tau that minimizes Q"""
        tau = torch.tensor([0.5], requires_grad=True)
        tau.data = tau.data.clamp(min=0, max=tau_c - eps)
        optimizer = optim.Adam([tau], lr=lr)

        a = self._hard_threshold(a, k)

        a = a.to("cpu")
        b = b.to("cpu")

        mask = (a != 0).float()
        b *= mask.to("cpu")

        for i in range(1000):
            optimizer.zero_grad()
            loss, _, _ = self._pruning_objective(
                X=X,
                y=y,
                w=a - tau * b,
                w_bar=w_bar,
                lam=lam,
                reg=reg,
                transport=transport,
                PI=PI,
            )
            loss.backward()
            optimizer.step()

            # print(f"\t optimized tau = {tau.data}")

            if tau.grad is None:
                raise Exception("Sorry, grads seem to be None")
            # Stop the loop when gradient is close to 0
            if np.abs(tau.grad) < eps or tau >= tau_c.to("cpu") or tau <= 0:
                break

        tau.data = tau.data.clamp(min=0, max=tau_c)
        return tau.data

    def _hard_threshold(self, x, k):
        # Set all but the largest k elements in x to zero
        if k < 1:
            return torch.zeros(len(x))
        weights = x.clone()
        threshold = weights.abs().flatten().kthvalue(int(weights.numel() - k + 1), dim=-1)[0]
        weights[weights.abs() < threshold] = 0
        return weights

    def _set_weights(self, weight_updates, module_param_indices_list, set_mask):
        for idx, module in enumerate(self._modules):
            weight = weight_updates[
                module_param_indices_list[idx] : module_param_indices_list[idx + 1]
            ]
            weight = weight.view(module.weight.shape)

            mask = (weight != 0).float()
            if set_mask:
                module.weight_mask = mask

            with torch.no_grad():
                module.weight.data = module.weight.data * mask

    def _pruning_objective(self, X, y, w, w_bar, lam, reg, transport, PI=None):
        n = X.shape[0]
        w = w.to(X.device)
        lam_torch = torch.tensor(lam, device=w.device)

        if not transport:
            PI, M = torch.eye(n) * 1 / n, None
            Q = (1 / 2) * torch.linalg.norm(y - X @ w) ** 2 + (
                n / 2
            ) * lam_torch * torch.linalg.norm(w - w_bar) ** 2
        else:
            if PI is None:
                PI, M = self._get_transportation_plan(
                    grads=X, w=w, w_target=w_bar, reg=reg, transport=transport
                )
            else:
                M = (
                    torch.cdist(
                        (X @ w_bar).reshape(n, 1),
                        (X @ w).reshape(n, 1),
                        p=2,
                    )
                    ** 2
                )
            ot_dist = torch.sum(PI * M.to(PI.device))
            # print('\tScaled OT distance', ot_dist*n)
            # print('\tSq Euclidean distance', torch.linalg.norm(X @ w_bar - X @ w)**2)
            Q = (n / 2) * ot_dist + (n / 2) * lam_torch * torch.linalg.norm(
                w - w_bar
            ) ** 2

        return Q, PI, M

    def _get_transportation_cost(self, grads, w, w_target):
        original_distr = grads @ w_target
        embedded_distr = grads @ w

        n = grads.shape[0]

        original_distr = original_distr.detach().cpu().numpy()
        embedded_distr = embedded_distr.detach().cpu().numpy()

        original_distr = (original_distr)
        embedded_distr = (embedded_distr)

        # Compute the cost matrix (squared Euclidean distance) between original_distr and embedded_distr
        M = ot.dist(
            original_distr.reshape(n, 1),
            embedded_distr.reshape(n, 1),
            metric="sqeuclidean",
        )

        return M

    def _get_transportation_plan(self, grads, w, w_target, reg, transport):
        w = w.to(grads.device)
        n = len(grads)
        if not transport:
            return torch.eye(n) * 1 / n, None

        M = self._get_transportation_cost(grads=grads, w=w, w_target=w_target)

        original_distr_mass = [1 / n for i in range(n)]
        embedded_distr_mass = [1 / n for i in range(n)]

        # PI = np.eye(n) / n
        # PI = ot.emd(original_distr_mass, embedded_distr_mass, M)
        PI = ot.bregman.sinkhorn(
            original_distr_mass, embedded_distr_mass, M, reg=reg, numItermax=5000
        )

        return torch.from_numpy(PI).float().to(w.device), torch.from_numpy(
            M
        ).float().to(w.device)

    def _get_weight_update(
        self,
        grads,
        target_weights,
        lam,
        transport,
        reg,
        dset,
        subset_inds,
        device,
        num_workers,
        module_param_indices_list,
        sparsity,
        pruning_stage,
    ):
        n = len(grads)
        params_num = grads.shape[1]
        lam_torch = torch.tensor(lam, device=device)
        n_torch = torch.tensor(n, device=device)
        w, _ = self._get_weights()
        w_bar = copy(target_weights)
        w = w.to(device)
        w_bar = w_bar.to(device)

        print(f"\t reg={reg}")
        non_zero_params_num = int(params_num * (1 - sparsity))
        grads, _, _, _ = self._compute_wgH(
            dset, subset_inds, device, num_workers, debug=False
        )

        X = grads.to(device)
        y = X @ w_bar

        print("\t w", w)
        print("\t w_bar", w_bar)

        Q, PI, M = self._pruning_objective(
            X=X,
            y=y,
            w=w,
            w_bar=w_bar,
            lam=lam,
            reg=reg,
            transport=transport,
        )
        print(f"\t Objective function value: {Q}")

        if not transport:
            print("\t Perform no transport update")
            delta_Qw = X.T @ (X @ w - y) + n_torch * lam_torch * (w - w_bar)
            print(f"\t delta_Qw={delta_Qw}")
            tau_c = self.__find_tau_c(a=w, b=delta_Qw, k=non_zero_params_num)
        else:
            print("\t Perform optimal transport update")
            delta_Qw = (X.T @ PI @ (X @ w - y) + lam_torch * (w - w_bar)) * n_torch

            # delta_Qw_part_1 = torch.zeros_like(w.reshape(len(w),1))
            # delta_Qw_part_2 = lam_torch * (w - w_bar)
            # for i in range(n):
            #     inner_sum = PI[i, :].unsqueeze(0) @ (X[i, :].unsqueeze(0) * w.reshape(len(w),1).T - X * w_bar.reshape(len(w_bar),1).T)
            #     delta_Qw_part_1 += (X[i, :].unsqueeze(1) * inner_sum.T)
            # delta_Qw_part_1 = delta_Qw_part_1.reshape(-1)
            # delta_Qw = (delta_Qw_part_1 + delta_Qw_part_2) 
            print(f"\t delta_Qw={delta_Qw}")
            tau_c = self.__find_tau_c(a=w, b=delta_Qw, k=non_zero_params_num)

        print(f"\t tau_c = {tau_c}")

        tau_m = self.__find_minimizing_tau(
            a=w,
            b=delta_Qw,
            tau_c=tau_c,
            k=non_zero_params_num,
            X=X,
            y=y,
            w_bar=w_bar,
            lam=lam,
            reg=reg,
            transport=transport,
            PI=PI,
        )
        print(f"\t tau_m = {tau_m}")
        if tau_m < tau_c.to("cpu"):
            tau = tau_m
        else:
            print("\t tau_m >= tau_c, and we optimize tau with gamma")
            gamma = 1.05
            tau = tau_c
            Q_best, _, _ = self._pruning_objective(
                X=X,
                y=y,
                w=w - tau * delta_Qw,
                w_bar=w_bar,
                lam=lam,
                reg=reg,
                transport=transport,
                PI=PI,
            )
            Q_gamma_tau, _, _ = self._pruning_objective(
                X=X,
                y=y,
                w=w - gamma * tau * delta_Qw,
                w_bar=w_bar,
                lam=lam,
                reg=reg,
                transport=transport,
                PI=PI,
            )
            while Q_best > Q_gamma_tau:
                Q_best = Q_gamma_tau
                tau = gamma * tau
                Q_gamma_tau, _, _ = self._pruning_objective(
                    X=X,
                    y=y,
                    w=w - gamma * tau * delta_Qw,
                    w_bar=w_bar,
                    lam=lam,
                    reg=reg,
                    transport=transport,
                    PI=PI,
                )
                # print(f"\t tau={tau} with gamma={gamma}")
        print(f"\t tau = {tau}")

        tau = tau.to(w.device)
        w_new = w - tau * delta_Qw
        print(f"\t tau*delta_Qw = {delta_Qw}")
        print(f"\t w_new = {w_new}")

        print(f"\t Non-zero value num of w_{pruning_stage}:", (w != 0).sum())
        print(f"\t w_{pruning_stage} = {w}")
        # self._target_weights = copy(w)

        w = self._hard_threshold(w_new, non_zero_params_num)
        print(f"\t Non-zero value num of w_{pruning_stage+1}", (w != 0).sum())
        print(f"\t w_{pruning_stage+1} = {w}")

        print("\t Model weights updated")

        self._set_weights(
            weight_updates=w,
            module_param_indices_list=module_param_indices_list,
            set_mask=True,
        )

        if self.args.ot:
            np.savetxt("X.csv", (X).detach().cpu().numpy(), delimiter=",")
            np.savetxt("w.csv", (w).detach().cpu().numpy(), delimiter=",")
            np.savetxt("w_bar.csv", (w_bar).detach().cpu().numpy(), delimiter=",")
            np.savetxt("PI.csv", PI.detach().cpu().numpy(), delimiter=",")


            _, indices_Xwbar = torch.sort(X @ w_bar)
            _, indices_Xw = torch.sort(X @ w)
            sorted_PI = PI[indices_Xwbar]
            sorted_PI = sorted_PI[:, indices_Xw]
            plt.imshow(sorted_PI.detach().cpu().numpy())
            plt.savefig("transportation_plan.png", format="png")
            # np.savetxt("M.csv", M, delimiter=",")

    def on_epoch_begin(
        self, dset, subset_inds, device, num_workers, epoch_num, **kwargs
    ):
        meta = {}
        if self._pruner_not_active(epoch_num):
            print("Pruner is not ACTIVEEEE yaa!")
            self._target_weights, self._original_mask = self._get_weights()
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

        pruning_stage = (epoch_num - self._start) // self._freq
        total_stages = (self._end - self._start) // self._freq
        print(f"PRUNING STAGE {pruning_stage}")
        sparsity = (
            pruning_stage / total_stages * self._target_sparsity
        )  # linear increasing sparsity
        # sparsity = (
        #     self._target_sparsity
        #     + (self._initial_sparsity - self._target_sparsity)
        #     * (1 - (pruning_stage-1) / (total_stages-1)) ** 3
        # ) # cubic increasing sparsity

        print(f"Sparsity={sparsity}")
        self._get_weight_update(
            grads=grads,
            target_weights=self._target_weights,
            lam=0,
            transport=self.args.ot,
            reg=1.0,
            dset=dset,
            subset_inds=subset_inds,
            device=device,
            num_workers=num_workers,
            module_param_indices_list=module_param_indices_list,
            sparsity=sparsity,
            pruning_stage=pruning_stage,
        )

        return True, meta

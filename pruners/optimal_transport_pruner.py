import os
import math
import numpy as np
import pickle
import matplotlib.pyplot as plt

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


class OptimalTransportPruner(GradualPruner):
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

    def _compute_sample_grad_weight(self, loss):

        ys = loss
        params = []
        for module in self._modules:
            for name, param in module.named_parameters():
                print(
                    "name is {} and shape of param is {} \n".format(name, param.shape)
                )

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

        # if self.args.dump_grads_mat:
        #    self._all_grads.append(grads)

        self._num_params = len(grads)

        gTw = params.T @ grads
        return grads, gTw

    def _get_weights(self):
        weights = []

        for idx, module in enumerate(self._modules):
            assert self._weight_only
            weights.append(module.weight.data.flatten())
        weights = torch.cat(weights).to(module.weight.device)
        return weights

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

        gTw = params.T @ grads

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

    @staticmethod
    def _get_param_stat(param, param_mask, fisher_inv_diag, param_idx):
        if param is None or param_mask is None:
            return None
        # w_i **2 x ((F)^-1)_ii,
        inv_fisher_diag_entry = fisher_inv_diag[
            param_idx : param_idx + param.numel()
        ].view_as(param)
        inv_fisher_diag_entry = inv_fisher_diag_entry.to(param.device)
        print(
            "mean value of statistic without eps = {} is ".format(1e-10),
            torch.mean((param**2) / inv_fisher_diag_entry),
        )
        print(
            "std value of statistic without eps = {} is ".format(1e-10),
            torch.std((param**2) / inv_fisher_diag_entry),
        )
        return ((param**2) / (inv_fisher_diag_entry + 1e-10) + 1e-10) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_outer_products_efficient_v1(self, mat, vec, num_parts=2):
        piece = int(math.ceil(len(vec) / num_parts))
        vec_len = len(vec)
        for i in range(num_parts):
            for j in range(num_parts):
                mat[
                    i * piece : min((i + 1) * piece, vec_len),
                    j * piece : min((j + 1) * piece, vec_len),
                ].add_(
                    torch.ger(
                        vec[i * piece : min((i + 1) * piece, vec_len)],
                        vec[j * piece : min((j + 1) * piece, vec_len)],
                    )
                )

    # save the fisher inv. it is the same for different target sparsity when seed is fixed
    def _dump_fisher_inv(self):
        # import pdb;pdb.set_trace()
        path = os.path.join(self.args.fisher_inv_path, f"fisher_inv.pkl")
        if not os.path.exists(path):
            fisher_inv_dict = {"f_inv": self._fisher_inv}
            # import pdb;pdb.set_trace()
            _dump(path, fisher_inv_dict)

    def _load_fisher_inv(self):
        path = os.path.join(self.args.fisher_inv_path, f"fisher_inv.pkl")
        if os.path.exists(path):
            # import pdb;pdb.set_trace()
            data = _load(path)
            self._fisher_inv = data["f_inv"]
            diag = data["f_inv"].diagonal()
            self._fisher_inv_diag = diag
            self._old_weights = self._get_weights()

    def _dump_all_grads(self):
        path = os.path.join(self.args.fisher_inv_path, f"all_grads.pkl")
        if not os.path.exists(path):
            _dump(path, {"all_grads": self._all_grads})

    def _load_all_grads(self, device):
        path = os.path.join(self.args.fisher_inv_path, f"all_grads.pkl")
        if os.path.exists(path):
            data = _load(path)
            # self._all_grads = data['all_grads'].to(device)
            self._all_grads = data["all_grads"]
            # if not self.args.offload_grads:
            #    self._all_grads = self._all_grads.to(device)
            self._num_params = self._all_grads.shape[1]
            # self._goal = self.args.fisher_subsample_size

    def _compute_wgH(self, dset, subset_inds, device, num_workers, debug=False):
        st_time = time.perf_counter()
        self._model = self._model.to(device)

        print("in woodfisher: len of subset_inds is ", len(subset_inds))

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
                loss = criterion(output, target, reduction="none")
            except:
                import pdb

                pdb.set_trace()
            # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
            # which is just a practical heuristic to utilize more datapoints

            ## compute grads, XX, yy
            g, _, gTw, w = self._compute_sample_fisher(loss, return_outer_product=False)
            Gs.append(g[None, :].detach().cpu().numpy())
            # GTWs.append(gTw[None,None].detach().cpu().numpy())
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
        grads = torch.cat(Gs, 0)
        wTgs = torch.cat(GTWs, 0)
        # FF = FF / self.args.fisher_subsample_size
        print(
            "# of examples done {} and the goal (#outer products) is {}".format(
                num_samples, goal
            )
        )
        print("# of batches done {}".format(num_batches))

        end_time = time.perf_counter()
        print(
            "Time taken to compute fisher inverse with woodburry is {} seconds".format(
                str(end_time - st_time)
            )
        )

        return grads, GTWs, w, ff

    def on_epoch_begin(
        self, dset, subset_inds, device, num_workers, epoch_num, **kwargs
    ):
        meta = {}
        if self._pruner_not_active(epoch_num):
            print("Pruner is not ACTIVEEEE yaa!")
            return False, {}

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        # reinit params if they were deleted during gradual pruning
        if not hasattr(self, "_all_grads"):
            self._all_grads = []
        if not hasattr(self, "_param_stats"):
            self._param_stats = []

        grads, wTgs, weights, fisher_matrix = self._compute_wgH(
            dset, subset_inds, device, num_workers, debug=False
        )

        H_approx = grads.T @ grads
        print(H_approx.shape)

        meta["prune_direction"] = []
        meta["original_param"] = []
        meta["mask_previous"] = []
        meta["mask_overall"] = []
        meta["quad_term"] = []
        meta["inspect_dic"] = {}

        for idx, module in enumerate(self._modules):
            pass

        return True, meta

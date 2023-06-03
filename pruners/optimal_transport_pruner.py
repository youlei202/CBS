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
                loss = criterion(output, target, reduction="mean")
            except:
                import pdb

                pdb.set_trace()
            # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
            # which is just a practical heuristic to utilize more datapoints

            ## compute grads, XX, yy
            g, _, gTw, w = self._compute_sample_fisher(loss, return_outer_product=False)
            Gs.append(torch.Tensor(g[None, :].detach().cpu().numpy()))
            GTWs.append(torch.Tensor(gTw[None, None].detach().cpu().numpy()))
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
        wTgs = torch.cat(tuple(GTWs), 0)
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
        threshold = np.partition(np.abs(x).flatten(), -k)[-k]
        x[np.abs(x) < threshold] = 0
        return x

    def _get_weight_update(self, w_target, k, X, y, tau=0.2, lam=0.5):
        n = len(X)
        w = self._get_weights()

        w_new = w - tau * (X.T @ (X @ w - y) + n * lam * (w - w_target))

        return self._hard_threshold(w_new, k) - w

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

        inv_timer = Timer("Fisher_inv")
        wf_timer = Timer("WF")
        self.grad_timer = Timer("Gradients")

        self._fisher_inv = None
        #############################################################
        # Step 1. Computer full fisher inverse via woodburry
        self._load_all_grads(device)
        self._load_fisher_inv()
        if len(self._all_grads) == 0 and self._fisher_inv is None:
            print("Computing woodburry fisher inverse - Lei")
            inv_timer.start()
            pruning_time = self._compute_woodburry_fisher_inverse(
                dset, subset_inds, device, num_workers
            )
            inv_timer.stop()
            self._all_grads = torch.stack(self._all_grads)
            self._dump_all_grads()
            self._dump_fisher_inv()
        else:
            print("Load fisher_inv from disk, the computation time on it is 0 sec")
            pruning_time = 0
        # import pdb;pdb.set_trace()

        # if self.args.dump_grads_mat:
        #    self._all_grads = torch.stack(self._all_grads)
        #    dump_tensor_to_mat(self._all_grads, self.args.run_dir, 'gradsU.mat', 'U', transpose=True)
        #    del self._all_grads

        assert self._num_params == self._fisher_inv_diag.shape[0]
        self._param_idx = 0

        flat_pruned_weights_list = []
        flat_module_weights_list = []
        module_shapes_list = []
        module_param_indices_list = []
        prune_masks = []
        past_weight_masks = []

        wf_timer.start()
        #############################################################
        # Step 2. Get param stats and either jointly or independently create sparsification masks!

        # Step 2.1: If independent, then compute param stats and masks at once.
        # Else, save param stats for all modules in an array

        for idx, module in enumerate(self._modules):
            # print(f'module is {module}')
            level = self._required_sparsity(epoch_num)
            # multiplying by the current mask makes the corresponding statistic
            # of those weights zero and keeps them removed.

            past_weight_masks.append(module.weight_mask)
            module_param_indices_list.append(self._param_idx)
            assert self._weight_only
            module_shapes_list.append(module.weight.shape)

            w_stat = self._get_param_stat(
                module.weight,
                module.weight_mask,
                self._fisher_inv_diag,
                self._param_idx,
            )
            self._param_idx += module.weight.numel()

            if self.args.woodburry_joint_sparsify:
                self._param_stats.append(w_stat.flatten())

            if module.bias is not None and not self._weight_only:
                print("sparsifying bias as well")
                b_stat = self._get_param_stat(
                    module.bias,
                    module.bias_mask,
                    self._fisher_inv_diag,
                    self._param_idx,
                )
                self._param_idx += module.bias.numel()
                if self.args.woodburry_joint_sparsify:
                    self._param_stats.append(b_stat.flatten())

            if not self.args.woodburry_joint_sparsify:
                module.weight_mask, module.bias_mask = self._get_pruning_mask(
                    w_stat, level
                ), self._get_pruning_mask(None if self._weight_only else b_stat, level)
        # Step 2.2 For the joint sparsification case, build a global param mask
        # based on the param stats saved across various modules!
        if self.args.woodburry_joint_sparsify:
            level = self._required_sparsity(epoch_num)
            global_param_mask = self._get_pruning_mask(
                flatten_tensor_list(self._param_stats), level
            )
            logging.info(
                f"shape of global param mask is {list(global_param_mask.shape)}"
            )
            mask_dict = {"mask": global_param_mask, "sparsity": level}
            mask_dict_path = os.path.join(
                self.args.fisher_inv_path, f"global_mask_sparsity{level}.pkl"
            )
            with open(mask_dict_path, "wb") as f_:
                pickle.dump(mask_dict, f_, pickle.HIGHEST_PROTOCOL)

            _param_count = 0
            for idx, module in enumerate(self._modules):
                module.weight_mask = global_param_mask[
                    _param_count : _param_count + module.weight.numel()
                ].view_as(module.weight)

                _param_count += module.weight.numel()

                if module.bias is not None and not self._weight_only:
                    module.bias_mask = global_param_mask[
                        _param_count + module.bias.numel()
                    ].view_as(module.bias)
                    _param_count += module.bias.numel()
                else:
                    module.bias_mask = None

            # del self._param_stats
            del global_param_mask

        # pruning_time += timer.stop('calculate the mask in woodfisher')
        #############################################################
        # Step 3. Now that sparsification masks have been computed whether jointly or independently,
        # put them together in a list, and apply the requisite OBS update to other remaining weights

        for idx, module in enumerate(self._modules):
            assert self._weight_only
            pruned_weights = past_weight_masks[idx] - module.weight_mask
            prune_mask = past_weight_masks[idx] > module.weight_mask
            prune_masks.append(prune_mask)
            # print(f'pruned_weights are {pruned_weights}')
            pruned_weights = pruned_weights.flatten().float()
            flat_pruned_weights_list.append(pruned_weights)
            flat_module_weights_list.append(module.weight.flatten())

        module_param_indices_list.append(self._param_idx)

        flat_pruned_weights_list = flatten_tensor_list(flat_pruned_weights_list)
        flat_module_weights_list = flatten_tensor_list(flat_module_weights_list)

        # compute the weight update across all modules
        scaled_basis_vector = self._get_pruned_wts_scaled_basis(
            flat_pruned_weights_list, flat_module_weights_list
        )

        grads, wTgs, weights, fisher_matrix = self._compute_wgH(
            dset, subset_inds, device, num_workers, debug=False
        )

        st_time = time.perf_counter()
        H_approx = grads.T @ grads
        end_time = time.perf_counter()
        print(
            "Time taken to compute the Hessian Approximation is {} seconds".format(
                str(end_time - st_time)
            )
        )
        print("The shape of the Hessian Approximation:", H_approx.shape)
        weights = self._get_weights()
        weight_updates = self._get_weight_update(
            w_target=self._old_weights.detach().numpy(),
            k=int(len(weights) * (1 - self._target_sparsity)),
            X=grads,
            y=grads @ self._old_weights.detach().numpy(),
        )
        # pruning_time += timer.stop('calculate weight_updates')
        if self._prune_direction:
            meta["prune_direction"] = []
            meta["original_param"] = []
            meta["mask_previous"] = []
            meta["mask_overall"] = []
            meta["quad_term"] = []

        # now apply the respective module wise weight update
        for idx, module in enumerate(self._modules):
            weight_update = weight_updates[
                module_param_indices_list[idx] : module_param_indices_list[idx + 1]
            ]
            cache_weight_update_shape = weight_update.shape
            weight_update = weight_update.view_as(module.weight)

            ## In case that \delta_w of the pruned weights might not be 0...
            if self._zero_after_prune:
                # This flag is used in case when analyze the loss approximation due to pruning.

                # It's purpose is to make those active in the prune_mask to be 0 weight
                # since later module.weight will be added to the weight_update.
                # because of the effect of separate OBS parameter readjustments terms in practice,
                # weight update by itself won't result in weight 0 - at the places pruned.

                # However, for most of the usage, this does not matter, as we multiply weight matrices
                # by the mask when considering pruning or retraining anyways!

                weight_update[prune_masks[idx]] = (
                    -1 * module.weight.data[prune_masks[idx]]
                )

            print(
                f"for param {idx}: norm of weight is {torch.norm(module.weight).item()}"
            )
            print(
                f"for param {idx}: norm of weight update is {torch.norm(weight_update).item()}"
            )

            if self.args.local_quadratic:
                weight_update = weight_update.view(cache_weight_update_shape)
                # (e^T F^-1 e)/2 which is what comes out,
                # when you plug in weight_update to quadratic term
                meta["quad_term"].append(
                    torch.dot(weight_update, scaled_basis_vector) / 2
                )
                weight_update = weight_update.view_as(module.weight)
                print("quad term comes out to be", meta["quad_term"])

            ## TODO: prune_direction seems for analysis of the change of pruning mask
            if self._prune_direction:
                meta["prune_direction"].append(weight_update)
                meta["original_param"].append(module.weight.data.clone())
                print("idx is ", idx)
                # print(flat_pruned_weights_list)
                # dirty hack that works when only 1 layer
                meta["mask_previous"].append(
                    module.weight_mask
                    + flat_pruned_weights_list.view(module_shapes_list[idx]).type(
                        module.weight_mask.dtype
                    )
                )
                meta["mask_overall"].append(module.weight_mask)

            # print('weight before is ', module.weight)
            if not self.args.not_update_weights:
                with torch.no_grad():
                    module.weight[:] = module.weight.data + weight_update
            # print('weight after is ', module.weight)
            print(
                f"for param {idx} after update: norm of weight is {torch.norm(module.weight).item()}"
            )

            # print(f'weights in parameter {idx} after pruning (only for pruned) are ', module.weight[prune_masks[idx]])
            if self._prune_direction:
                print(
                    f"weights in meta[original_param][{idx}] after pruning (only for pruned) are ",
                    meta["original_param"][idx],
                )

        wf_timer.stop()
        inv_timer.info("Fisher Inverse (with Gradients)")
        wf_timer.info("WoodFisher (with Gradients and fisher_inv)")
        self.grad_timer.info("Gradients")
        wf_time = wf_timer.sum - inv_timer.sum
        print(
            f"Time taken to Fisher_inv (without gradients) is {inv_timer.sum - self.grad_timer.sum:.2f} seconds"
        )
        print(
            f"Time taken to WF (without gradients and fisher_inv) is {wf_time:.2f} seconds"
        )

        self._release_grads()
        new_w = weight_updates + self._old_weights
        score = flatten_tensor_list(self._param_stats)
        analyze_new_weights(
            new_w=new_w,
            old_w=self._old_weights,
            update_w=weight_updates,
            score=score,
            info=f"damp_{self.args.fisher_damp:.2E}",
        )

        # check if all the params whose fisher inverse was computed their value has been taken
        print(
            f"param_idx is {self._param_idx} and fisher_inv_shape[0] is {self._fisher_inv_diag.shape[0]} \n"
        )
        assert self._param_idx == self._fisher_inv_diag.shape[0]

        del self._fisher_inv

        if self._inspect_inv:
            meta["inspect_dic"] = self.inspect_dic
        return True, meta

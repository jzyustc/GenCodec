"""SOAP optimizer — Shampoo with Adam Pre-conditioning.

Reference: https://github.com/nikhilvyas/SOAP (MIT).
Paper: "SOAP: Improving and Stabilizing Shampoo using Adam" (Vyas et al., 2024).

Single-file implementation. Drop-in replacement for AdamW for most cases.
"""

# MIT License
#
# Copyright (c) 2024 Nikhil Vyas
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from itertools import chain

import torch
from torch.optim.optimizer import Optimizer


class SOAP(Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Args:
        params: iterable of parameters or named parameter groups
        lr (float): learning rate
        betas (Tuple[float, float]): coefficients used for computing running
            averages of gradient and its square (default: (0.95, 0.95))
        shampoo_beta (float): If >= 0, use this beta for the preconditioner
            running average; if < 0, use betas[1]. (default: -1)
        eps (float): term added to the denominator to improve numerical
            stability (default: 1e-8)
        weight_decay (float): weight decay (L2 penalty) (default: 0.01)
        precondition_frequency (int): how often to update the preconditioner
            (default: 10)
        max_precond_dim (int): maximum dimension of the preconditioner — any
            tensor dim larger than this is left unprojected (default: 10000)
        merge_dims (bool): merge consecutive small dims to reduce
            preconditioner count (default: False)
        precondition_1d (bool): apply preconditioning to 1d parameters
            (default: False)
        normalize_grads (bool): normalize gradients to unit RMS before applying
            (default: False)
        data_format (str): "channels_first" or "channels_last" — affects how
            conv weights' dims are ordered for merging (default: "channels_first")
        correct_bias (bool): apply Adam bias correction (default: True)
    """

    def __init__(
        self,
        params,
        lr=3e-3,
        betas=(0.95, 0.95),
        shampoo_beta=-1,
        eps=1e-8,
        weight_decay=0.01,
        precondition_frequency=10,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
        normalize_grads=False,
        data_format="channels_first",
        correct_bias=True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        """Merge consecutive small dims of grad until each is > max_precond_dim
        (or the merged dim would exceed it)."""
        assert self._data_format in ("channels_first", "channels_last")
        if self._data_format == "channels_last" and grad.dim() == 4:
            shape = grad.shape
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape

        new_shape = []
        curr_shape = 1
        for sh in shape:
            tmp = curr_shape * sh
            if tmp > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = tmp
        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)
        return grad.reshape(new_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # initialise preconditioner the first time we see this tensor
                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0 else group["betas"][1],
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self.update_preconditioner(grad, state,
                                               max_precond_dim=group["max_precond_dim"],
                                               merge_dims=group["merge_dims"],
                                               precondition_1d=group["precondition_1d"])
                    continue  # first step: collect stats only

                # project grad into eigenbasis
                grad_projected = self.project(grad, state,
                                              merge_dims=group["merge_dims"],
                                              max_precond_dim=group["max_precond_dim"])
                beta1, beta2 = group["betas"]

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad_projected)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_projected, grad_projected, value=1 - beta2)

                state["step"] += 1
                step = state["step"]

                # Adam-style preconditioning in eigenbasis
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                exp_avg_projected = self.project(exp_avg, state,
                                                 merge_dims=group["merge_dims"],
                                                 max_precond_dim=group["max_precond_dim"])
                norm_grad = self.project_back(exp_avg_projected / denom, state,
                                              merge_dims=group["merge_dims"],
                                              max_precond_dim=group["max_precond_dim"])

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                # update preconditioner stats and (every N steps) eigenbasis
                self.update_preconditioner(
                    grad,
                    state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )

        return loss

    def init_preconditioner(self, grad, state, precondition_frequency=10,
                            shampoo_beta=0.95, max_precond_dim=10000,
                            precondition_1d=False, merge_dims=False):
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype)
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(
                        torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype)
                    )
        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(self, grad, state, max_precond_dim=10000,
                              merge_dims=False, precondition_1d=False):
        if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
            pass
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer = torch.tensordot(
                            new_grad,
                            new_grad,
                            dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                        )
                        state["GG"][idx].lerp_(outer, 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer = torch.tensordot(
                            grad,
                            grad,
                            dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                        )
                        state["GG"][idx].lerp_(outer, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)

    def get_orthogonal_matrix(self, mat):
        out = []
        for m in mat:
            if len(m) == 0:
                out.append([])
                continue
            try:
                _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device, dtype=m.dtype))
            except Exception:
                _, Q = torch.linalg.eigh(
                    m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device, dtype=torch.float64)
                )
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, dims=[1])
            out.append(Q)
        return out

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000, merge_dims=False):
        precond_list = state["GG"]
        orth_list = state["Q"]

        out = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if len(m) == 0:
                out.append([])
                continue
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            o = o[:, sort_idx]
            try:
                Q, _ = torch.linalg.qr(m @ o)
            except Exception:
                Q, _ = torch.linalg.qr((m.to(torch.float64) @ o.to(torch.float64)))
                Q = Q.to(m.dtype)
            out.append(Q)
        return out

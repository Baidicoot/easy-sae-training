import copy
from typing import Type, Optional, Dict, Any, List, Union

import torch
from torch.func import stack_module_state, functional_call
import torchopt


class Ensemble:
    def __init__(
        self,
        models: List[torch.nn.Module],
        optimizer_func: Type[torchopt.base.GradientTransformation],
        optimizer_kwargs: Dict[str, Any],
        model_hyperparams: Dict[str, Any],
        device: Optional[Union[str, torch.device]] = None,
        no_stacking: bool = False,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_models = len(models)
        self.params, self.buffers = stack_module_state(models)

        self.sig = copy.deepcopy(models[0]).to("meta")  #
        self.no_stacking = no_stacking

        self.optimizer_func = optimizer_func
        self.optimizer_kwargs = optimizer_kwargs

        self.optimizer = optimizer_func(**optimizer_kwargs)
        self.optim_states = torch.vmap(self.optimizer.init)(self.params)

        self.model_hyperparams = model_hyperparams

        self.init_functions()

    def init_functions(self):
        """
        Initializes functions for a unified backward pass of ensemble models.

        - `self.calc_grads(params, buffers, batch)`:
            Args:
                params, buffers (Dict[str, Any]): Tensors dictionary; first dimension is ensemble dimension.
                batch: Single data batch.
            Returns:
                grads, outputs (Dict[str, Any]): Gradients and model outputs, both with ensemble dimension first.

        - `self.update(grads, opt_state)`:
            Vectorized `update()` from `Optimizer` in `torchopt`.
            Args:
                grads, opt_state (Dict[str, Any]): Tensors dictionary; first dimension is ensemble dimension.
            Returns:
                updates, new_opt_state (Dict[str, Any]): Updated tensors and optimizer state; ensemble dimension first.
        """

        def call_single_model(params: Dict[str, Any], buffers: Dict[str, Any], batch):
            outputs = functional_call(self.sig, (params, buffers), batch)
            return outputs[0], outputs

        def calc_grads(params: Dict[str, Any], buffers: Dict[str, Any], batch):
            return torch.func.grad(call_single_model, has_aux=True)(params, buffers, batch)

        self.calc_grads = torch.vmap(calc_grads)
        self.update = torch.vmap(self.optimizer.update)

    def unstack(self, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            device = self.device

        for i in range(self.n_models):
            model = type(self.sig)(**self.model_hyperparams).to(device)
            for k, v in model.named_parameters():
                v.requires_grad_(False)
                v.copy_(self.params[k][i])
                v.requires_grad_(True)

                assert torch.allclose(v, self.params[k][i].to(device))

            for buffer_k, _ in model.named_buffers():
                setattr(model, buffer_k, self.buffers[buffer_k][i].clone())

                assert torch.allclose(getattr(model, buffer_k), self.buffers[buffer_k][i].to(device))

            yield model

    def to_device(self, device: Union[str, torch.device]):
        self.device = device

        for t in self.params.values():
            t.to(device)

        for t in self.buffers.values():
            t.to(device)

        for t in self.optim_states.values():
            t.to(device)

    def step_batch(self, minibatches, expand_dims: bool = True):
        with torch.no_grad():
            if expand_dims:
                minibatches = minibatches.expand(self.n_models, *minibatches.shape)

            grads, outputs = self.calc_grads(self.params, self.buffers, minibatches)

            updates, new_optim_states = self.update(grads, self.optim_states)

            self.optim_states = new_optim_states

            torchopt.apply_updates(self.params, updates)

            return outputs

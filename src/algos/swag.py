import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
import math
import numpy as np
from .algo import BayesianOptimizer
from .opt_util import apply_lr

class SwagModel(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model.cpu()
        self.update_every_batches = config.get("update_every_batches", -1)
        self.mean_samples = config.get("mean_samples", 10)
        self.deviation_samples = config.get("deviation_samples", 10)
        assert self.mean_samples >= self.deviation_samples
        self.start_epoch = config.get("start_epoch", 0)
        self.use_lr_cycles = config.get("use_lr_cycles", False)
        self.max_lr = config.get("max_lr", 0.005)
        self.min_lr = config.get("min_lr", 0.001)
        self.use_low_rank_cov = config.get("use_low_rank_cov", True)

        self.weights = parameters_to_vector(self.model.parameters()).cpu()
        self.sq_weights = self.weights**2
        self.updates = 0
        self.deviations = torch.zeros((self.weights.shape[0], self.deviation_samples))
        self.param_dist_valid = False
        self.param_dist = None
        self.batches_since_swag_start = 0

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "model": self.model.state_dict(destination, prefix, keep_vars),
            "updates": self.updates,
            "batches_since_swag_start": self.batches_since_swag_start,
            "weights": self.weights,
            "sq_weights": self.sq_weights,
            "deviations": self.deviations
        }

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.updates = state["updates"]
        self.batches_since_swag_start = state["batches_since_swag_start"]
        self.weights = state["weights"]
        self.sq_weights = state["sq_weights"]
        self.deviations = state["deviations"]
        self.param_dist_valid = False

    def train_model(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, log, scheduler_factory=None, use_amp=False, report_every_epochs=1):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())
        scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if self.update_every_batches == -1:
            self.update_every_batches = np.ceil(len(loader) * (epochs - self.start_epoch) / self.mean_samples)

        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for batch_idx, (data, target, *_) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", enabled=use_amp):
                    output = self.model(data)
                    loss = loss_fn(output, target)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.cpu().item()
                self.swag_update(epoch, batch_idx, optimizer)
            epoch_loss /= len(loader)

            if scheduler is not None:
                scheduler.step()

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                log.info(f"Epoch {epoch}: loss {epoch_loss}")
                self.report_status(log)
        if report_every_epochs >= 0:
            log.info(f"Final loss {epoch_loss}")
            self.report_status(log)

    def infer(self, input, samples):
        if samples <= 0:
            return [self.model(input)]
        
        old_params = parameters_to_vector(self.model.parameters())
        self.update_param_dist(input.device)
        outputs = []
        for _ in range(samples):
            weight_sample = self.param_dist.sample().to(input.device)
            vector_to_parameters(weight_sample, self.model.parameters())
            outputs.append(self.model(input))
        vector_to_parameters(old_params, self.model.parameters())
        return torch.stack(outputs)

    @property
    def mean(self):
        return self.weights
    
    def update_param_dist(self, device):
        if not self.param_dist_valid:
            diag = 0.5 * (torch.relu(self.sq_weights - self.weights**2) + 1e-6) # Adding 1e-6 for numerical stability
            diag = diag
            cov_factor = self.deviations / math.sqrt(2 * (self.deviation_samples - 1))
            cov_factor = cov_factor
            #self.param_dist = torch.distributions.MultivariateNormal(self.mean, cov_factor, diag)
            if self.use_low_rank_cov:
                self.param_dist = torch.distributions.LowRankMultivariateNormal(self.mean, cov_factor, diag)
            else:
                self.param_dist = torch.distributions.Normal(self.mean, torch.sqrt(2 * diag))
            self.param_dist_valid = True

    def swag_update(self, epoch, batch_idx, optimizer):
        if epoch >= self.start_epoch:
            self.batches_since_swag_start += 1

            if self.batches_since_swag_start % self.update_every_batches == 0:
                self.updates += 1
                params = parameters_to_vector(self.model.parameters()).cpu()
                self.weights = (self.updates * self.weights + params) / (self.updates + 1)
                self.sq_weights = (self.updates * self.sq_weights + params**2) / (self.updates + 1)
                self.deviations = torch.roll(self.deviations, -1, 1)
                self.deviations[:,-1] = params - self.weights
                self.param_dist_valid = False

            if self.use_lr_cycles:
                t = 1 - (self.batches_since_swag_start % self.update_every_batches) / self.update_every_batches
                self.lr = t * (self.max_lr - self.min_lr) + self.min_lr
                for g in optimizer.param_groups:
                    g["lr"] = self.lr

    def report_status(self, log):
        log.info(f"SWAG: Collected {np.minimum(self.updates, self.deviation_samples)} out of {self.deviation_samples} deviation samples and {self.updates} out of {self.mean_samples} parameter samples")

class SwagOptimizer(BayesianOptimizer):
    '''
        Stochastic Weight Averaging-Gaussian
    '''

    def __init__(self, params, base_optimizer, update_interval, start_epoch=0, deviation_samples=30):
        super().__init__(params, {})

        self.start_epoch = start_epoch
        self.update_interval = math.floor(update_interval)
        self.param_dist = None
        self.deviation_samples = deviation_samples

        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["original_param"] = param.data.detach().clone()

        self.state["__base_optimizer"] = base_optimizer
        self.state["__epoch"] = 0
        self.state["__steps_since_swag_start"] = 0
        self.state["__updates"] = 0
        self.state["__mean"] = parameters_to_vector(self._params()).cpu()
        self.state["__sq_weights"] = self.state["__mean"]**2
        self.state["__deviations"] = torch.zeros((self.state["__mean"].shape[0], self.deviation_samples))
        self.state["__params_dirty"] = False

    def step(self, forward_closure, backward_closure, grad_scaler=None):
        self._restore_original_params()
        self.state["__base_optimizer"].zero_grad()

        loss = forward_closure()
        backward_closure(loss)

        if grad_scaler is not None:
            grad_scaler.step(self.state["__base_optimizer"])
        else:
            self.state["__base_optimizer"].step()

        self._swag_update()

        return loss

    def sample_parameters(self):
        self._update_param_dist()
        self._save_original_params()
        self.state["__params_dirty"] = True
        new_params = self.param_dist.sample().to(self._params_device())
        vector_to_parameters(new_params, self._params())

    def complete_epoch(self):
        self.state["__epoch"] += 1

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
    
    def get_base_optimizer(self):
        return self.state["__base_optimizer"]

        # Required as PyTorch's optimizer only casts per-param state in load_state_dict()
        # param_device = self._params_device()
        # if param_device != self.state["__mean"].device:
        #     self.state["__mean"] = self.state["__mean"].to(param_device)
        #     self.state["__sq_weights"] = self.state["__sq_weights"].to(param_device)
        #     self.state["__deviations"] = self.state["__deviations"].to(param_device)
    
    def _restore_original_params(self):
        if self.state["__params_dirty"]:
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state[param]
                    param.data = state["original_param"].detach().clone()
            self.state["__params_dirty"] = False

    def _save_original_params(self):
        if not self.state["__params_dirty"]:
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state[param]
                    state["original_param"] = param.data.detach().clone()

    def _swag_update(self):
        if self.state["__epoch"] >= self.start_epoch:
            self.state["__steps_since_swag_start"] += 1

            if self.state["__steps_since_swag_start"] % self.update_interval == 0:
                assert not self.state["__params_dirty"]
                with torch.no_grad():
                    self.state["__updates"] += 1
                    updates = self.state["__updates"]
                    params = parameters_to_vector(self._params()).cpu()
                    self.state["__mean"] = (updates * self.state["__mean"] + params) / (updates + 1)
                    self.state["__sq_weights"] = (updates * self.state["__sq_weights"] + params**2) / (updates + 1)
                    self.state["__deviations"] = torch.roll(self.state["__deviations"], -1, 1)
                    self.state["__deviations"][:,-1] = params - self.state["__mean"]
                    self.param_dist = None

    def _update_param_dist(self):
        if self.param_dist is None:
            device = self._params_device()
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False): # Disable autocast for LowRankMultivariateNormal's cholesky_solve
                    diag = 0.5 * (torch.relu(self.state["__sq_weights"].to(device).float() - self.state["__mean"].to(device).float()**2) + 1e-6) # Adding 1e-6 for numerical stability
                    cov_factor = self.state["__deviations"].to(device).float() / math.sqrt(2 * (self.deviation_samples - 1))
                    self.param_dist = torch.distributions.LowRankMultivariateNormal(self.state["__mean"].to(device).float(), cov_factor, diag)

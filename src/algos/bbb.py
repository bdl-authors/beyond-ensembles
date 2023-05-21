import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .algo import BayesianOptimizer
from .util import GaussianParameter
from .opt_util import apply_lr

class BBBModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "model": self.model.state_dict(destination, prefix, keep_vars)
        }

    def load_state_dict(self, dict):
        self.model.load_state_dict(dict["model"])

    def train_model(self, epochs, data_loss_fn, optimizer_factory, loader, batch_size, device, log, mc_samples=5, kl_rescaling=1, scheduler_factory=None, report_every_epochs=1, batch_as_full_estimator=False, components=1, use_amp=False):
        r"""
        Args:
            data_loss_fn: Must use 'mean' reduction!

            batch_as_full_estimator (bool, optional): 
                True corresponds to treating the batch data loss as an estimator for the full dataset (i.e. loss = N * mean_data_loss + kl_rescaling * kl_loss) as used by Rank-1 VI (Dusenberry et al.). 
                False corresponds to true minibatch training where the KL loss is distributed over all batches (i.e. loss = mean_data_loss + kl_rescaling * kl_loss / (B * num_b)) similar to BBB (Blundell et al. use the same formula multiplied by B)
        """
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())
        scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if not batch_as_full_estimator:
            pi = kl_rescaling / (len(loader) * batch_size)
        else:
            pi = kl_rescaling

        # kl_grads = []
        # data_grads = []
        for epoch in range(epochs):
            epoch_loss = torch.tensor(0.0, device=device)
            epoch_kl_loss = torch.tensor(0.0, device=device)
            epoch_data_loss = torch.tensor(0.0, device=device)
            for data, target, *_ in loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", enabled=use_amp):
                    kl_loss = torch.tensor(0.0, device=data.device)
                    data_loss = torch.tensor(0.0, device=data.device)
                    for _ in range(mc_samples * components):
                        output = self.model(data)
                        kl_loss += collect_kl(self.model)
                        data_loss += data_loss_fn(output, target)

                    if batch_as_full_estimator:
                        data_loss *= len(loader) * batch_size
                    loss = (pi * kl_loss + data_loss) / (mc_samples * components)

                scaler.scale(loss).backward()
                # kl_loss /= mc_samples
                # kl_loss *= pi
                # data_loss /= mc_samples
                # kl_loss.backward(retain_graph=True)
                #kl_grad = torch.cat([module.rho_grads() for module in self.model if hasattr(module, "rho_grads")])
                #kl_grads.append(kl_grad)
                # data_loss.backward()
                #data_grad = torch.cat([module.rho_grads() for module in self.model if hasattr(module, "rho_grads")])
                #data_grad -= kl_grad
                #data_grads.append(data_grad)

                #nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(optimizer)
                scaler.update()
                if not loss.isnan().any():
                    epoch_loss += loss.detach()
                    epoch_kl_loss += (pi * kl_loss / (mc_samples * components)).detach()
                    epoch_data_loss += (data_loss / (mc_samples * components)).detach()
                else:
                    log.info("Skiping batch due to nan loss")
            if batch_as_full_estimator:
                epoch_loss /= (len(loader)**2 * batch_size)
            else:
                epoch_loss /= len(loader)
                epoch_data_loss /= len(loader)
                epoch_kl_loss /= len(loader)

            if scheduler is not None:
                scheduler.step()

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                log.info(f"Epoch {epoch}: loss {epoch_loss:.4f}, data loss {epoch_data_loss:.4f}, kl loss {epoch_kl_loss:.4f}")
        if report_every_epochs >= 0:
            log.info(f"Final loss {epoch_loss}")
        #return torch.stack(kl_grads), torch.stack(data_grads)

    def infer(self, input, samples):
        self.model.eval()
        return torch.stack([self.model(input) for _ in range(samples)])

class GaussianPrior:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dist = torch.distributions.Normal(mu, sigma)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def kl_divergence(self, mu2, sigma2):
        #kl = 0.5 * (2 * torch.log(sigma2 / self.sigma) - 1 + (self.sigma / sigma2).pow(2) + ((mu2 - self.mu) / sigma2).pow(2))
        kl = 0.5 * (2 * torch.log(self.sigma / sigma2) - 1 + (sigma2 / self.sigma).pow(2) + ((self.mu - mu2) / self.sigma).pow(2))
        return kl.sum()

class MixturePrior:
    def __init__(self, pi, sigma1, sigma2, validate_args=None):
        self.pi = torch.tensor(pi)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = torch.distributions.Normal(0, sigma1, validate_args)
        self.dist2 = torch.distributions.Normal(0, sigma2, validate_args)

    def log_prob(self, value):
        prob1 = torch.log(self.pi) + torch.clamp(self.dist1.log_prob(value), -23, 0)
        prob2 = torch.log(1 - self.pi) + torch.clamp(self.dist2.log_prob(value), -23, 0)
        return torch.logaddexp(prob1, prob2)

    def kl_divergence(self, mu2, sigma2):
        return -self.log_prob(mu2).sum()

def collect_kl(model) -> torch.tensor:
    return sum(getattr(layer, "kl", 0) + collect_kl(layer) for layer in model.children())


class BBBOptimizer(BayesianOptimizer):
    '''
        Bayes By Backprop. You need to use Bayesian layers for the layers that should be treated as Bayesian.
    '''
    def __init__(self, params, base_optimizer, prior, dataset_size, mc_samples=1, kl_rescaling=1, components=1, l2_scale=0):
        defaults = {
            "prior": prior,
            "l2_scale": l2_scale
        }
        super().__init__(params, defaults)
        self.state["__base_optimizer"] = base_optimizer
        self.mc_samples = mc_samples
        self.kl_rescaling = kl_rescaling
        self.components = components
        self.dataset_size = dataset_size

    def step(self, forward_closure, backward_closure, grad_scaler=None):
        self.state["__base_optimizer"].zero_grad()
        
        total_data_loss = None
        for _ in range(self.mc_samples):
            if total_data_loss is None:
                total_data_loss = forward_closure()
            else:
                total_data_loss += forward_closure()

        # Collect KL loss & reg only once
        total_kl_loss = torch.tensor(0.0, device=self._params_device())
        for group in self.param_groups:
            for param in group["params"]:
                if hasattr(param, "get_parameter_kl"):
                    total_kl_loss += param.get_parameter_kl(group["prior"])
                elif not getattr(param, "_is_gaussian_mean", False) and not getattr(param, "_is_gaussian_rho", False):
                    total_kl_loss += group["l2_scale"] / 2 * param.pow(2).sum()

        pi = self.kl_rescaling / self.dataset_size
        # don't divide the kl loss by the mc sample count as we have only been collection it once
        loss = pi * total_kl_loss + total_data_loss / (self.mc_samples * self.components)
        if not loss.isnan().any():
            backward_closure(loss)

            if grad_scaler is not None:
                grad_scaler.step(self.state["__base_optimizer"])
            else:
                self.state["__base_optimizer"].step()

        return loss


    def sample_parameters(self):
        '''
            The parameters sample themself
        '''
        pass

    def get_base_optimizer(self):
        return self.state["__base_optimizer"]

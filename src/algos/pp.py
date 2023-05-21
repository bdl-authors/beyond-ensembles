import torch
import torch.nn as nn
from .algo import BayesianOptimizer
from .opt_util import apply_lr

class MAP(nn.Module):
    def __init__(self, model_builder):
        super().__init__()
        #self.model = generate_model(layers)
        self.model = model_builder()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "model": self.model.state_dict(destination, prefix, keep_vars),
        }

    def load_state_dict(self, dict):
        self.model.load_state_dict(dict["model"])

    def train_model(self, epochs, loss_fn, optimizer_factory, loader, batch_size, device, log, mc_samples=1, scheduler_factory=None, report_every_epochs=1, early_stopping=None, use_amp=False):
        self.model.to(device)
        self.model.train()
        optimizer = optimizer_factory(self.model.parameters())
        scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        epoch_loss = 0
        for epoch in range(epochs):
            #cw_logging.getLogger().info(f"Epoch {epoch}")
            epoch_loss = self.run_epoch(loss_fn, loader, optimizer, batch_size, device, log, mc_samples=mc_samples, use_amp=use_amp, grad_scaler=scaler)

            if scheduler is not None:
                scheduler.step()

            if report_every_epochs > 0 and epoch % report_every_epochs == 0:
                log.info(f"Epoch {epoch}: loss {epoch_loss}")

            if early_stopping is not None and early_stopping.should_stop(self.infer, epoch):
                break

        if report_every_epochs >= 0:
            log.info(f"Final loss {epoch_loss}")

    def run_epoch(self, loss_fn, loader, optimizer, config, device, grad_scaler=None) -> torch.tensor:
        epoch_loss = torch.tensor(0.0, device=device)
        for data, target, *_ in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                output = self.model(data)
                loss = loss_fn(output, target)

            if grad_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            epoch_loss += loss.detach()
        return epoch_loss / len(loader)

    def forward(self, input, samples=1):
        return self.infer(input, samples)

    def infer(self, input, samples):
        self.model.eval()
        return torch.stack([self.model(input) for _ in range(samples)])

    def submodels(self):
        return [self]


class MAPOptimizer(BayesianOptimizer):
    '''
        Maximum A Posteriori

        This simply optimizes a point estimate of the parameters with the given base_optimizer.
    '''

    def __init__(self, params, base_optimizer):
        super().__init__(params, {})
        self.state["__base_optimizer"] = base_optimizer

    def step(self, forward_closure, backward_closure, grad_scaler=None):
        self.zero_grad()

        loss = forward_closure()
        backward_closure(loss)

        if grad_scaler is not None:
            grad_scaler.step(self.state["__base_optimizer"])
        else:
            self.state["__base_optimizer"].step()

        return loss

    def sample_parameters(self):
        pass

    def get_base_optimizer(self):
        return self.state["__base_optimizer"]

from types import LambdaType
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def combined_variance_output(input, ensemble):
    means = torch.zeros((input.shape[0], len(ensemble)))
    variances = torch.zeros((input.shape[0], len(ensemble)))
    for i, model in enumerate(ensemble):
        output = model(input)
        means[:,i] = output[:,0]
        variances[:,i] = torch.log1p(torch.exp(output[:,1])) + 10e-6
    mean = torch.mean(means, dim=1)
    # Variance from https://github.com/cameronccohen/deep-ensembles/blob/master/Tutorial.ipynb
    variance = torch.mean(variances + means**2, dim=1) - mean**2 
    return mean, variance

class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "models": [model.state_dict(destination, prefix, keep_vars) for model in self.models]
        }

    def load_state_dict(self, dict):
        for model, state in zip(self.models, dict["models"]):
            model.load_state_dict(state)

    def train_model(self, *args, **kwargs):
        for i, model in enumerate(self.models):
            print(f"Training ensemble member {i}")
            model.train_model(*args, **kwargs)

    def infer(self, input, samples, *args, **kwargs):
        assert samples >= len(self.models)
        limit = kwargs.get("limit", len(self.models))
        outputs = []
        for i, model in enumerate(self.models):
            if i >= limit:
                break
            if i == len(self.models) - 1:
                outputs.append(model.infer(input, samples - i * (samples // len(self.models)), *args, **kwargs))
            else:
                outputs.append(model.infer(input, samples // len(self.models), *args, **kwargs))
        return torch.concat(outputs)

    def submodels(self):
        return self.models


class DeepEnsemble(nn.Module):
    '''
        Stores modules together with their optimizers
    '''
    def __init__(self, models_and_optimizers):
        super().__init__()
        self.models = nn.ModuleList(list(map(lambda p: p[0], models_and_optimizers)))
        self.optimizers = list(map(lambda p: p[1], models_and_optimizers))

    def state_dict(self, prefix='', keep_vars=False):
        return {
            "models": self.models.state_dict(prefix=prefix, keep_vars=keep_vars),
            "optimizers": list(map(lambda o: o.state_dict(), self.optimizers))
        }

    def load_state_dict(self, state_dict, strict=True):
        self.models.load_state_dict(state_dict["models"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(optimizer_state)

    def predict(self, predict_closure, samples, multisample=False):
        '''
            Makes <samples> predictions with this ensemble

            predict_closure takes a model (that is part of this ensemble) as an argument and makes a single prediction with it. This function already calls sample_parameters()
        '''
        if len(self.models) == 1 and getattr(self.models[0], "supports_multisample", False) and multisample:
            return predict_closure(self.models[0], n_samples=samples)

        output = []
        preds_per_model = samples // len(self.models_and_optimizers)
        for i, (model, optimizer) in enumerate(self.models_and_optimizers):
            model_samples = preds_per_model if i > 0 else (samples - (len(self.models_and_optimizers) - 1) * preds_per_model)
            for _ in range(model_samples):
                optimizer.sample_parameters()
                output.append(predict_closure(model))
        return torch.stack(output)

    @property
    def models_and_optimizers(self):
        return list(zip(self.models, self.optimizers))

import multiprocessing
from itertools import repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import itertools
import importlib
import time
import math
from src.algos import util
from src.swag import SWAGWrapper
from src.bbb import run_bbb_epoch, BBBLinear, GaussianPrior
import gpytorch
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

def point_predictor(layers, epochs, dataloader, batch_size):
    pp_model = util.generate_model(layers, "relu", "sigmoid")

    optimizer = torch.optim.SGD(pp_model.parameters(), lr=0.01)
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            output = pp_model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_pp(input, samples):
        return [pp_model(input) for _ in range(samples)]

    return eval_pp

def swag(layers, epochs, dataloader, batch_size, update_freq, updates):
    swag_model = util.generate_model(layers, "relu", "sigmoid")
    optimizer = torch.optim.SGD(swag_model.parameters(), lr=0.01) # Without weight_decay the covariance matrix is not positive definit???
    wrapper = SWAGWrapper(swag_model, update_freq, updates)
    for epoch in range(epochs):
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            output = swag_model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        wrapper.update(epoch)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_swag(input, samples):
        torch.manual_seed(42)
        return [wrapper.sample(swag_model, input) for _ in range(samples)]
    
    return eval_swag

def ensemble(ensemble_count, layers, epochs, dataloader, batch_size):
    models = [util.generate_model(layers, "relu", "sigmoid") for _ in range(ensemble_count)]

    for i, model in enumerate(models):
        print(f"Training model {i}")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            epoch_loss = torch.tensor(0, dtype=torch.float)
            for data, target in dataloader:
                optimizer.zero_grad()
                output = model(data)
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
        print(f"  Final loss {epoch_loss / (len(dataloader) * batch_size)}")


    def eval_esemble(input, samples):
        assert samples == len(models)
        return [model(input) for model in models]

    return eval_esemble

def mc_droupout(p, layers, epochs, dataloader, batch_size):
    mc_model = util.generate_model(layers, "relu", "sigmoid", scale=1/(1 - p), dropout_p=p)
    optimizer = torch.optim.SGD(mc_model.parameters(), lr=0.01)

    for epoch in range(epochs):
        mc_model.train()
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            output = mc_model(data)
            loss = F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_dropout(input, samples):
        mc_model.train() # Enable dropout
        return [mc_model(input) for _ in range(samples)]

    return eval_dropout

def bbb(prior, mc_sample, sampling, layers, epochs, dataloader, batch_size, device):
    bbb_model = util.generate_model(layers, "relu", "sigmoid", linear_fn=lambda i, o: BBBLinear(i, o, prior, prior, device, mc_sample=mc_sample, sampling=sampling))
    bbb_model.to(device)
    optimizer = torch.optim.SGD(bbb_model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(epochs):
        loss = run_bbb_epoch(bbb_model, optimizer, loss_fn, dataloader, device)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {loss / (len(dataloader) * batch_size)}")


    def eval_bbb(input, samples):
        return [bbb_model(input) for _ in range(samples)]
    
    return eval_bbb

def bbb_intel(config, layers, epochs, dataloader, batch_size):
    intel_model = util.generate_model(layers, "relu", "sigmoid")

    dnn_to_bnn(intel_model, config)

    optimizer = torch.optim.Adam(intel_model.parameters(), 0.01)
    for epoch in range(epochs):
        intel_model.train()
        epoch_loss = torch.tensor(0, dtype=torch.float)
        for data, target in dataloader:
            optimizer.zero_grad()
            output = intel_model(data)
            loss = get_kl_loss(intel_model) / batch_size + F.binary_cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss / (len(dataloader) * batch_size)}")
    print(f"Final loss {epoch_loss / (len(dataloader) * batch_size)}")

    def eval_bayes(input, samples):
        intel_model.eval()
        return [intel_model(input) for _ in range(samples)]

    return eval_bayes
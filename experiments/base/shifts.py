from posixpath import split
import torch
import pandas as pd

from experiments.base.normalized import NormalizedTensorDataset


def _split(data):
    return torch.tensor(data.iloc[:,6:].values).float(), torch.tensor(data["fact_temperature"].values).float().unsqueeze(-1)

class WeatherShiftsDataset:
    def __init__(self, path):
        self.path = path
        self.trainset_loaded = False
    
    def trainloader(self, batch_size, shuffle=True, small=False):
        name = "Shifts/weather/shifts_canonical_dev_in.csv" if small else "Shifts/weather/shifts_canonical_train.csv"
        print("Loading dataset...")
        data = pd.read_csv(self.path + name).dropna()
        print(f"Loaded {len(data)} data points. Normalizing...")
        dataset = NormalizedTensorDataset(*_split(data))
        print("Normalization completed.")

        self.data_mean = dataset.data_mean
        self.data_std = dataset.data_std
        self.target_mean = dataset.target_mean
        self.target_std = dataset.target_std
        self.trainset_loaded = True

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def in_testloader(self, batch_size, shuffle=True):
        assert self.trainset_loaded
        data = pd.read_csv(self.path + "Shifts/weather/shifts_canonical_eval_in.csv").dropna()
        dataset = NormalizedTensorDataset(*_split(data), data_mean=self.data_mean, data_std=self.data_std, target_mean=self.target_mean, target_std=self.target_std)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def out_testloader(self, batch_size, shuffle=True):
        assert self.trainset_loaded
        data = pd.read_csv(self.path + "Shifts/weather/shifts_canonical_eval_out.csv").dropna()
        dataset = NormalizedTensorDataset(*_split(data), data_mean=self.data_mean, data_std=self.data_std, target_mean=self.target_mean, target_std=self.target_std)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def in_valloader(self, batch_size, shuffle=False, size=1000):
        assert self.trainset_loaded
        data = pd.read_csv(self.path + "Shifts/weather/shifts_canonical_dev_in.csv", nrows=size).dropna()
        dataset = NormalizedTensorDataset(*_split(data), data_mean=self.data_mean, data_std=self.data_std, target_mean=self.target_mean, target_std=self.target_std)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


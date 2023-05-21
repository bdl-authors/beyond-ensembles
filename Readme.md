# Beyond Deep Ensembles - A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift

This repository contains the implementation and evaluation code for all algorithms and experiments from the paper.

## Structure of the Code
[`src`](./src/) contains the implementation of the algorithms ([`src/algos`](./src/algos/)), evaluation metrics ([`src/eval`](./src/eval/)), and architectures that we implemented from scratch ([`src/architectures`](./src/architectures/)).


## Usage of the Algorithms
All algorithms are implemented as PyTorch optimizers that.
Because many algorithms require special handling of the forward and backward pass, the optimizer's `step` methods require `forward` and `backward` closures to be passed to them.
The `forward_closure` closure should execute a single forward pass and must *not* call `backward()` on the loss, but return it.
The `backward_closure` closure should execute a single backward pass on the passed `loss`: `loss.backward()` or `scaler.scale(loss).backward()` if using a gradient scaler.
You need to call `complete_epoch()` on the optimizer after each *epoch*, as some algorithms (mainly SWAG) want to do some bookkeeping here.

All algorithms are subclasses of the [`BayesianOptimizer`](./src/algos/algo.py), which contains further documentation and special code to handle gradient scalers.


## Reproduction of the Experiments

### Setup
Make sure that you have PyTorch 2.0 and a compatible version of TorchVision installed.
Then run
```
pip install matplotlib tabulate wilds netcal cw2 transformers wandb laplace-torch
pip install git+https://github.com/treforevans/uci_datasets.git
```
WILDS also requires a version of [TorchScatter](https://github.com/rusty1s/pytorch_scatter) that is compatible with PyTorch 2.0.

Use the following code snippet to selectively download WILDS datasets (e.g. for iwildcam):
```python
from .experiments.base import wilds1

wilds1.download_dataset("./data/", "iwildcam")
```
You can also specify a different path, but then you have to adapt all pathes in the experiment configuration files.

If you want to reproduce the CIFAR-10 experiments, you also need to run the following commands:
```
pip install jax==0.4.1 jaxlib==0.4.1+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow tensorflow_datasets
pip install dm-haiku
```
Finally, you need to download the HMC samples from Izmailow et al. with `gsutil`:
```
gsutil -m cp -r gs://gresearch/bnn-posteriors/v1/hmc/cifar10/ ./data/Wilson/
```

### Running the Experiments
For each task there is a corresponding directory below [`experiments`](./experiments/).
Each directory contains a Python file with the name of the task (e.g. `iwildcam.py`) and a YAML file with the same name.
First, run the non-MultiX algorithms by running e.g.  `python3 iwildcam.py iwildcam.yaml` for iWildCam.
Then, evaluate the MultiX models (reuses the trained models from the single-mode algorithms) by running `python3 eval_ensembles.py eval_ensembles.yaml` in the same directory as before.
Finally, fit the Laplace approximations on top of the MAP models by running `python3 fit_laplace.py fit_laplace.yaml` in the same directory.
All scripts print their results to stdout and to WandB if you are logged in and `disable_wandb` is `False` in the YAML files.
The experiment directories also contain Jupyter Notebooks to query the results from WandB, plot them, and print LaTeX tables from them.

For UCI, you only need to run `python3 uci.py uci.yaml`, as this script also fits the Laplace approximations and evaluates MultiX.
For PovertyMap-wilds, the `eval_ensembles` script is also not required as the main script also trains the ensembles.

# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Using vanilla PyTorch to perform optimization in SVI.
#
# This tutorial demonstrates how to use standard PyTorch optimizers, dataloaders and training loops
# to perform optimization in SVI. This is useful when you want to use custom optimizers,
# learning rate schedules, dataloaders, or other advanced training techniques,
# or just to simplify integration with other elements of the PyTorch ecosystem.

import argparse
import pandas as pd
from typing import Callable
from pathlib import Path
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule
from pyro.contrib.autoguide import AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from torch.utils.data import DataLoader, TensorDataset

# We define a model as usual. This model is data parallel and supports subsampling.

def model(X, n_cat, obs=None):
    input_dim = X.shape[1]
    device = X.device  # Get the device of the input tensor
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(1, n_cat, device=device), 
                                             5.*torch.ones(1, n_cat, device=device)).to_event())  # Prior for the bias/intercept
    beta  = pyro.sample("beta", dist.Normal(torch.zeros(input_dim, n_cat, device=device), 
                                            5.*torch.ones(input_dim, n_cat, device=device)).to_event()) # Priors for the regression coeffcients
    
    with pyro.plate("data"):
        y = pyro.sample("y", dist.Categorical(logits=alpha + X.matmul(beta)), obs=obs)
        
    return y
        
def load_data(parent_dir):
    data_dir = parent_dir[2].joinpath("data", "processed")
    X_train = pd.read_csv(data_dir.joinpath("X_train.csv")).values
    X_test = pd.read_csv(data_dir.joinpath("X_test.csv")).values
    y_train = pd.read_csv(data_dir.joinpath("y_train.csv")).values
    y_test = pd.read_csv(data_dir.joinpath("y_test.csv")).values
    return X_train, X_test, y_train, y_test

def main(args, parent_dir):
    X_train, X_test, y_train, y_test = load_data(parent_dir)
    n_cat = 2
    device = torch.device("cuda" if args.cuda else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Define guide function
    guide = AutoMultivariateNormal(model)

    # Reset parameter values
    pyro.clear_param_store()

    # Setup the optimizer
    adam_params = {"lr": args.learning_rate}
    optimizer = ClippedAdam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO()
    svi = SVI(model, guide, optimizer, loss=elbo)

    # Training loop with mini-batch gradient steps
    for epoch in range(args.n_steps):
        epoch_loss = svi.step(X_train, n_cat, y_train.reshape(-1))
        if epoch % args.log_interval == 0:
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")
    # lets save the trained model
    torch.save({"model" : model.state_dict(), "guide" : guide}, "mymodel.pt")
    pyro.get_param_store().save("mymodelparams.pt")

if __name__ == "__main__":
    assert pyro.__version__.startswith("1.9.0")
    parent_dir = Path(__file__).parents
    parser = argparse.ArgumentParser(
        description="Process some integers..."
    )
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--batch-size", default=10000, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--seed", default=20200723, type=int)
    parser.add_argument("--n_steps", default=5000, type=int)
    parser.add_argument("--cuda", action="store_true", default=torch.cuda.is_available())
    args = parser.parse_args()
    main(args, parent_dir)
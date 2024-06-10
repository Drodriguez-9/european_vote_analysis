# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Bayesian Neural Network
================================

We demonstrate how to use NUTS to do inference on a simple (small)
Bayesian neural network with two hidden layers.

.. image:: ../_static/img/examples/bnn.png
    :align: center
"""

import argparse
import os
import time
import pandas as pd
from src.data.make_dataset_alternative import split_data, load_data

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from sklearn.metrics import recall_score

matplotlib.use("Agg")  # noqa: E402


# the non-linearity we use in our neural network
def nonlin(x):
    return jnp.tanh(x)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H, D_Y=1):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample final layer of weights and neural network output
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    assert w3.shape == (D_H, D_Y)
    z3 = jnp.matmul(z2, w3)  # <= output of the neural network
    assert z3.shape == (N, D_Y)

    if Y is not None:
        assert z3.shape == Y.shape

    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)
    probs = sigmoid(z3)
    # observe data
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("Y", dist.Bernoulli(probs).to_event(1), obs=Y)


# helper function for HMC inference
def run_inference(model, args, rng_key, X, Y, D_H):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, D_H)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace["Y"]["value"]

# create artificial regression dataset
def get_data(sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    parent_dir = Path(__file__).parents
    df = load_data(parent_dir)
    X, X_test, Y, Y_test = split_data(df)   
     
    X_test.to_csv(parent_dir[2].joinpath("data", "processed", "X_test.csv"))
    Y_test.to_csv(parent_dir[2].joinpath("data", "processed", "y_test.csv"))

    X, Y, X_test, Y_test = X.values.astype(float), Y.values.astype(float), X_test.values.astype(float), Y_test.values.astype(float)
    N, D_X = X.shape
    Y = Y[:, np.newaxis]
    # Y -= jnp.mean(Y)
    # Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    return X, Y, X_test, Y_test


def main(args):
    D_H = args.num_hidden
    X, Y, X_test, Y_test = get_data()
    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y, D_H)
    path = rf"/zhome/58/f/181392/DTU/MBML/european_vote_analysis/src/data/{args.samples_name}"
    np.savez(path, **samples)
    # predict Y_test at inputs X_test
    vmap_args = (
        samples,
        random.split(rng_key_predict, args.num_samples * args.num_chains),
    )
    predictions = vmap(
        lambda samples, rng_key: predict(model, rng_key, samples, X_test, D_H)
    )(*vmap_args)
    predictions = predictions[..., 0]
    predictions_plot = jnp.exp(predictions[..., 0])  # Convert logits to probabilities

    # compute mean prediction and confidence interval around median
    mean_prediction = jnp.mean(predictions, axis=0)
    mean_prediction_plot = jnp.mean(predictions_plot, axis=0)
    percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)
    percentiles_plot = np.percentile(predictions_plot, [5.0, 95.0], axis=0)

    # Calculate accuracy for binary classification
    for threshold in np.arange(0.1, 0.5, 0.1):
        print("--------------------")
        print(f"threshold: {threshold}")
        mean_prediction_class = (mean_prediction >= threshold).astype(int)
        accuracy = accuracy_score(Y_test, mean_prediction_class)
        precision = precision_score(Y_test, mean_prediction_class)
        recall = recall_score(Y_test, mean_prediction_class)
        auc_score = roc_auc_score(Y_test, mean_prediction_class)

        print(f"Test accuracy: {accuracy * 100:.2f}%")
        print(f"Test precision: {precision * 100:.2f}%")
        print(f"Test recall: {recall * 100:.2f}%")
        print(f"Roc Auc Score: {auc_score:.4f}")
    

    # make plots
    # fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # # # plot training data
    # # ax.plot(X[:, 1], Y[:, 0], "kx")
    # # # plot 90% confidence level of predictions
    # # ax.fill_between(
    # #     X_test[:, 1], percentiles[0, :], percentiles[1, :], color="lightblue"
    # # )
    # # # plot mean prediction
    # # ax.plot(X_test[:, 1], mean_prediction, "blue", ls="solid", lw=2.0)
    # # ax.set(xlabel="X", ylabel="Y", title="Mean predictions with 90% CI")

    # # plt.savefig(Path.joinpath(args.path, "bnn_plot.pdf"))
    # # make plots
    # fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # # plot training data
    # ax.scatter(X[:, 1], Y, color="black", alpha=0.5, label='Training Data')

    # # plot 90% confidence level of predictions
    # ax.fill_between(
    #     X_test[:, 1], percentiles_plot[0, :], percentiles_plot[1, :], color="lightblue", alpha=0.5, label='90% CI'
    # )

    # # plot mean prediction
    # ax.plot(X_test[:, 1], mean_prediction_plot, "blue", ls="solid", lw=2.0, label='Mean Prediction')

    # ax.set(xlabel="X", ylabel="Probability", title="Mean predictions with 90% CI")
    # ax.legend()
    # plt.savefig(args.path / "bnn_plot.pdf")


if __name__ == "__main__":
    parent_dir = Path(__file__).parents
    assert numpyro.__version__.startswith("0.15.0")
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("--path", nargs="?", default=Path.joinpath(parent_dir[2], "reports", "figures"), type=str)
    parser.add_argument("--samples_name", nargs="?", default="samples", type=str)
    parser.add_argument("--num-warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--num-samples", nargs="?", default=2000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-hidden", nargs="?", default=3, type=int)
    parser.add_argument("--device", default="gpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    main(args)


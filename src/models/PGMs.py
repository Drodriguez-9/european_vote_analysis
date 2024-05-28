import torch
import pyro
import pyro.distributions as dist


def model_base(X, n_cat, obs=None):
    input_dim = X.shape[1]
    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(1, n_cat),
                                             5. * torch.ones(1, n_cat)).to_event())  # Prior for the bias/intercept
    beta = pyro.sample("beta", dist.Normal(torch.zeros(input_dim, n_cat),
                                           5. * torch.ones(input_dim,
                                                           n_cat)).to_event())  # Priors for the regression coeffcients

    with pyro.plate("data"):
        y = pyro.sample("y", dist.Categorical(logits=alpha + X.matmul(beta)), obs=obs)

    return y

def model_baseNN(X, n_cat, obs=None):
    # TODO: Diego fill in the Neural net output version
    return y


def model_hierarchical(X, n_cat1, n_cat2, cat2, obs=None):
    input_dim = X.shape[1]
    alpha_mu = pyro.sample("alpha_mu", dist.Normal(torch.zeros(n_cat1),
                                                   10. * torch.ones(n_cat1)).to_event())  # Prior for the bias mean
    alpha_sigma = pyro.sample("alpha_sigma", dist.HalfCauchy(
        10. * torch.ones(n_cat1)).to_event())  # Prior for the bias standard deviation
    beta = pyro.sample("beta", dist.Normal(torch.zeros(input_dim, n_cat1),
                                           10. * torch.ones(input_dim,
                                                            n_cat1)).to_event())  # Priors for the regression coefficents

    with pyro.plate("ind", n_cat2):
        alpha = pyro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma).to_event(
            1))  # Draw the individual parameter for each individual

    with pyro.plate("data", X.shape[0]):
        logits = alpha[cat2] + X.matmul(beta)
        y = pyro.sample("y", dist.Categorical(logits=logits), obs=obs)  # If you use logits you don't need to do sigmoid
    return y



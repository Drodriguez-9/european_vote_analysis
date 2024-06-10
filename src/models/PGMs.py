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


# first try
def model_multi_hierarchical(X, categoricals, n_cat, obs=None):
    input_dim = X.shape[1]
    cat_dim = categoricals.shape[1]

    logits = 0

    #importance = pyro.sample("importance", dist.HalfCauchy(10. * torch.ones(cat_dim)).to_event())


    for cat in  pyro.plate("cats", cat_dim):
        # Prior for the bias mean
        alpha_mu = pyro.sample("alpha_mu{cat}",
                               dist.Normal(torch.zeros(2),
                                           10. * torch.ones(2)).to_event())

        # Prior for the bias standard deviation
        alpha_sigma = pyro.sample("alpha_sigma{cat}",
                                  dist.HalfCauchy(10. * torch.ones(2)).to_event())
        # Prior for the weight mean
        beta_mu = pyro.sample("beta_mu{cat}",
                              dist.Normal(torch.zeros(2),
                                          10. * torch.ones(2)).to_event())

        # Prior for the weight standard deviation
        beta_sigma = pyro.sample("beta_sigma{cat}",
                                 dist.HalfCauchy(10. * torch.ones(2)).to_event())

        with pyro.plate("ind", n_cat):
            # Draw the individual parameter for each individual
            alpha = pyro.sample("alpha_{cat}",
                                dist.Normal(alpha_mu, alpha_sigma).to_event(1))

            # Priors for the regression coefficents
            beta = pyro.sample("beta_{cat}",
                               dist.Normal(beta_mu, beta_sigma).to_event(1))

        logits += alpha + X.matmul(beta)


    with pyro.plate("data", X.shape[0]):
        logits = alpha[ind] + X.matmul(beta)
        # If you use logits you don't need to do sigmoid
        y = pyro.sample("y",
                        dist.Categorical(logits=logits), obs=obs)

    return y


# second try
def model_categorical_weights(X, categoricals, n_cat, obs=None):
    device = X.device
    input_dim = X.shape[1]
    cat_dim = categoricals.shape[1]

    logits = 0

    l_prior = pyro.sample("l_prior", dist.Beta(2. * torch.ones(cat_dim, device=device),
                                               1. * torch.ones(cat_dim, device=device)).to_event())
    """
    # Prior for the bias mean
    alpha_mu = pyro.sample("alpha_mu",
                           dist.Normal(torch.zeros(input_dim, 2),
                                       10. * torch.ones(cat_dim, 2)).to_event())

    # Prior for the bias standard deviation
    alpha_sigma = pyro.sample("alpha_sigma",
                              dist.HalfCauchy(10. * torch.ones(input_dim, 2)).to_event())
    # Prior for the weight mean
    beta_mu = pyro.sample("beta_mu",
                          dist.Normal(torch.zeros(input_dim, cat_dim, 2),
                                      10. * torch.ones(input_dim, cat_dim, 2)).to_event())

    # Prior for the weight standard deviation
    beta_sigma = pyro.sample("beta_sigma",
                             dist.HalfCauchy(10. * torch.ones(input_dim, cat_dim, 2)).to_event())

    with pyro.plate("inputs", input_dim) as dim:
        # Draw the individual parameter for each individual
        alpha = pyro.sample("alpha",
                            dist.Normal(alpha_mu[dim], alpha_sigma[dim]).to_event())

        # Priors for the regression coefficents
        beta = pyro.sample("beta",
                           dist.Normal(beta_mu[dim], beta_sigma[dim]).to_event())    
    """

    alpha = pyro.sample("alpha", dist.Normal(torch.zeros(cat_dim, n_cat, device=device),
                                             1. * torch.ones(cat_dim, n_cat, device=device)).to_event())
    # Prior for the bias/intercept
    beta = pyro.sample("beta", dist.Normal(torch.zeros(input_dim, n_cat, cat_dim, device=device),
                                           1. * torch.ones(input_dim, n_cat, cat_dim, device=device)).to_event())
    # Priors for the regression coeffcients
    #l = categoricals
    with pyro.plate("data", X.shape[0]):
        l = pyro.sample("l", dist.Bernoulli(l_prior).to_event(), obs=categoricals)
        #print(f"l: {l.shape}")
        #print(f"beta: {beta.shape}")
        #print(f"alpha: {alpha.shape}" )
        #print(f"X: {X.shape}" )
        alpha_l = l.matmul(alpha)  # (x.shape[0], n_cat)
        beta_l = torch.einsum('inc,rc->rni', beta, l)  #(batch_size, n_cat, input_dim)
        #beta_l = beta.matmul(l.T)
        #print(f"alpha_l: {alpha_l.shape}" )
        #print(f"beta_l: {beta_l.shape}" )
        #print(torch.einsum('ri,rni->rn', X, beta_l).shape)
        logits = alpha_l + torch.einsum('ri,rni->rn', X, beta_l)
        #logits = alpha_l + X.T.matmul(beta_l)
        # If you use logits you don't need to do sigmoid
        y = pyro.sample("y",
                        dist.Categorical(logits=logits), obs=obs)

    return y

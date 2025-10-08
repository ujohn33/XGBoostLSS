import torch
from torch.nn.functional import softplus, gumbel_softmax, softmax


def nan_to_num(predt: torch.tensor) -> torch.tensor:
    """
    Replace nan, inf and -inf with the mean of predt.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.nan_to_num(predt,
                             nan=float(torch.nanmean(predt)),
                             posinf=float(torch.nanmean(predt)),
                             neginf=float(torch.nanmean(predt))
                             )

    return predt


def identity_fn(predt: torch.tensor) -> torch.tensor:
    """
    Identity mapping of predt.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = nan_to_num(predt) + torch.tensor(0, dtype=predt.dtype)

    return predt


def exp_fn(predt: torch.tensor) -> torch.tensor:
    """
    Exponential function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.exp(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def exp_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Exponential function used for Student-T distribution.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.exp(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt + torch.tensor(2.0, dtype=predt.dtype)


def softplus_fn(predt: torch.tensor) -> torch.tensor:
    """
    Softplus function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = softplus(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def softplus_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Softplus function used for Student-T distribution.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = softplus(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt + torch.tensor(2.0, dtype=predt.dtype)


def squareplus_fn(predt: torch.tensor) -> torch.tensor:
    """
    Square-Plus function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    b = torch.tensor(4., dtype=predt.dtype)
    predt = 0.5 * (predt + torch.sqrt(predt ** 2 + b)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def squareplus_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Square-Plus function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    b = torch.tensor(4., dtype=predt.dtype)
    predt = 0.5 * (predt + torch.sqrt(predt ** 2 + b)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt + torch.tensor(2.0, dtype=predt.dtype)


def sigmoid_fn(predt: torch.tensor) -> torch.tensor:
    """
    Function used to ensure predt are scaled to (0,1).

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.sigmoid(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)
    predt = torch.clamp(predt, 1e-03, 1-1e-03)

    return predt


def relu_fn(predt: torch.tensor) -> torch.tensor:
    """
    Function used to ensure predt are scaled to max(0, predt).

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.relu(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def relu_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Function used to ensure predt are scaled to max(0, predt).

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.relu(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt + torch.tensor(2.0, dtype=predt.dtype)


def softmax_fn(predt: torch.tensor) -> torch.tensor:
    """
    Softmax function used to ensure predt is adding to one.


    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = softmax(nan_to_num(predt), dim=1) + torch.tensor(0, dtype=predt.dtype)

    return predt


def gumbel_softmax_fn(predt: torch.tensor,
                      tau: float = 1.0
                      ) -> torch.tensor:
    """
    Gumbel-softmax function used to ensure predt is adding to one.

    The Gumbel-softmax distribution is a continuous distribution over the simplex, which can be thought of as a "soft"
    version of a categorical distribution. It's a way to draw samples from a categorical distribution in a
    differentiable way. The motivation behind using the Gumbel-Softmax is to make the discrete sampling process of
    categorical variables differentiable, which is useful in gradient-based optimization problems. To sample from a
    Gumbel-Softmax distribution, one would use the Gumbel-max trick: add a Gumbel noise to logits and apply the softmax.
    Formally, given a vector z, the Gumbel-softmax function s(z,tau)_i for a component i at temperature tau is
    defined as:

        s(z,tau)_i = frac{e^{(z_i + g_i) / tau}}{sum_{j=1}^M e^{(z_j + g_j) / tau}}

    where g_i is a sample from the Gumbel(0, 1) distribution. The parameter tau (temperature) controls the sharpness
    of the output distribution. As tau approaches 0, the mixing probabilities become more discrete, and as tau
    approaches infty, the mixing probabilities become more uniform. For more information we refer to

        Jang, E., Gu, Shixiang and Poole, B. "Categorical Reparameterization with Gumbel-Softmax", ICLR, 2017.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.
    tau: float, non-negative scalar temperature.
        Temperature parameter for the Gumbel-softmax distribution. As tau -> 0, the output becomes more discrete, and as
        tau -> inf, the output becomes more uniform.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    torch.manual_seed(123)
    predt = gumbel_softmax(nan_to_num(predt), tau=tau, dim=1) + torch.tensor(0, dtype=predt.dtype)

    return predt


# Add common evaluation functions
def evaluate_nll(distribution, params_df, y_true):
    """
    Calculate negative log-likelihood for a distribution with given parameters and true values.
    
    Arguments
    ---------
    distribution: torch.distributions.Distribution
        The distribution to evaluate
    params_df: pd.DataFrame
        DataFrame containing distribution parameters
    y_true: np.ndarray or torch.tensor
        True values
        
    Returns
    -------
    float:
        Negative log-likelihood value
    """
    try:
        # Convert parameters to tensors
        params = [torch.tensor(params_df[param].values, dtype=torch.float32) 
                 for param in params_df.columns]
        
        # Convert y_true to tensor if it's not already
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)
        
        # Create distribution with parameters
        dist_kwargs = dict(zip(params_df.columns, params))
        dist = distribution(**dist_kwargs)
        
        # Calculate negative log-likelihood
        nll = -dist.log_prob(y_true).mean().item()
        return nll
    except Exception as e:
        print(f"Error calculating NLL: {str(e)}")
        return float('nan')


def evaluate_crps(distribution, params_df, y_true, n_samples=100):
    """
    Calculate Continuous Ranked Probability Score (CRPS) for a distribution.
    
    Arguments
    ---------
    distribution: torch.distributions.Distribution
        The distribution to evaluate
    params_df: pd.DataFrame
        DataFrame containing distribution parameters
    y_true: np.ndarray or torch.tensor
        True values
    n_samples: int
        Number of samples to draw for CRPS calculation
        
    Returns
    -------
    tuple:
        CRPS score, calibration component, sharpness component
    """
    try:
        # Convert parameters to tensors
        params = [torch.tensor(params_df[param].values, dtype=torch.float32) 
                 for param in params_df.columns]
        
        # Convert y_true to tensor if it's not already
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)
            
        # Create distribution with parameters
        dist_kwargs = dict(zip(params_df.columns, params))
        dist = distribution(**dist_kwargs)
        
        # Generate samples from distribution
        samples = dist.sample((n_samples,)).transpose(0, 1)
        
        # Calculate CRPS using empirical CDF
        y_true_expanded = y_true.unsqueeze(1).expand(-1, n_samples)
        crps_per_observation = (samples - y_true_expanded).abs().mean(dim=1) - 0.5 * (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1,2))
        
        # Calculate components
        calibration = (samples - y_true_expanded).abs().mean(dim=1).mean()
        sharpness = 0.5 * (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1,2)).mean()
        
        return crps_per_observation.mean().item(), calibration.item(), sharpness.item()
    except Exception as e:
        print(f"Error calculating CRPS: {str(e)}")
        return float('nan'), float('nan'), float('nan')

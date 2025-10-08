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


def evaluate_nll(distribution_class, params_df, y_true):
    """
    Calculate negative log-likelihood for a distribution with given parameters and true values.
    
    Arguments
    ---------
    distribution_class: class
        The distribution class to evaluate
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
        # Convert y_true to tensor if it's not already
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32).reshape(-1, 1)
        elif len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        # Create distribution instance with default parameters
        dist_instance = distribution_class()
        
        # Extract parameter values from DataFrame
        param_tensors = []
        for param_name in dist_instance.distribution_arg_names:
            if param_name in params_df:
                param_tensors.append(torch.tensor(params_df[param_name].values, 
                                                dtype=torch.float32).reshape(-1, 1))
            else:
                raise ValueError(f"Parameter {param_name} not found in params_df")
        
        # Create a dictionary that maps the actual pytorch distribution
        distribution = dist_instance.distribution
        
        # Apply the response functions to transform parameters appropriately
        transformed_params = {}
        for i, (name, fn) in enumerate(dist_instance.param_dict.items()):
            if callable(fn):  # Apply the response function if it's callable
                transformed_params[name] = fn(param_tensors[i])
            else:
                transformed_params[name] = param_tensors[i]
        
        # Create the pytorch distribution
        pred_dist = distribution(**transformed_params)
        
        # Calculate negative log likelihood
        nll = -pred_dist.log_prob(y_true).mean().item()
        
        return nll
    except Exception as e:
        print(f"Error calculating NLL: {str(e)}")
        return float('inf')  # Return infinity for errors to avoid selecting problematic distributions


def evaluate_crps(distribution_class, params_df, y_true, n_samples=100):
    """
    Calculate Continuous Ranked Probability Score (CRPS) for a distribution.
    
    Arguments
    ---------
    distribution_class: class
        The distribution class to evaluate
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
        # Convert y_true to tensor if it's not already
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32).reshape(-1, 1)
        elif len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        
        # Create distribution instance with default parameters
        dist_instance = distribution_class()
        
        # Extract parameter values from DataFrame
        param_tensors = []
        for param_name in dist_instance.distribution_arg_names:
            if param_name in params_df:
                param_tensors.append(torch.tensor(params_df[param_name].values, 
                                                dtype=torch.float32).reshape(-1, 1))
            else:
                raise ValueError(f"Parameter {param_name} not found in params_df")
        
        # Create a dictionary that maps the actual pytorch distribution
        distribution = dist_instance.distribution
        
        # Apply the response functions to transform parameters appropriately
        transformed_params = {}
        for i, (name, fn) in enumerate(dist_instance.param_dict.items()):
            if callable(fn):  # Apply the response function if it's callable
                transformed_params[name] = fn(param_tensors[i])
            else:
                transformed_params[name] = param_tensors[i]
        
        # Create the pytorch distribution
        pred_dist = distribution(**transformed_params)
        
        # Generate samples from distribution - shape [batch_size, n_samples]
        samples = pred_dist.sample((n_samples,))
        
        # Get batch size
        batch_size = samples.shape[0]
        
        # Term 1: Mean absolute difference between samples and observations
        # [batch_size, n_samples] -> [batch_size]
        term1 = (samples - y_true).abs().mean(dim=1)
        
        # Term 2: Mean pairwise absolute difference between samples
        # Create tensors for pairwise differences
        samples_i = samples.unsqueeze(2)  # [batch_size, n_samples, 1]
        samples_j = samples.unsqueeze(1)  # [batch_size, 1, n_samples]
        
        # Calculate pairwise absolute differences and average
        # [batch_size, n_samples, n_samples] -> [batch_size]
        term2 = 0.5 * (samples_i - samples_j).abs().mean(dim=(1,2))
        
        # CRPS = term1 - term2
        crps_per_observation = term1 - term2
        
        # Calculate average CRPS and components
        crps_mean = crps_per_observation.mean().item()
        calibration = term1.mean().item()
        sharpness = term2.mean().item()
        
        return crps_mean, calibration, sharpness
        
    except Exception as e:
        import traceback
        print(f"Error calculating CRPS: {str(e)}")
        traceback.print_exc()  # Print full stack trace
        return float('inf'), float('inf'), float('inf')
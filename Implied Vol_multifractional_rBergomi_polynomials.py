import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import gamma as gamma_function
from numpy.fft import rfft, irfft

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12
})



def bsm_price(S, K, T, r, q, sigma, flag):
    """
    Calculates the Black-Scholes-Merton option price.

    Parameters:
    ----------
    S: Underlying asset price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    q: Continuous dividend yield
    sigma: Volatility
    flag: Option type ('c' for call, 'p' for put)

    Returns:
    -------
    float
        The calculated price of the option.
    """
    # Handle edge cases with zero/near-zero time to maturity or volatility by returning intrinsic value.
    if T <= 1e-6 or sigma <= 1e-6:
        return np.maximum(0, S * np.exp(-q * T) - K * np.exp(-r * T)) if flag == 'c' else np.maximum(0, K * np.exp(-r * T) - S * np.exp(-q * T))
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if flag == 'c':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def find_implied_vol(target_price, S, K, T, r, q, flag):
    """
    Calculates the implied volatility for an option.
    This is the volatility that makes the BSM price equal to the market price.

    Parameters:
    ----------
    price: Market price of the option
    S, K, T, r, q, flag: BSM model parameters

    Returns:
    -------
    float
        The implied volatility, or np.nan if not found.
    """
    # The objective is to find the root of: BSM_price(sigma) - target_price = 0
    objective = lambda sigma: bsm_price(S, K, T, r, q, sigma, flag) - target_price
    
    try:
        # Use a numerical solver (Brent's method) to find the root for sigma
        # within a reasonable search range (e.g., 0.01% to 500%).
        return brentq(objective, 1e-4, 5.0)
    except (ValueError, RuntimeError): # If no root is found, return NaN
        return np.nan

def plot_with_nan_handling(x, y, **kwargs):
    """
    Plots data on a specific axes object, correctly handling NaN values by creating gaps in the line.
    It expects 'ax' to be passed as a keyword argument.
    """
    # Explicitly get the 'ax' object from kwargs and remove it so it's not passed to ax.plot
    ax = kwargs.pop('ax', plt.gca())
    
    plot_kwargs = kwargs # The rest of the kwargs are for plotting

    x = np.array(x)
    y = np.array(y)
    is_nan = np.isnan(y)
    
    start_idx = 0
    for i in range(1, len(y)):
        if is_nan[i] and not is_nan[i-1]: # End of a non-NaN segment
            ax.plot(x[start_idx:i], y[start_idx:i], **plot_kwargs)
            # Remove label for subsequent segments to avoid duplicates in legend
            if 'label' in plot_kwargs:
                del plot_kwargs['label']
        elif not is_nan[i] and is_nan[i-1]: # Start of a new non-NaN segment
            start_idx = i
    
    # Plot the last segment if it's not a NaN value
    if not is_nan[-1]:
        ax.plot(x[start_idx:], y[start_idx:], **plot_kwargs)
   

def price_rbergomi_constant_h(K, T, T1, H1, H2, eta, rho, xi, S0, r, q, n_steps, n_paths, rng):
    """
    Prices an option using the slow but theoretically consistent 'true memory' rBergomi model.
    This version correctly handles a time-varying Hurst parameter H(t) for both the
    stochastic process Y_t and its theoretical variance Var(Y_t) for the drift correction.

    This ensures the resulting variance process V_t is a true martingale under the risk-neutral measure.

    Args:
        K (float): Strike price.
        T (float): Time to maturity.
        T1 (float): The time at which the Hurst parameter regime switches (tau).
        H1 (float): Hurst parameter for the first regime [0, T1].
        H2 (float): Hurst parameter for the second regime (T1, T].
        eta (float): Volatility of volatility parameter.
        rho (float): Correlation between the two Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        n_steps (int): Number of time steps for the simulation.
        n_paths (int): Number of Monte Carlo simulation paths.
        rng (np.random.Generator): An isolated random number generator instance to ensure
                                   simulations are independent and reproducible.
    """
    # Define the time step size for the Euler discretization.
    dt = T / n_steps
    # Create the discrete time grid for the simulation.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate fundamental random drivers ---
    # Generate the increments for two independent Brownian motions, W and W_perp.
    # dW drives the volatility process.
    # dW_perp is used to construct the correlated process for the stock price.
    # The provided 'rng' object ensures that simulations are isolated and reproducible.
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    dW_perp = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))

    # Construct the increments for the correlated Brownian motion dZ using a Cholesky-like decomposition.
    # dZ is correlated with dW with a correlation coefficient of rho.
    # dZ_t = rho * dW_t + sqrt(1 - rho^2) * dW_perp_t
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # Define the piecewise-constant Hurst function H(t), which creates different roughness regimes.
    def H_func(t):
        """Returns the Hurst parameter based on the time t."""
        return H1 if t <= T1 else H2

    # --- Step 2: Generate the fractional stochastic process Y_paths ---
    # This simulates the fractional integral Y_t = integral from 0 to t of K(t,s) dW_s,
    # where the kernel K(t,s) depends on a time-varying Hurst parameter H(s).
    # Kernel K(t,s) = sqrt(2*H(s)) * (t-s)^(H(s)-0.5).
    # This is computationally intensive due to the nested loops (O(n_steps^2)).
    Y_paths = np.zeros((n_paths, n_steps + 1))
    # Loop over each time step t_i in the grid to calculate Y_{t_i}.
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum = np.zeros(n_paths)
        # Inner loop approximates the integral by summing contributions from all past shocks dW_j.
        for j in range(i):
            tj = t_grid[j]
            # Crucially, the Hurst parameter is determined by the time of the past shock, H(s=tj).
            H_past = H_func(tj)
            # Calculate the value of the discretized fractional kernel.
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            # Add the contribution of this past shock to the integral sum for each path.
            integral_sum += kernel_val * dW[:, j]
        Y_paths[:, i] = integral_sum

    # --- Step 3: Calculate the theoretical variance for the drift correction ---
    # To ensure V_t is a martingale, its drift must be corrected by subtracting 0.5 * eta^2 * Var(Y_t).
    # This block computes the deterministic variance Var(Y_t) = integral from 0 to t of K(t,s)^2 ds.
    # The formula is: Var(Y_t) = integral from 0 to t of [2*H(s) * (t - s)^(2*H(s)-1)] ds.
    variance_drift = np.zeros(n_steps + 1)
    # Loop over each time step t_i to calculate Var(Y_{t_i}).
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        # Inner loop approximates the integral using a Riemann sum.
        for j in range(i):
            tj = t_grid[j]
            # Again, use the Hurst parameter from the time of integration, H(s=tj).
            H_s = H_func(tj)
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 4: Construct the Variance (V) process ---
    # The variance process is defined as V_t = xi * exp(eta * Y_t - 0.5 * eta^2 * Var(Y_t)).
    # We use the simulated Y_paths and the calculated variance_drift for this.
    # `np.newaxis` is used to broadcast the 1D variance_drift array to match the 2D Y_paths shape.
    V = xi * np.exp(eta * Y_paths - 0.5 * eta**2 * variance_drift[np.newaxis, :])

    # --- Step 5: Construct the final Stock Price (ST) ---
    # We solve for the terminal stock price ST by discretizing the SDE:
    # dS_t / S_t = (r - q) dt + sqrt(V_t) dZ_t
    # The solution is S_T = S_0 * exp( (r-q)T - 0.5 * integral_0^T V_s ds + integral_0^T sqrt(V_s) dZ_s ).
    # This is calculated efficiently for all paths using vectorized numpy operations.
    integral_V_dt = np.sum(V[:, :-1] * dt, axis=1)  # The Ito correction term integral.
    integral_sqrtV_dZ = np.sum(np.sqrt(V[:, :-1]) * dZ, axis=1) # The stochastic integral.
    
    ST = S0 * np.exp((r - q) * T - 0.5 * integral_V_dt + integral_sqrtV_dZ)

    # --- Step 6: Calculate the Option Price ---
    # Determine if the option is a call or a put based on its moneyness.
    # This is a simple heuristic; a more robust implementation might take an explicit flag.
    is_call = K >= S0
    if is_call:
        # Calculate call option payoffs for all paths.
        payoffs = np.maximum(ST - K, 0)
    else:
        # Calculate put option payoffs for all paths.
        payoffs = np.maximum(K - ST, 0)
    
    # The final option price is the discounted average of all simulated payoffs,
    # according to the principles of risk-neutral Monte Carlo pricing.
    option_price = np.mean(payoffs) * np.exp(-r * T)
    
    return option_price

def price_rbergomi_linear_h(K, T, beta0, beta1, eta, rho, xi, S0, r, q, n_steps, n_paths, rng):
    """
    Prices an option using the slow but theoretically consistent 'true memory' rBergomi model.
    This version correctly handles a time-varying Hurst parameter H(t) for both the
    stochastic process Y_t and its theoretical variance Var(Y_t) for the drift correction.

    This ensures the resulting variance process V_t is a true martingale under the risk-neutral measure.

    Args:
        K (float): Strike price.
        T (float): Time to maturity.
        beta0 (float): parameter of the polynomial (degree 0)
        beta1 (float): parameter of the polynomial (degree 1)
        eta (float): Volatility of volatility parameter.
        rho (float): Correlation between the two Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        n_steps (int): Number of time steps for the simulation.
        n_paths (int): Number of Monte Carlo simulation paths.
        rng (np.random.Generator): An isolated random number generator instance to ensure
                                   simulations are independent and reproducible.
    """
    # Define the time step size for the Euler discretization.
    dt = T / n_steps
    # Create the discrete time grid for the simulation.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate fundamental random drivers ---
    # Generate the increments for two independent Brownian motions, W and W_perp.
    # dW drives the volatility process.
    # dW_perp is used to construct the correlated process for the stock price.
    # The provided 'rng' object ensures that simulations are isolated and reproducible.
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    dW_perp = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))

    # Construct the increments for the correlated Brownian motion dZ using a Cholesky-like decomposition.
    # dZ is correlated with dW with a correlation coefficient of rho.
    # dZ_t = rho * dW_t + sqrt(1 - rho^2) * dW_perp_t
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # Define the piecewise-constant Hurst function H(t), which creates different roughness regimes.
        
    def H_func(t, beta0=0.1, beta1=0.3):
        """Returns the Hurst parameter based on the time t."""
        x = beta0 + beta1 * t 
        return np.clip (x, 1e-4, 0.4999) #the clip prevents H(t) from being outside the interval (0, 1/2)
    
    """
    Logistic H(t):
        def H_func(t, H_min=0.05, H_max=0.45, gamma=10, t_star=0.25):
            return H_min + (H_max - H_min) / (1 + np.exp(-gamma * (t - t_star)))
        
    Linear H(t):
        def H_func(t, beta0=0.1, beta1=0.3):
            return beta0 + beta1 * t  
    
    Sinusoidal H(t):
        def H_func(t, T, alpha0=0.1, alpha1=0.3):
            return alpha0 + alpha1 * np.sin((2 * np.pi * t) / T)  
    """

    # --- Step 2: Generate the fractional stochastic process Y_paths ---
    # This simulates the fractional integral Y_t = integral from 0 to t of K(t,s) dW_s,
    # where the kernel K(t,s) depends on a time-varying Hurst parameter H(s).
    # Kernel K(t,s) = sqrt(2*H(s)) * (t-s)^(H(s)-0.5).
    # This is computationally intensive due to the nested loops (O(n_steps^2)).
    Y_paths = np.zeros((n_paths, n_steps + 1))
    # Loop over each time step t_i in the grid to calculate Y_{t_i}.
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum = np.zeros(n_paths)
        # Inner loop approximates the integral by summing contributions from all past shocks dW_j.
        for j in range(i):
            tj = t_grid[j]
            # Crucially, the Hurst parameter is determined by the time of the past shock, H(s=tj).
            H_past = H_func(tj)
            # Calculate the value of the discretized fractional kernel.
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            # Add the contribution of this past shock to the integral sum for each path.
            integral_sum += kernel_val * dW[:, j]
        Y_paths[:, i] = integral_sum

    # --- Step 3: Calculate the theoretical variance for the drift correction ---
    # To ensure V_t is a martingale, its drift must be corrected by subtracting 0.5 * eta^2 * Var(Y_t).
    # This block computes the deterministic variance Var(Y_t) = integral from 0 to t of K(t,s)^2 ds.
    # The formula is: Var(Y_t) = integral from 0 to t of [2*H(s) * (t - s)^(2*H(s)-1)] ds.
    variance_drift = np.zeros(n_steps + 1)
    # Loop over each time step t_i to calculate Var(Y_{t_i}).
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        # Inner loop approximates the integral using a Riemann sum.
        for j in range(i):
            tj = t_grid[j]
            # Again, use the Hurst parameter from the time of integration, H(s=tj).
            H_s = H_func(tj)
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 4: Construct the Variance (V) process ---
    # The variance process is defined as V_t = xi * exp(eta * Y_t - 0.5 * eta^2 * Var(Y_t)).
    # We use the simulated Y_paths and the calculated variance_drift for this.
    # `np.newaxis` is used to broadcast the 1D variance_drift array to match the 2D Y_paths shape.
    V = xi * np.exp(eta * Y_paths - 0.5 * eta**2 * variance_drift[np.newaxis, :])

    # --- Step 5: Construct the final Stock Price (ST) ---
    # We solve for the terminal stock price ST by discretizing the SDE:
    # dS_t / S_t = (r - q) dt + sqrt(V_t) dZ_t
    # The solution is S_T = S_0 * exp( (r-q)T - 0.5 * integral_0^T V_s ds + integral_0^T sqrt(V_s) dZ_s ).
    # This is calculated efficiently for all paths using vectorized numpy operations.
    integral_V_dt = np.sum(V[:, :-1] * dt, axis=1)  # The Ito correction term integral.
    integral_sqrtV_dZ = np.sum(np.sqrt(V[:, :-1]) * dZ, axis=1) # The stochastic integral.
    
    ST = S0 * np.exp((r - q) * T - 0.5 * integral_V_dt + integral_sqrtV_dZ)

    # --- Step 6: Calculate the Option Price ---
    # Determine if the option is a call or a put based on its moneyness.
    # This is a simple heuristic; a more robust implementation might take an explicit flag.
    is_call = K >= S0
    if is_call:
        # Calculate call option payoffs for all paths.
        payoffs = np.maximum(ST - K, 0)
    else:
        # Calculate put option payoffs for all paths.
        payoffs = np.maximum(K - ST, 0)
    
    # The final option price is the discounted average of all simulated payoffs,
    # according to the principles of risk-neutral Monte Carlo pricing.
    option_price = np.mean(payoffs) * np.exp(-r * T)
    
    return option_price

def price_rbergomi_binomial_h(K, T, beta0, beta1, beta2, eta, rho, xi, S0, r, q, n_steps, n_paths, rng):
    """
    Prices an option using the slow but theoretically consistent 'true memory' rBergomi model.
    This version correctly handles a time-varying Hurst parameter H(t) for both the
    stochastic process Y_t and its theoretical variance Var(Y_t) for the drift correction.

    This ensures the resulting variance process V_t is a true martingale under the risk-neutral measure.

    Args:
        K (float): Strike price.
        T (float): Time to maturity.
        beta0 (float): parameter of the polynomial (degree 0)
        beta1 (float): parameter of the polynomial (degree 1)
        beta2 (float): parameter of the polynomial (degree 2)
        eta (float): Volatility of volatility parameter.
        rho (float): Correlation between the two Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        n_steps (int): Number of time steps for the simulation.
        n_paths (int): Number of Monte Carlo simulation paths.
        rng (np.random.Generator): An isolated random number generator instance to ensure
                                   simulations are independent and reproducible.
    """
    # Define the time step size for the Euler discretization.
    dt = T / n_steps
    # Create the discrete time grid for the simulation.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate fundamental random drivers ---
    # Generate the increments for two independent Brownian motions, W and W_perp.
    # dW drives the volatility process.
    # dW_perp is used to construct the correlated process for the stock price.
    # The provided 'rng' object ensures that simulations are isolated and reproducible.
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    dW_perp = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))

    # Construct the increments for the correlated Brownian motion dZ using a Cholesky-like decomposition.
    # dZ is correlated with dW with a correlation coefficient of rho.
    # dZ_t = rho * dW_t + sqrt(1 - rho^2) * dW_perp_t
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # Define the piecewise-constant Hurst function H(t), which creates different roughness regimes.
        
    def H_func(t, beta0=0.15, beta1=0.45, beta2=0.25):
        """Returns the Hurst parameter based on the time t."""
        x = beta0 + beta1 * t + beta2 * t**2
        return np.clip (x, 1e-4, 0.4999) #the clip prevents H(t) from being outside the interval (0, 1/2)

    # --- Step 2: Generate the fractional stochastic process Y_paths ---
    # This simulates the fractional integral Y_t = integral from 0 to t of K(t,s) dW_s,
    # where the kernel K(t,s) depends on a time-varying Hurst parameter H(s).
    # Kernel K(t,s) = sqrt(2*H(s)) * (t-s)^(H(s)-0.5).
    # This is computationally intensive due to the nested loops (O(n_steps^2)).
    Y_paths = np.zeros((n_paths, n_steps + 1))
    # Loop over each time step t_i in the grid to calculate Y_{t_i}.
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum = np.zeros(n_paths)
        # Inner loop approximates the integral by summing contributions from all past shocks dW_j.
        for j in range(i):
            tj = t_grid[j]
            # Crucially, the Hurst parameter is determined by the time of the past shock, H(s=tj).
            H_past = H_func(tj)
            # Calculate the value of the discretized fractional kernel.
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            # Add the contribution of this past shock to the integral sum for each path.
            integral_sum += kernel_val * dW[:, j]
        Y_paths[:, i] = integral_sum

    # --- Step 3: Calculate the theoretical variance for the drift correction ---
    # To ensure V_t is a martingale, its drift must be corrected by subtracting 0.5 * eta^2 * Var(Y_t).
    # This block computes the deterministic variance Var(Y_t) = integral from 0 to t of K(t,s)^2 ds.
    # The formula is: Var(Y_t) = integral from 0 to t of [2*H(s) * (t - s)^(2*H(s)-1)] ds.
    variance_drift = np.zeros(n_steps + 1)
    # Loop over each time step t_i to calculate Var(Y_{t_i}).
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        # Inner loop approximates the integral using a Riemann sum.
        for j in range(i):
            tj = t_grid[j]
            # Again, use the Hurst parameter from the time of integration, H(s=tj).
            H_s = H_func(tj)
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 4: Construct the Variance (V) process ---
    # The variance process is defined as V_t = xi * exp(eta * Y_t - 0.5 * eta^2 * Var(Y_t)).
    # We use the simulated Y_paths and the calculated variance_drift for this.
    # `np.newaxis` is used to broadcast the 1D variance_drift array to match the 2D Y_paths shape.
    V = xi * np.exp(eta * Y_paths - 0.5 * eta**2 * variance_drift[np.newaxis, :])

    # --- Step 5: Construct the final Stock Price (ST) ---
    # We solve for the terminal stock price ST by discretizing the SDE:
    # dS_t / S_t = (r - q) dt + sqrt(V_t) dZ_t
    # The solution is S_T = S_0 * exp( (r-q)T - 0.5 * integral_0^T V_s ds + integral_0^T sqrt(V_s) dZ_s ).
    # This is calculated efficiently for all paths using vectorized numpy operations.
    integral_V_dt = np.sum(V[:, :-1] * dt, axis=1)  # The Ito correction term integral.
    integral_sqrtV_dZ = np.sum(np.sqrt(V[:, :-1]) * dZ, axis=1) # The stochastic integral.
    
    ST = S0 * np.exp((r - q) * T - 0.5 * integral_V_dt + integral_sqrtV_dZ)

    # --- Step 6: Calculate the Option Price ---
    # Determine if the option is a call or a put based on its moneyness.
    # This is a simple heuristic; a more robust implementation might take an explicit flag.
    is_call = K >= S0
    if is_call:
        # Calculate call option payoffs for all paths.
        payoffs = np.maximum(ST - K, 0)
    else:
        # Calculate put option payoffs for all paths.
        payoffs = np.maximum(K - ST, 0)
    
    # The final option price is the discounted average of all simulated payoffs,
    # according to the principles of risk-neutral Monte Carlo pricing.
    option_price = np.mean(payoffs) * np.exp(-r * T)
    
    return option_price

def price_rbergomi_cubic_h(K, T, beta0, beta1, beta2, beta3, eta, rho, xi, S0, r, q, n_steps, n_paths, rng):
    """
    Prices an option using the slow but theoretically consistent 'true memory' rBergomi model.
    This version correctly handles a time-varying Hurst parameter H(t) for both the
    stochastic process Y_t and its theoretical variance Var(Y_t) for the drift correction.

    This ensures the resulting variance process V_t is a true martingale under the risk-neutral measure.
    
    Prices an option under the multifractional rough Bergomi model
    with a logistic Hurst function:
    
        H(t) = H_min + (H_max - H_min) / (1 + exp(-gamma*(t - t_star)))
    
    This smoothly generalises the two-regime piecewise constant model
    of the standard rough Bergomi framework.

    Args:
        K (float): Strike price.
        T (float): Time to maturity.
        H_min (float): Minimum Hurst parameter (roughness level for t << t_star)
        H_max (float): Maximum Hurst parameter (smoothness level for t >> t_star)
        gamma (float): Steepness of the logistic transition (larger = sharper)
        t_star (float): Midpoint of the transition (inflection point)
        eta (float): Volatility of volatility parameter.
        rho (float): Correlation between the two Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        n_steps (int): Number of time steps for the simulation.
        n_paths (int): Number of Monte Carlo simulation paths.
        rng (np.random.Generator): An isolated random number generator instance to ensure
                                   simulations are independent and reproducible.
    """
    # Define the time step size for the Euler discretization.
    dt = T / n_steps
    # Create the discrete time grid for the simulation.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate fundamental random drivers ---
    # Generate the increments for two independent Brownian motions, W and W_perp.
    # dW drives the volatility process.
    # dW_perp is used to construct the correlated process for the stock price.
    # The provided 'rng' object ensures that simulations are isolated and reproducible.
    dW = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    dW_perp = rng.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))

    # Construct the increments for the correlated Brownian motion dZ using a Cholesky-like decomposition.
    # dZ is correlated with dW with a correlation coefficient of rho.
    # dZ_t = rho * dW_t + sqrt(1 - rho^2) * dW_perp_t
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # Define the piecewise-constant Hurst function H(t), which creates different roughness regimes.
        
    def H_func(t, beta0=0.15, beta1=0.45, beta2=0.25, beta3=0.8):
        """
        Logistic Hurst function, capturing smooth transition
        from H_min to H_max around t_star.
        """
        x = beta0 + beta1*t + beta2*t**2 + beta3*t**3
        return np.clip(x, 1e-4, 0.4999) #the clip prevents H(t) from being outside the interval (0, 1/2)

    # --- Step 2: Generate the fractional stochastic process Y_paths ---
    # This simulates the fractional integral Y_t = integral from 0 to t of K(t,s) dW_s,
    # where the kernel K(t,s) depends on a time-varying Hurst parameter H(s).
    # Kernel K(t,s) = sqrt(2*H(s)) * (t-s)^(H(s)-0.5).
    # This is computationally intensive due to the nested loops (O(n_steps^2)).
    Y_paths = np.zeros((n_paths, n_steps + 1))
    # Loop over each time step t_i in the grid to calculate Y_{t_i}.
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum = np.zeros(n_paths)
        # Inner loop approximates the integral by summing contributions from all past shocks dW_j.
        for j in range(i):
            tj = t_grid[j]
            # Crucially, the Hurst parameter is determined by the time of the past shock, H(s=tj).
            H_past = H_func(tj)
            # Calculate the value of the discretized fractional kernel.
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            # Add the contribution of this past shock to the integral sum for each path.
            integral_sum += kernel_val * dW[:, j]
        Y_paths[:, i] = integral_sum

    # --- Step 3: Calculate the theoretical variance for the drift correction ---
    # To ensure V_t is a martingale, its drift must be corrected by subtracting 0.5 * eta^2 * Var(Y_t).
    # This block computes the deterministic variance Var(Y_t) = integral from 0 to t of K(t,s)^2 ds.
    # The formula is: Var(Y_t) = integral from 0 to t of [2*H(s) * (t - s)^(2*H(s)-1)] ds.
    variance_drift = np.zeros(n_steps + 1)
    # Loop over each time step t_i to calculate Var(Y_{t_i}).
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        # Inner loop approximates the integral using a Riemann sum.
        for j in range(i):
            tj = t_grid[j]
            # Again, use the Hurst parameter from the time of integration, H(s=tj).
            H_s = H_func(tj)
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 4: Construct the Variance (V) process ---
    # The variance process is defined as V_t = xi * exp(eta * Y_t - 0.5 * eta^2 * Var(Y_t)).
    # We use the simulated Y_paths and the calculated variance_drift for this.
    # `np.newaxis` is used to broadcast the 1D variance_drift array to match the 2D Y_paths shape.
    V = xi * np.exp(eta * Y_paths - 0.5 * eta**2 * variance_drift[np.newaxis, :])

    # --- Step 5: Construct the final Stock Price (ST) ---
    # We solve for the terminal stock price ST by discretizing the SDE:
    # dS_t / S_t = (r - q) dt + sqrt(V_t) dZ_t
    # The solution is S_T = S_0 * exp( (r-q)T - 0.5 * integral_0^T V_s ds + integral_0^T sqrt(V_s) dZ_s ).
    # This is calculated efficiently for all paths using vectorized numpy operations.
    integral_V_dt = np.sum(V[:, :-1] * dt, axis=1)  # The Ito correction term integral.
    integral_sqrtV_dZ = np.sum(np.sqrt(V[:, :-1]) * dZ, axis=1) # The stochastic integral.
    
    ST = S0 * np.exp((r - q) * T - 0.5 * integral_V_dt + integral_sqrtV_dZ)

    # --- Step 6: Calculate the Option Price ---
    # Determine if the option is a call or a put based on its moneyness.
    # This is a simple heuristic; a more robust implementation might take an explicit flag.
    is_call = K >= S0
    if is_call:
        # Calculate call option payoffs for all paths.
        payoffs = np.maximum(ST - K, 0)
    else:
        # Calculate put option payoffs for all paths.
        payoffs = np.maximum(K - ST, 0)
    
    # The final option price is the discounted average of all simulated payoffs,
    # according to the principles of risk-neutral Monte Carlo pricing.
    option_price = np.mean(payoffs) * np.exp(-r * T)
    
    return option_price

def simulate_rbergomi_constant_h_path(T, T1, H1, H2, eta, rho, xi, S0, n_steps):
    """
    Simulates a single Variance (V_t) and Price (S_t) path for the rBergomi model
    with a piecewise-constant Hurst parameter (regime-switching).

    This function uses a theoretically consistent approach where the variance of the
    simulated fractional process Y_t matches its corresponding theoretical drift
    correction term, ensuring the variance process V_t is a martingale.

    Args:
        T (float): Total time to maturity.
        T1 (float): Time of the regime switch for the Hurst parameter.
        H1 (float): Hurst parameter for the first regime [0, T1].
        H2 (float): Hurst parameter for the second regime (T1, T].
        eta (float): Volatility of volatility.
        rho (float): Correlation between the two driving Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        n_steps (int): The number of time steps for the simulation.

    Returns:
        (tuple): A tuple containing (t_grid, S_path, V_path), which are the
                 time points, the simulated stock price path, and the variance path.
    """
    # --- Setup: Time Discretization ---
    # Calculate the size of each time step.
    dt = T / n_steps
    # Create the grid of time points from 0 to T.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate Correlated Brownian Increments ---
    # Generate increments for two independent Brownian motions, W and W_perp.
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    dW_perp = np.random.normal(0, np.sqrt(dt), n_steps)
    # Construct the correlated Brownian motion Z using correlation rho.
    # This is used to drive the stock price process.
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # --- Step 2: Define the Piecewise-Constant Hurst Function ---
    # This function implements the regime switch for the model's "roughness".
    def H_func(t):
        """Returns the Hurst parameter H based on the time t."""
        return H1 if t <= T1 else H2

    # --- Step 3: Simulate the Fractional Process Y_t ---
    # This simulates the fractional integral Y_t = integral[0,t] K(t,s) dW_s,
    # where the kernel K(t,s) = sqrt(2H(s)) * (t-s)^(H(s)-0.5).
    # This nested loop is computationally intensive but theoretically correct.
    Y_path = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):  # For each time step t_i...
        ti = t_grid[i]
        integral_sum = 0.0
        for j in range(i):  # ...sum the effects of all past shocks dW_j.
            tj = t_grid[j]
            # The kernel's roughness depends on the Hurst param at the time of the shock (tj).
            H_past = H_func(tj)
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            integral_sum += kernel_val * dW[j]
        Y_path[i] = integral_sum

    # --- Step 4: Calculate the Theoretical Variance of Y_t ---
    # This computes the deterministic drift correction term needed to make V_t a martingale.
    # The term is E[Y_t^2] = Var(Y_t) = integral[0,t] K(t,s)^2 ds.
    # This calculation must be consistent with the simulation in Step 3.
    variance_drift = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        for j in range(i):
            tj = t_grid[j]
            H_s = H_func(tj)
            # The integrand is K(t,s)^2 = 2*H(s) * (t-s)^(2H(s)-1).
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 5: Construct the Variance and Stock Price Paths ---
    # Construct the variance path V_t = xi * exp(eta*Y_t - 0.5*eta^2*E[Y_t^2]).
    V_path = xi * np.exp(eta * Y_path - 0.5 * eta**2 * variance_drift)

    # Construct the stock price path S_t using a log-Euler scheme.
    # The SDE is dS_t/S_t = sqrt(V_t) dZ_t (assuming r=q=0 for path simulation).
    S_path = np.zeros(n_steps + 1)
    S_path[0] = S0
    for i in range(n_steps):
        # Ensure variance is non-negative to avoid sqrt(negative number) errors.
        V_t = np.maximum(V_path[i], 1e-12)
        # S_t+1 = S_t * exp(-0.5*V_t*dt + sqrt(V_t)*dZ_t)
        S_path[i+1] = S_path[i] * np.exp(-0.5 * V_t * dt + np.sqrt(V_t) * dZ[i])

    return t_grid, S_path, V_path

def simulate_rbergomi_linear_h_path(T, beta0, beta1, eta, rho, xi, S0, n_steps):
    """
    Simulates a single Variance (V_t) and Price (S_t) path for the rBergomi model
    with the Hurst parameter being a linear function.

    This function uses a theoretically consistent approach where the variance of the
    simulated fractional process Y_t matches its corresponding theoretical drift
    correction term, ensuring the variance process V_t is a martingale.

    Args:
        T (float): Total time to maturity.
        T1 (float): Time of the regime switch for the Hurst parameter.
        H1 (float): Hurst parameter for the first regime [0, T1].
        H2 (float): Hurst parameter for the second regime (T1, T].
        eta (float): Volatility of volatility.
        rho (float): Correlation between the two driving Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        n_steps (int): The number of time steps for the simulation.

    Returns:
        (tuple): A tuple containing (t_grid, S_path, V_path), which are the
                 time points, the simulated stock price path, and the variance path.
    """
    # --- Setup: Time Discretization ---
    # Calculate the size of each time step.
    dt = T / n_steps
    # Create the grid of time points from 0 to T.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate Correlated Brownian Increments ---
    # Generate increments for two independent Brownian motions, W and W_perp.
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    dW_perp = np.random.normal(0, np.sqrt(dt), n_steps)
    # Construct the correlated Brownian motion Z using correlation rho.
    # This is used to drive the stock price process.
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # --- Step 2: Define the Piecewise-Constant Hurst Function ---
    # This function implements the regime switch for the model's "roughness".
    def H_func(t, beta0=0.15, beta1=0.45):
        """Returns the Hurst parameter based on the time t."""
        x = beta0 + beta1 * t 
        return np.clip (x, 1e-4, 0.4999) #the clip prevents H(t) from being outside the interval (0, 1/2)

    # --- Step 3: Simulate the Fractional Process Y_t ---
    # This simulates the fractional integral Y_t = integral[0,t] K(t,s) dW_s,
    # where the kernel K(t,s) = sqrt(2H(s)) * (t-s)^(H(s)-0.5).
    # This nested loop is computationally intensive but theoretically correct.
    Y_path = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):  # For each time step t_i...
        ti = t_grid[i]
        integral_sum = 0.0
        for j in range(i):  # ...sum the effects of all past shocks dW_j.
            tj = t_grid[j]
            # The kernel's roughness depends on the Hurst param at the time of the shock (tj).
            H_past = H_func(tj)
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            integral_sum += kernel_val * dW[j]
        Y_path[i] = integral_sum

    # --- Step 4: Calculate the Theoretical Variance of Y_t ---
    # This computes the deterministic drift correction term needed to make V_t a martingale.
    # The term is E[Y_t^2] = Var(Y_t) = integral[0,t] K(t,s)^2 ds.
    # This calculation must be consistent with the simulation in Step 3.
    variance_drift = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        for j in range(i):
            tj = t_grid[j]
            H_s = H_func(tj)
            # The integrand is K(t,s)^2 = 2*H(s) * (t-s)^(2H(s)-1).
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 5: Construct the Variance and Stock Price Paths ---
    # Construct the variance path V_t = xi * exp(eta*Y_t - 0.5*eta^2*E[Y_t^2]).
    V_path = xi * np.exp(eta * Y_path - 0.5 * eta**2 * variance_drift)

    # Construct the stock price path S_t using a log-Euler scheme.
    # The SDE is dS_t/S_t = sqrt(V_t) dZ_t (assuming r=q=0 for path simulation).
    S_path = np.zeros(n_steps + 1)
    S_path[0] = S0
    for i in range(n_steps):
        # Ensure variance is non-negative to avoid sqrt(negative number) errors.
        V_t = np.maximum(V_path[i], 1e-12)
        # S_t+1 = S_t * exp(-0.5*V_t*dt + sqrt(V_t)*dZ_t)
        S_path[i+1] = S_path[i] * np.exp(-0.5 * V_t * dt + np.sqrt(V_t) * dZ[i])

    return t_grid, S_path, V_path

def simulate_rbergomi_binomial_h_path(T, beta0, beta1, beta2, eta, rho, xi, S0, n_steps):
    """
    Simulates a single Variance (V_t) and Price (S_t) path for the rBergomi model
    with the Hurst parameter being a linear function.

    This function uses a theoretically consistent approach where the variance of the
    simulated fractional process Y_t matches its corresponding theoretical drift
    correction term, ensuring the variance process V_t is a martingale.

    Args:
        T (float): Total time to maturity.
        T1 (float): Time of the regime switch for the Hurst parameter.
        H1 (float): Hurst parameter for the first regime [0, T1].
        H2 (float): Hurst parameter for the second regime (T1, T].
        eta (float): Volatility of volatility.
        rho (float): Correlation between the two driving Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        n_steps (int): The number of time steps for the simulation.

    Returns:
        (tuple): A tuple containing (t_grid, S_path, V_path), which are the
                 time points, the simulated stock price path, and the variance path.
    """
    # --- Setup: Time Discretization ---
    # Calculate the size of each time step.
    dt = T / n_steps
    # Create the grid of time points from 0 to T.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate Correlated Brownian Increments ---
    # Generate increments for two independent Brownian motions, W and W_perp.
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    dW_perp = np.random.normal(0, np.sqrt(dt), n_steps)
    # Construct the correlated Brownian motion Z using correlation rho.
    # This is used to drive the stock price process.
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # --- Step 2: Define the Piecewise-Constant Hurst Function ---
    # This function implements the regime switch for the model's "roughness".
    def H_func(t, beta0=0.15, beta1=0.45, beta2=0.25):
        """Returns the Hurst parameter based on the time t."""
        x = beta0 + beta1 * t + beta2 * t**2 
        return np.clip (x, 1e-4, 0.4999) #the clip prevents H(t) from being outside the interval (0, 1/2)

    # --- Step 3: Simulate the Fractional Process Y_t ---
    # This simulates the fractional integral Y_t = integral[0,t] K(t,s) dW_s,
    # where the kernel K(t,s) = sqrt(2H(s)) * (t-s)^(H(s)-0.5).
    # This nested loop is computationally intensive but theoretically correct.
    Y_path = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):  # For each time step t_i...
        ti = t_grid[i]
        integral_sum = 0.0
        for j in range(i):  # ...sum the effects of all past shocks dW_j.
            tj = t_grid[j]
            # The kernel's roughness depends on the Hurst param at the time of the shock (tj).
            H_past = H_func(tj)
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            integral_sum += kernel_val * dW[j]
        Y_path[i] = integral_sum

    # --- Step 4: Calculate the Theoretical Variance of Y_t ---
    # This computes the deterministic drift correction term needed to make V_t a martingale.
    # The term is E[Y_t^2] = Var(Y_t) = integral[0,t] K(t,s)^2 ds.
    # This calculation must be consistent with the simulation in Step 3.
    variance_drift = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        for j in range(i):
            tj = t_grid[j]
            H_s = H_func(tj)
            # The integrand is K(t,s)^2 = 2*H(s) * (t-s)^(2H(s)-1).
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 5: Construct the Variance and Stock Price Paths ---
    # Construct the variance path V_t = xi * exp(eta*Y_t - 0.5*eta^2*E[Y_t^2]).
    V_path = xi * np.exp(eta * Y_path - 0.5 * eta**2 * variance_drift)

    # Construct the stock price path S_t using a log-Euler scheme.
    # The SDE is dS_t/S_t = sqrt(V_t) dZ_t (assuming r=q=0 for path simulation).
    S_path = np.zeros(n_steps + 1)
    S_path[0] = S0
    for i in range(n_steps):
        # Ensure variance is non-negative to avoid sqrt(negative number) errors.
        V_t = np.maximum(V_path[i], 1e-12)
        # S_t+1 = S_t * exp(-0.5*V_t*dt + sqrt(V_t)*dZ_t)
        S_path[i+1] = S_path[i] * np.exp(-0.5 * V_t * dt + np.sqrt(V_t) * dZ[i])

    return t_grid, S_path, V_path

def simulate_rbergomi_cubic_h_path(T, beta0, beta1, beta2, beta3, eta, rho, xi, S0, n_steps):
    """
    Simulates a single Variance (V_t) and Price (S_t) path for the rBergomi model
    with a piecewise-constant Hurst parameter (regime-switching).

    This function uses a theoretically consistent approach where the variance of the
    simulated fractional process Y_t matches its corresponding theoretical drift
    correction term, ensuring the variance process V_t is a martingale.

    Args:
        T (float): Total time to maturity.
        H_min (float): Minimum Hurst parameter (roughness level for t << t_star)
        H_max (float): Maximum Hurst parameter (smoothness level for t >> t_star)
        gamma (float): Steepness of the logistic transition (larger = sharper)
        t_star (float): Midpoint of the transition (inflection point)
        eta (float): Volatility of volatility.
        rho (float): Correlation between the two driving Brownian motions.
        xi (float): Initial forward variance (V_0).
        S0 (float): Initial stock price.
        n_steps (int): The number of time steps for the simulation.

    Returns:
        (tuple): A tuple containing (t_grid, S_path, V_path), which are the
                 time points, the simulated stock price path, and the variance path.
    """
    # --- Setup: Time Discretization ---
    # Calculate the size of each time step.
    dt = T / n_steps
    # Create the grid of time points from 0 to T.
    t_grid = np.linspace(0, T, n_steps + 1)

    # --- Step 1: Generate Correlated Brownian Increments ---
    # Generate increments for two independent Brownian motions, W and W_perp.
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    dW_perp = np.random.normal(0, np.sqrt(dt), n_steps)
    # Construct the correlated Brownian motion Z using correlation rho.
    # This is used to drive the stock price process.
    dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

    # --- Step 2: Define the Piecewise-Constant Hurst Function ---
    # This function implements the regime switch for the model's "roughness".
    def H_func(t, beta0=0.15, beta1=0.45, beta2=0.25, beta3=0.8):
        
        x = beta0 + beta1*t + beta2*t**2 + beta3*t**3
        return np.clip(x, 1e-4, 0.4999)

    # --- Step 3: Simulate the Fractional Process Y_t ---
    # This simulates the fractional integral Y_t = integral[0,t] K(t,s) dW_s,
    # where the kernel K(t,s) = sqrt(2H(s)) * (t-s)^(H(s)-0.5).
    # This nested loop is computationally intensive but theoretically correct.
    Y_path = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):  # For each time step t_i...
        ti = t_grid[i]
        integral_sum = 0.0
        for j in range(i):  # ...sum the effects of all past shocks dW_j.
            tj = t_grid[j]
            # The kernel's roughness depends on the Hurst param at the time of the shock (tj).
            H_past = H_func(tj)
            kernel_val = np.sqrt(2 * H_past) * (ti - tj)**(H_past - 0.5)
            integral_sum += kernel_val * dW[j]
        Y_path[i] = integral_sum

    # --- Step 4: Calculate the Theoretical Variance of Y_t ---
    # This computes the deterministic drift correction term needed to make V_t a martingale.
    # The term is E[Y_t^2] = Var(Y_t) = integral[0,t] K(t,s)^2 ds.
    # This calculation must be consistent with the simulation in Step 3.
    variance_drift = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        ti = t_grid[i]
        integral_sum_var = 0.0
        for j in range(i):
            tj = t_grid[j]
            H_s = H_func(tj)
            # The integrand is K(t,s)^2 = 2*H(s) * (t-s)^(2H(s)-1).
            integrand = 2 * H_s * (ti - tj)**(2 * H_s - 1)
            integral_sum_var += integrand * dt
        variance_drift[i] = integral_sum_var

    # --- Step 5: Construct the Variance and Stock Price Paths ---
    # Construct the variance path V_t = xi * exp(eta*Y_t - 0.5*eta^2*E[Y_t^2]).
    V_path = xi * np.exp(eta * Y_path - 0.5 * eta**2 * variance_drift)

    # Construct the stock price path S_t using a log-Euler scheme.
    # The SDE is dS_t/S_t = sqrt(V_t) dZ_t (assuming r=q=0 for path simulation).
    S_path = np.zeros(n_steps + 1)
    S_path[0] = S0
    for i in range(n_steps):
        # Ensure variance is non-negative to avoid sqrt(negative number) errors.
        V_t = np.maximum(V_path[i], 1e-12)
        # S_t+1 = S_t * exp(-0.5*V_t*dt + sqrt(V_t)*dZ_t)
        S_path[i+1] = S_path[i] * np.exp(-0.5 * V_t * dt + np.sqrt(V_t) * dZ[i])

    return t_grid, S_path, V_path
# ==============================================================================
# MAIN SIMULATION AND PLOTTING T = 0.5
# ==============================================================================
# --- Model Parameters ---
S0_main, r_main, q_main = 1.0, 0.0, 0.0
T_main, xi_main, eta_main, rho_main = 0.5, 0.042025, 1.8, -0.8
beta0_main, beta1_main, beta2_main, beta3_main = 0.15, 0.45, 0.25, 0.8
n_steps, n_paths = 100, 300000
moneyness_k_f = np.linspace(0.6, 1.3, 15)
strikes = S0_main * moneyness_k_f
ivols_h1, ivols_h2, ivols_linear, ivols_binomial, ivols_cubic = [], [], [], [], []

# lower and upper bound of the Hurst parameter
H1_main = 0.1
H2_main = 0.35
T1_main = 0

print("--- Starting simulations with the model ---")

# --- Create isolated Random Number Generators for each model ---
# This is the definitive solution to ensure independence between simulations.
rng_h1 = np.random.default_rng(seed=0)
rng_h2 = np.random.default_rng(seed=1)
rng_linear = np.random.default_rng(seed=2)
rng_binomial = np.random.default_rng(seed=3)
rng_cubic = np.random.default_rng(seed=4)


# --- Model 1: Constant H = 0.1 ---
print(f"--- Model 1/5: Constant H={H1_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H1_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H1_main, H1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h1)
    ivols_h1.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 2: Constant H = 0.35 ---
print(f"\n--- Model 2/5: Constant H={H2_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H2_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H2_main, H2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h2)
    ivols_h2.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 3: Linear H(t): H(t) = 0.15 + 0.45t ---
print(f"\n--- Model 3/5: Linear H(t): H(t) = 0.15 + 0.45t ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_linear_h(K, T_main, beta0_main, beta1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_linear)
    ivols_linear.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 4: Sinusoidal H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---
print(f"\n--- Model 4/5: Quadratic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_binomial_h(K, T_main, beta0_main, beta1_main, beta2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_binomial)
    ivols_binomial.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 5: Logistic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---
print(f"\n--- Model 5/5: Cubic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_cubic_h(K, T_main, beta0_main, beta1_main, beta2_main, beta3_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_cubic)
    ivols_cubic.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))    

# --- Plot the Results ---
plt.figure(figsize=(12, 8))
plot_with_nan_handling(moneyness_k_f, ivols_h1, marker='o', linestyle='--', color='blue', label=f'Constant $H = {H1_main}$')
#plot_with_nan_handling(moneyness_k_f, ivols_h2, marker='s', linestyle='--', color='orangered', label=f'Constant $H = {H2_main}$')
plot_with_nan_handling(moneyness_k_f, ivols_linear, marker='^', linestyle='-.', color='darkgreen', lw=2.5, markersize=8, label=f'Linear H(t) = 0.15 + 0.45t')
plot_with_nan_handling(moneyness_k_f, ivols_binomial, marker='v', linestyle=':', color='red', lw=2.5, markersize=8, label=f'Quadratic H(t) = 0.15 + 0.45t + 0.25t^2')
plot_with_nan_handling(moneyness_k_f, ivols_cubic, marker='D', linestyle='-', color='purple', lw=2.5, markersize=8, label=f'Cubic H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3')

plt.title(f'Rough Bergomi Implied Volatility Smile: Constant vs. Linear vs. Quadratic vs. Cubic Hurst Parameter (T={T_main})\n$\\xi={xi_main}$, $\\eta={eta_main}$, $\\rho={rho_main}$', fontsize=16)
plt.xlabel('Moneyness $(K/F)$', fontsize=12)
plt.ylabel('Implied Volatility $\\sigma_{BS}(K, T)$', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('regime_switch_smile.png', dpi=600)
print("\nPlot saved as 'regime_switch_smile.png'")
plt.show()
plt.close()

# ==============================================================================
# MAIN SIMULATION AND PLOTTING T = 1.0
# ==============================================================================
# --- Model Parameters ---
S0_main, r_main, q_main = 1.0, 0.0, 0.0
T_main, xi_main, eta_main, rho_main = 1.0, 0.042025, 1.8, -0.8
beta0_main, beta1_main, beta2_main, beta3_main = 0.15, 0.45, 0.25, 0.8
n_steps, n_paths = 100, 300000
moneyness_k_f = np.linspace(0.6, 1.3, 15)
strikes = S0_main * moneyness_k_f
ivols_h1, ivols_h2, ivols_linear, ivols_binomial, ivols_cubic = [], [], [], [], []

# lower and upper bound of the Hurst parameter
H1_main = 0.1
H2_main = 0.35
T1_main = 0

print("--- Starting simulations with the model ---")

# --- Create isolated Random Number Generators for each model ---
# This is the definitive solution to ensure independence between simulations.
rng_h1 = np.random.default_rng(seed=0)
rng_h2 = np.random.default_rng(seed=1)
rng_linear = np.random.default_rng(seed=2)
rng_binomial = np.random.default_rng(seed=3)
rng_cubic = np.random.default_rng(seed=4)


# --- Model 1: Constant H = 0.1 ---
print(f"--- Model 1/5: Constant H={H1_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H1_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H1_main, H1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h1)
    ivols_h1.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 2: Constant H = 0.35 ---
print(f"\n--- Model 2/5: Constant H={H2_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H2_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H2_main, H2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h2)
    ivols_h2.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 3: Linear H(t): H(t) = 0.15 + 0.45t ---
print(f"\n--- Model 3/5: Linear H(t): H(t) = 0.15 + 0.45t ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_linear_h(K, T_main, beta0_main, beta1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_linear)
    ivols_linear.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 4: Sinusoidal H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---
print(f"\n--- Model 4/5: Quadratic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_binomial_h(K, T_main, beta0_main, beta1_main, beta2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_binomial)
    ivols_binomial.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 5: Logistic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---
print(f"\n--- Model 5/5: Cubic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_cubic_h(K, T_main, beta0_main, beta1_main, beta2_main, beta3_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_cubic)
    ivols_cubic.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))    

# --- Plot the Results ---
plt.figure(figsize=(12, 8))
plot_with_nan_handling(moneyness_k_f, ivols_h1, marker='o', linestyle='--', color='blue', label=f'Constant $H = {H1_main}$')
#plot_with_nan_handling(moneyness_k_f, ivols_h2, marker='s', linestyle='--', color='orangered', label=f'Constant $H = {H2_main}$')
plot_with_nan_handling(moneyness_k_f, ivols_linear, marker='^', linestyle='-.', color='darkgreen', lw=2.5, markersize=8, label=f'Linear H(t) = 0.15 + 0.45t')
plot_with_nan_handling(moneyness_k_f, ivols_binomial, marker='v', linestyle=':', color='red', lw=2.5, markersize=8, label=f'Quadratic H(t) = 0.15 + 0.45t + 0.25t^2')
plot_with_nan_handling(moneyness_k_f, ivols_cubic, marker='D', linestyle='-', color='purple', lw=2.5, markersize=8, label=f'Cubic H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3')

plt.title(f'Rough Bergomi Implied Volatility Smile: Constant vs. Linear vs. Quadratic vs. Cubic Hurst Parameter (T={T_main})\n$\\xi={xi_main}$, $\\eta={eta_main}$, $\\rho={rho_main}$', fontsize=16)
plt.xlabel('Moneyness $(K/F)$', fontsize=12)
plt.ylabel('Implied Volatility $\\sigma_{BS}(K, T)$', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('regime_switch_smile.png', dpi=600)
print("\nPlot saved as 'regime_switch_smile.png'")
plt.show()
plt.close()

# ==============================================================================
# MAIN SIMULATION AND PLOTTING T = 1.5
# ==============================================================================
# --- Model Parameters ---
S0_main, r_main, q_main = 1.0, 0.0, 0.0
T_main, xi_main, eta_main, rho_main = 1.5, 0.042025, 1.8, -0.8
beta0_main, beta1_main, beta2_main, beta3_main = 0.15, 0.45, 0.25, 0.8
n_steps, n_paths = 100, 300000
moneyness_k_f = np.linspace(0.6, 1.3, 15)
strikes = S0_main * moneyness_k_f
ivols_h1, ivols_h2, ivols_linear, ivols_binomial, ivols_cubic = [], [], [], [], []

# lower and upper bound of the Hurst parameter
H1_main = 0.1
H2_main = 0.35
T1_main = 0

print("--- Starting simulations with the model ---")

# --- Create isolated Random Number Generators for each model ---
# This is the definitive solution to ensure independence between simulations.
rng_h1 = np.random.default_rng(seed=0)
rng_h2 = np.random.default_rng(seed=1)
rng_linear = np.random.default_rng(seed=2)
rng_binomial = np.random.default_rng(seed=3)
rng_cubic = np.random.default_rng(seed=4)


# --- Model 1: Constant H = 0.1 ---
print(f"--- Model 1/5: Constant H={H1_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H1_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H1_main, H1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h1)
    ivols_h1.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 2: Constant H = 0.35 ---
print(f"\n--- Model 2/5: Constant H={H2_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H2_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H2_main, H2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h2)
    ivols_h2.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 3: Linear H(t): H(t) = 0.15 + 0.45t ---
print(f"\n--- Model 3/5: Linear H(t): H(t) = 0.15 + 0.45t ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_linear_h(K, T_main, beta0_main, beta1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_linear)
    ivols_linear.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 4: Sinusoidal H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---
print(f"\n--- Model 4/5: Quadratic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_binomial_h(K, T_main, beta0_main, beta1_main, beta2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_binomial)
    ivols_binomial.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 5: Logistic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---
print(f"\n--- Model 5/5: Cubic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_cubic_h(K, T_main, beta0_main, beta1_main, beta2_main, beta3_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_cubic)
    ivols_cubic.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))    

# --- Plot the Results ---
plt.figure(figsize=(12, 8))
plot_with_nan_handling(moneyness_k_f, ivols_h1, marker='o', linestyle='--', color='blue', label=f'Constant $H = {H1_main}$')
#plot_with_nan_handling(moneyness_k_f, ivols_h2, marker='s', linestyle='--', color='orangered', label=f'Constant $H = {H2_main}$')
plot_with_nan_handling(moneyness_k_f, ivols_linear, marker='^', linestyle='-.', color='darkgreen', lw=2.5, markersize=8, label=f'Linear H(t) = 0.15 + 0.45t')
plot_with_nan_handling(moneyness_k_f, ivols_binomial, marker='v', linestyle=':', color='red', lw=2.5, markersize=8, label=f'Quadratic H(t) = 0.15 + 0.45t + 0.25t^2')
plot_with_nan_handling(moneyness_k_f, ivols_cubic, marker='D', linestyle='-', color='purple', lw=2.5, markersize=8, label=f'Cubic H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3')

plt.title(f'Rough Bergomi Implied Volatility Smile: Constant vs. Linear vs. Quadratic vs. Cubic Hurst Parameter (T={T_main})\n$\\xi={xi_main}$, $\\eta={eta_main}$, $\\rho={rho_main}$', fontsize=16)
plt.xlabel('Moneyness $(K/F)$', fontsize=12)
plt.ylabel('Implied Volatility $\\sigma_{BS}(K, T)$', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('regime_switch_smile.png', dpi=600)
print("\nPlot saved as 'regime_switch_smile.png'")
plt.show()
plt.close()


# ==============================================================================
# MAIN SIMULATION AND PLOTTING T = 2.0
# ==============================================================================
# --- Model Parameters ---
S0_main, r_main, q_main = 1.0, 0.0, 0.0
T_main, xi_main, eta_main, rho_main = 2.0, 0.042025, 1.8, -0.8
beta0_main, beta1_main, beta2_main, beta3_main = 0.15, 0.45, 0.25, 0.8
n_steps, n_paths = 100, 300000
moneyness_k_f = np.linspace(0.6, 1.3, 15)
strikes = S0_main * moneyness_k_f
ivols_h1, ivols_h2, ivols_linear, ivols_binomial, ivols_cubic = [], [], [], [], []

# lower and upper bound of the Hurst parameter
H1_main = 0.1
H2_main = 0.35
T1_main = 0

print("--- Starting simulations with the model ---")

# --- Create isolated Random Number Generators for each model ---
# This is the definitive solution to ensure independence between simulations.
rng_h1 = np.random.default_rng(seed=0)
rng_h2 = np.random.default_rng(seed=1)
rng_linear = np.random.default_rng(seed=2)
rng_binomial = np.random.default_rng(seed=3)
rng_cubic = np.random.default_rng(seed=4)


# --- Model 1: Constant H = 0.1 ---
print(f"--- Model 1/5: Constant H={H1_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H1_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H1_main, H1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h1)
    ivols_h1.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 2: Constant H = 0.35 ---
print(f"\n--- Model 2/5: Constant H={H2_main} ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # To simulate a constant H, we pass H2_main as both H1 and H2.
    price = price_rbergomi_constant_h(K, T_main, T1_main, H2_main, H2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_h2)
    ivols_h2.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))

# --- Model 3: Linear H(t): H(t) = 0.15 + 0.45t ---
print(f"\n--- Model 3/5: Linear H(t): H(t) = 0.15 + 0.45t ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_linear_h(K, T_main, beta0_main, beta1_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_linear)
    ivols_linear.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 4: Sinusoidal H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---
print(f"\n--- Model 4/5: Quadratic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_binomial_h(K, T_main, beta0_main, beta1_main, beta2_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_binomial)
    ivols_binomial.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))
    
# --- Model 5: Logistic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---
print(f"\n--- Model 5/5: Cubic H(t): H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3 ---")
for K in strikes:
    flag = 'c' if K >= S0_main else 'p'
    # Here we use H1_main and H2_main to simulate the regime switch.
    price = price_rbergomi_cubic_h(K, T_main, beta0_main, beta1_main, beta2_main, beta3_main, eta_main, rho_main, xi_main, S0_main, r_main, q_main, n_steps, n_paths, rng=rng_cubic)
    ivols_cubic.append(find_implied_vol(price, S0_main, K, T_main, r_main, q_main, flag))    

# --- Plot the Results ---
plt.figure(figsize=(12, 8))
plot_with_nan_handling(moneyness_k_f, ivols_h1, marker='o', linestyle='--', color='blue', label=f'Constant $H = {H1_main}$')
#plot_with_nan_handling(moneyness_k_f, ivols_h2, marker='s', linestyle='--', color='orangered', label=f'Constant $H = {H2_main}$')
plot_with_nan_handling(moneyness_k_f, ivols_linear, marker='^', linestyle='-.', color='darkgreen', lw=2.5, markersize=8, label=f'Linear H(t) = 0.15 + 0.45t')
plot_with_nan_handling(moneyness_k_f, ivols_binomial, marker='v', linestyle=':', color='red', lw=2.5, markersize=8, label=f'Quadratic H(t) = 0.15 + 0.45t + 0.25t^2')
plot_with_nan_handling(moneyness_k_f, ivols_cubic, marker='D', linestyle='-', color='purple', lw=2.5, markersize=8, label=f'Cubic H(t) = 0.15 + 0.45t + 0.25t^2 + 0.8t^3')

plt.title(f'Rough Bergomi Implied Volatility Smile: Constant vs. Linear vs. Quadratic vs. Cubic Hurst Parameter (T={T_main})\n$\\xi={xi_main}$, $\\eta={eta_main}$, $\\rho={rho_main}$', fontsize=16)
plt.xlabel('Moneyness $(K/F)$', fontsize=12)
plt.ylabel('Implied Volatility $\\sigma_{BS}(K, T)$', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('regime_switch_smile.png', dpi=600)
print("\nPlot saved as 'regime_switch_smile.png'")
plt.show()
plt.close()

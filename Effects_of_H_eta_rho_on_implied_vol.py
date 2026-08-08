import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import gamma as gamma_function
from numpy.fft import rfft, irfft
import time

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

class rBergomi:
    """
    A fully vectorized rBergomi simulation engine using the Hybrid Scheme.
    """
    def __init__(self, n_steps=252, T=1.0, H=0.10):
        self.T = float(T)
        self.H = float(H)
        # Alpha is the exponent in the Riemann-Liouville fractional Brownian motion kernel (t-s)^(H-1/2)
        self.alpha = self.H - 0.5
        self.n_steps = int(n_steps)
        self.dt = self.T / self.n_steps
        self.time_grid = np.linspace(0, T, self.n_steps + 1)
        
        # Covariance matrix for the bivariate normal vector [W1, W2] used in the Hybrid Scheme.
        var_W1 = self.dt
        var_W2 = (self.dt**(2 * self.H)) / (2 * self.H)
        cov_W1_W2 = (self.dt**(self.H + 0.5)) / (self.H + 0.5)
        self.cov_matrix = np.array([[var_W1, cov_W1_W2], [cov_W1_W2, var_W2]])

        # Pre-compute the gamma_k coefficients for the convolution in the Hybrid Scheme.
        # This relates to the approximation of the kernel function for the "far" part of the integral.
        k = np.arange(2, self.n_steps + 1)
        b_k = ((k**(self.alpha + 1) - (k - 1)**(self.alpha + 1)) / (self.alpha + 1))**(1 / self.alpha)
        self.gamma_k = (b_k * self.dt)**self.alpha

    def generate_paths(self, N_paths, rho, eta, xi, S0, r, q):
        """Generates all asset price paths in a single vectorized operation."""
        # Generate correlated Brownian motion increments for the Hybrid Scheme
        dW_vectors = np.random.multivariate_normal(
            mean=[0, 0], cov=self.cov_matrix, size=(N_paths, self.n_steps)
        )
        W1_increments = dW_vectors[:, :, 0] # Drives both volatility and price
        W2_increments = dW_vectors[:, :, 1] # "Near" part of the volatility integral
        
        # Generate the orthogonal component of the asset's Brownian motion
        dW_perp = np.random.randn(N_paths, self.n_steps) * np.sqrt(self.dt)

        # Simulate the Riemann-Liouville process Y, which is a Truncated Brownian Semi-Stationary (TBSS) process.
        Y = np.zeros((N_paths, self.n_steps + 1))

        # Pad the kernel `gamma_k` with a leading zero to match the signal's length.
        # Original `self.gamma_k` has length (n_steps - 1).
        gamma_k_padded = np.zeros(self.n_steps)
        # Place the original coefficients from the second element onwards.
        gamma_k_padded[1:] = self.gamma_k
        
        # 2. Set the FFT length to 2*n for a standard, robust convolution.
        fft_len = 2 * self.n_steps

        # 3. Perform the FFT on arrays of the same size. This operation is now numerically stable.
        gamma_k_fft = rfft(gamma_k_padded, fft_len)
        W1_fft = rfft(W1_increments, fft_len, axis=1)
        
        # 4. Multiply in the Fourier domain and then transform back.
        convolution_sum = irfft(gamma_k_fft * W1_fft, fft_len, axis=1)[:, :self.n_steps]
        
        Y[:, 1:] = np.sqrt(2 * self.H) * (W2_increments + convolution_sum)
        
        # Simulate the variance process V(t) where V is the forward variance (xi)
        # multiplied by the Wick exponential of the process Y. The variance of (eta * Y_t) is (eta^2 * t^(2H)).
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * self.time_grid**(2 * self.H))
        
        # Simulate the asset price process S(t) using an Euler scheme on the log-price.
        S = np.zeros((N_paths, self.n_steps + 1))
        S[:, 0] = S0
        V_sliced = np.maximum(V[:, :-1], 1e-12) # Use variance from the start of the interval, avoid sqrt(0)
        
        # Construct the correlated Brownian motion dB for the asset price
        dB = rho * W1_increments + np.sqrt(1 - rho**2) * dW_perp
        
        log_increments = (r - q - 0.5 * V_sliced) * self.dt + np.sqrt(V_sliced) * dB
        log_S_paths = np.log(S0) + np.cumsum(log_increments, axis=1)
        S[:, 1:] = np.exp(log_S_paths)
        
        return S

def _simulate_hybrid_components_mBm(H_func, n_steps, T):
    """
    Generates the core components for the Hybrid Scheme adapted for multifractional Brownian motion.
    H is frozen at each grid point t_i, i.e. H_i = H_func(t_i). This follows the procedure outlined in 
    Section 6.1.2 and is based on the implementation of the Hybrid Scheme for fractional Brownian motion, with 
    the key changes being:
        1. H_func replaces the scaler H
        2. Covariance matrix Sigma_i is rebuilt at each step using H_i
        3. The weights, b_k_star, are recomputed at each step using local alpha_i
        4. sqrt(2*H) normalisation uses local H_i, i.e. sqrt(2*H_i)
        5. W2 increments are drawn from locally adapted distribution
    """
    # --- Step 1: Set up core Parameters ---
    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)
    
    # Storage 
    W1_increments = np.zeros(n_steps)
    W2_increments = np.zeros(n_steps)
    Y = np.zeros(n_steps + 1)
    convolution_sum = np.zeros(n_steps)
    
    # --- Step 2: Draw correlated (W1_i, W2_i) at each grid point ---
    # Covariance matrix Sigma_i depends on H_i = H_func(t_i) so it must be rebuilt at every step 

    for i in range(n_steps):
            t_i = t_grid[i + 1]
            H_i = H_func(t_i)                                       
            alpha_i = H_i - 0.5                                     
    
            # Local covariance matrix for step i                    
            var_W1     = dt
            var_W2     = dt ** (2 * H_i) / (2 * H_i)               
            cov_W1_W2  = dt ** (H_i + 0.5) / (H_i + 0.5)          
    
            cov_matrix_i = np.array([                              
                [var_W1,    cov_W1_W2],
                [cov_W1_W2, var_W2   ]])
    
            # Draw one bivariate normal vector from local Sigma_i   
            vec = np.random.multivariate_normal([0, 0], cov_matrix_i)
            W1_increments[i] = vec[0]
            W2_increments[i] = vec[1]    
            
    # --- Step 2: Convolution (memory / distant part) ---
    # gamma_k also depends on alpha_i = H_func(t_i) - 0.5 so we build a time-varying convolution sum

    for i in range(1, n_steps):
        t_i    = t_grid[i + 1]
        H_i    = H_func(t_i)                                  
        alpha_i = H_i - 0.5                                    

        n_steps_per_year = int(n_steps / T) if T > 0 else n_steps

        # b_k_star weights use local alpha_i                  
        k = np.arange(2, i + 2)
        base    = (k ** (alpha_i + 1) - (k - 1) ** (alpha_i + 1)) / (alpha_i + 1)
        b_k_star = np.abs(base) ** (1 / alpha_i)
        coeffs  = (b_k_star / n_steps_per_year) ** alpha_i
    
        # Convolve against past W1 increments up to step i 
        # coeffs[0] multiplies W1[i-1], coeffs[1] multiplies W1[i-2], etc.
        past_W1 = W1_increments[:i][::-1]                      # reverse for convolution
        length  = min(len(coeffs), len(past_W1))
        convolution_sum[i] = np.dot(coeffs[:length], past_W1[:length])
    
    # --- Step 3: Assemble Y using local sqrt(2*H_i)         
    for i in range(n_steps):
        t_i = t_grid[i + 1]
        H_i = H_func(t_i)                                      
        Y[i + 1] = np.sqrt(2 * H_i) * (W2_increments[i] + convolution_sum[i])

    return Y, W1_increments

def simulate_rbergomi_path_mBm(H_func, rho, eta, xi, S0, T, n_steps):
    """
    Simulates a single price and volatility path for the multifractional rough Bergomi model with continuous H(t)
    
    Parameters:
    -----------
    H_func (callable): H:[0,T]->(0,0.5), the Hurst function
    rho (float): leverage correlation
    eta (float): volatility of volatility
    xi (float): initial forward variance
    S0 (float): initial stock price
    T (float): time horizon
    n_steps (int): number of time steps
    """
    
    # Step 1: Generate Y and W1 using frozen-H hybrid scheme
    Y_path, W1_increments = _simulate_hybrid_components_mBm(H_func, n_steps, T)

    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)
    
    # Step 2: Variance process with time-varying correction term
    H_vals = np.array([H_func(t) for t in time_grid])         
    correction = 0.5 * eta**2 * time_grid ** (2 * H_vals) 

    variance_path = xi * np.exp(eta * Y_path - correction)
    
    # Step 3: Stock price — unchanged from fBm version
    dW_perp = np.sqrt(dt) * np.random.randn(n_steps)
    dB = rho * W1_increments + np.sqrt(1 - rho**2) * dW_perp

    S_path = np.zeros(n_steps + 1)
    S_path[0] = S0
    V_t = np.maximum(variance_path[:-1], 1e-12)

    log_increments = -0.5 * V_t * dt + np.sqrt(V_t) * dB
    S_path[1:] = S0 * np.exp(np.cumsum(log_increments))

    volatility_path = np.sqrt(variance_path)

    return time_grid, S_path, volatility_path
        
def generate_smile(H, rho, eta, xi, S0, T, n_steps, n_paths, moneyness):
    """Generates a single implied volatility smile for a given set of rBergomi parameters."""
    print(f"Starting simulation for H = {H:.2f}...")
    start_time = time.time()
    
    model = rBergomi(n_steps=n_steps, T=T, H=H)
    S_paths = model.generate_paths(n_paths, rho, eta, xi, S0, r=0.0, q=0.0)
    S_T = S_paths[:, -1] # Get the terminal stock prices
    
    implied_vols = []
    strike_range = moneyness * S0
    
    for K in strike_range:
        # Use Out-of-the-Money (OTM) options to reduce estimator noise.
        option_type = 'c' if K >= S0 else 'p'
        
        # Calculate the option price via Monte Carlo average of payoffs
        model_price = np.mean(np.maximum(S_T - K, 0) if option_type == 'c' else np.maximum(K - S_T, 0))
        
        # Back out the implied volatility from the model price
        iv = find_implied_vol(model_price, S0, K, T, r=0.0, q=0.0, flag=option_type)
        implied_vols.append(iv)
        
    print(f"Finished H = {H:.2f} in {(time.time() - start_time):.2f} seconds.")
    return implied_vols

def approximate_skew(H, tau, rho, eta, sigma_bar):
    """
    Calculates the approximate ATM forward skew according to equation.
    This function is vectorized to work even if H is a numpy array.
    """
    # Handle cases where H is near zero or invalid to avoid math errors.
    H = np.asfarray(H)
    result = np.full_like(H, np.nan, dtype=float)
    valid_mask = H > 0

    h_valid = H[valid_mask]

    # Helper terms D_H and E_H.
    D_H = np.sqrt(2 * h_valid) / (h_valid + 0.5)
    E_H = D_H / (h_valid + 1.5)

    # First main term of the equation.
    term1 = (rho * eta / 2.0) * E_H * (tau**(h_valid - 0.5))

    # Second main term of the equation.
    bracket_term_part1 = (D_H**2) / (1 + h_valid)
    bracket_term_part2 = 1 + (gamma_function(h_valid + 1.5)**2) / gamma_function(2 * h_valid + 3)
    bracket_term_part3 = (3.0 / 2.0) * (E_H**2)
    bracket = bracket_term_part1 * bracket_term_part2 - bracket_term_part3
    
    term2 = (1.0 / 4.0) * (rho**2) * (eta**2) * sigma_bar * (tau**(2 * h_valid)) * bracket

    result[valid_mask] = term1 + term2
    return result

def calculate_atm_forward_skew_structure(H, rho, eta, xi, S0, maturity_grid, n_paths_skew):
    """
    Calculates the term structure of the ATM forward skew for a given rBergomi parameter set.

    This function simulates the model at various maturities to estimate psi(tau), which is
    a key feature for model validation.
    The skew is calculated at each maturity point using numerical differentiation.
    """
    print(f"\nCalculating skew term structure for H = {H:.2f}...")
    skew_structure = []
    
    # Define a small step 'epsilon' for the finite difference calculation of the derivative.
    epsilon = 0.001 
    
    total_maturities = len(maturity_grid)
    # Iterate through each maturity T in the provided grid to build the term structure.
    for i, T in enumerate(maturity_grid):
        print(f"\r  -> Calculating maturity {i+1}/{total_maturities} (T={T:.2f} years)...", end="")
        
        n_steps = int(252 * T)
        if n_steps == 0: continue # Skip very short maturities that result in zero steps.
            
        # Initialize the rBergomi model for the current maturity T.
        model = rBergomi(n_steps=n_steps, T=T, H=H)
        
        # For each maturity, simulate the terminal asset price S_T using the rBergomi model.
        paths = model.generate_paths(n_paths_skew, rho, eta, xi, S0, r=0.0, q=0.0)
        S_T = paths[:, -1]
        
        # Define two strike prices slightly above and below the at-the-money (ATM) forward price
        # to approximate the derivative at that point. S0 is used as the ATM forward price.
        K_atm = S0
        K_up = K_atm * (1 + epsilon)
        K_down = K_atm * (1 - epsilon)
        
        # Price the two options (K_up and K_down) via Monte Carlo...
        price_up = np.mean(np.maximum(S_T - K_up, 0))
        # ...and then back out their respective implied volatilities.
        iv_up = find_implied_vol(price_up, S0, K_up, T, r=0.0, q=0.0, flag='c')
        
        price_down = np.mean(np.maximum(S_T - K_down, 0))
        iv_down = find_implied_vol(price_down, S0, K_down, T, r=0.0, q=0.0, flag='c')
        
        # Calculate the ATM forward skew using a central difference formula.
        # This approximates the derivative of implied volatility with respect to log-moneyness.
        # The use of abs() aligns with thesis, where psi(tau)
        # is defined as the absolute value of the skew.
        if iv_up is not None and iv_down is not None:
            # Numerical derivative: (change in implied vol) / (change in log-moneyness)
            log_moneyness_skew = abs((iv_up - iv_down) / (np.log(K_up / S0) - np.log(K_down / S0)))
            skew_structure.append(log_moneyness_skew)
        else:
            skew_structure.append(np.nan) # Append NaN if implied vol calculation fails
            
    print(f"\nCalculation finished for H = {H:.2f}.")
    return np.array(skew_structure)

###############################################################################
# Simulate Rough Bergomi Path
###############################################################################

H_test = 0.07      # Hurst Parameter (for a 'rough' volatility path)

def H_sinusoidal(t, alpha_0=0.25, alpha_1=0.225, T=1.0):
    return np.clip(alpha_0 + alpha_1 * np.sin((3 * np.pi * t) / T), 1e-4, 0.4999)

rho_test = -0.9    # Strong negative correlation between price and volatility
eta_test = 1.9     # Volatility of volatility
xi_test = 0.055    # Initial forward variance level (Note: this is variance, not vol)
S0_test = 100      # Initial stock price
T_test = 1.0       # Time horizon of 1 year
n_steps_test = 252 # Number of trading days in a year

# --- Run the Simulation ---
time_grid, price_path, volatility_path = simulate_rbergomi_path_mBm(
    H_func=H_sinusoidal,
    rho=rho_test,
    eta=eta_test,
    xi=xi_test,
    S0=S0_test,
    T=T_test,
    n_steps=n_steps_test
)

# Plot
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
fig.suptitle(r'Simulated Multifractional Rough Bergomi Path', fontsize=18)

# H(t)
H_vals_plot = [H_sinusoidal(t) for t in time_grid]
ax[2].plot(time_grid, H_vals_plot, color='purple', lw=1.5)
ax[2].set_ylabel('$H(t)$', fontsize=13)
ax[2].set_ylim(0, 0.5)
ax[2].grid(True)

# Price Path
price_label = fr'Price Path $(S_t)$, $H(t) = {H_sinusoidal}$'
ax[0].plot(time_grid, price_path, color='blue', label=price_label)
ax[0].set_ylabel(r'Stock Price $(S_t)$', fontsize=14)
ax[0].grid(True)

# Volatility Path
vol_label = fr'Volatility Path $(\sqrt{{v_t}})$, $H(t) = {H_sinusoidal}$'
ax[1].plot(time_grid, volatility_path, color='red', label=vol_label)
ax[1].set_ylabel(r'Volatility $(\sqrt{v_t})$', fontsize=14)
ax[1].set_xlabel(r'Time $(t)$', fontsize=14)
ax[1].grid(True)
ax[1].set_ylim(bottom=0)


# Adjust layout to prevent title overlap.
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig('mBm_rbergomi_path.png', dpi=600)
print("Plot saved as 'rbergomi_path_simulation.png'")
plt.show()

###############################################################################
# Implied Vol Smiles for different H(t)
###############################################################################

print("--- Starting Analysis for T = 0.5 Years ---")

# --- Parameters for T=0.5 ---

T_analysis_0_5 = 0.5
xi_fixed = 0.04 
eta_fixed = 1.8
rho_fixed = -0.8
S0_fixed = 1.0 
n_steps_0_5 = int(252 * T_analysis_0_5)
n_paths_analysis = 250000 
moneyness_range = np.linspace(0.6, 1.3, 30)

# Specifications for H (constant)
H_values = [0.49, 0.40, 0.30, 0.20, 0.10] 
colors_constant = ['blue', 'red', 'green', 'orange', 'purple']

# Specification for H (time-varying)
H_specs_mBm = [
    {
        "name": "Linear",
        "label": r"$H(t)=0.05+0.30t$",
        "func": lambda t: np.clip(0.05 + 0.30 * t, 1e-4, 0.4999),
        "color": "blue"
    },
    {
        "name": "Logistic",
        "label": r"$H(t)=0.05+0.40/(1+e^{-10(t-0.5)})$",
        "func": lambda t: np.clip(0.05 + 0.40 / (1 + np.exp(-10 * (t - 0.5))), 1e-4, 0.4999),
        "color": "red"
    },
    {
        "name": "Quadratic",
        "label": r"$H(t)=0.15+0.45t+0.25t^2$",
        "func": lambda t: np.clip(0.15 + 0.45 * t + 0.25 * t**2, 1e-4, 0.4999),
        "color": "green"
    },
    {
        "name": "Sinusoidal",
        "label": r"$H(t)=0.25+0.225\sin(2\pi t/T)$",
        "func": lambda t: np.clip(0.25 + 0.225 * np.sin((2 * np.pi * t) / T_analysis_0_5), 1e-4, 0.4999),
        "color": "orange"
    },
]

# Step 1: Simulate constant-H smiles 
print("Simulating constant-H smiles...")
all_smiles_constant = {}
for H_val in H_values:
    all_smiles_constant[H_val] = generate_smile(
        H=H_val, rho=rho_fixed, eta=eta_fixed, xi=xi_fixed,
        S0=S0_fixed, T=T_analysis_0_5, n_steps=n_steps_0_5,
        n_paths=n_paths_analysis, moneyness=moneyness_range)

# Step 2: Simulate mBm smiles 
print("Simulating mBm time-varying H(t) smiles...")

def generate_smile_mBm(H_func, rho, eta, xi, S0, T, n_steps, n_paths, moneyness, r_func=None, q_func=None):
    """
    Generates an implied volatility smile for the mBm rough Bergomi model.
    Wraps simulate_mBm_rbergomi_path to collect terminal prices
    across n_paths, then inverts Black-Scholes for each strike.
    """
    print(f"  Simulating {n_paths} paths for mBm...")
    start = time.time()

    # Collect terminal stock prices across all paths
    S_T = np.zeros(n_paths)
    for path_idx in range(n_paths):
        _, S_path, _ = simulate_rbergomi_path_mBm(H_func = H_func, rho = rho, eta = eta, xi = xi,
            S0 = S0, T = T, n_steps = n_steps)
        S_T[path_idx] = S_path[-1]

    # Use actual r and q if provided, otherwise 0
    r_use = r_func(T) if r_func is not None else 0.0
    q_use = q_func(T) if q_func is not None else 0.0

    implied_vols = []
    for K in moneyness * S0:
        flag = 'c' if K >= S0 else 'p'
        model_price = np.mean(np.maximum(S_T - K, 0) if flag == 'c' else np.maximum(K - S_T, 0))
        iv = find_implied_vol(model_price, S0, K, T, r=r_use, q=q_use, flag=flag)
        implied_vols.append(iv)

    print(f"  Done in {time.time()-start:.1f}s")
    return implied_vols

all_smiles_mBm = {}
for spec in H_specs_mBm:
    print(f"Running mBm spec: {spec['name']}")
    all_smiles_mBm[spec["name"]] = generate_smile_mBm(H_func = spec["func"], rho = rho_fixed, eta = eta_fixed,
        xi = xi_fixed, S0 = S0_fixed, T = T_analysis_0_5, n_steps = n_steps_0_5, n_paths  = n_paths_analysis,
        moneyness = moneyness_range)

# Step 3: Plotting
fig, (ax_const, ax_mBm) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

subtitle = (rf'$T={T_analysis_0_5}\ \mathrm{{years}},\ 'rf'\xi={xi_fixed},\ \eta={eta_fixed},\ \rho={rho_fixed}$')

# Constant H 
for i, H_val in enumerate(H_values):
    ax_const.plot(moneyness_range, all_smiles_constant[H_val], color = colors_constant[i], lw = 1.8, label = rf'$H={H_val:.2f}$')

ax_const.set_title('Constant Hurst Parameter\n' + subtitle, fontsize=13)
ax_const.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_const.set_ylabel(rf'Implied Volatility $\sigma_{{\mathrm{{BS}}}}(K,T={T_analysis_0_5})$', fontsize=13)
ax_const.legend(fontsize=11)
ax_const.grid(True, alpha=0.4)

# Time-varying H(t)
for spec in H_specs_mBm:
    ax_mBm.plot(moneyness_range, all_smiles_mBm[spec["name"]], color = spec["color"], lw = 1.8, label = spec["label"])

ax_mBm.set_title('Time-Varying Hurst Function $H(t)$\n' + subtitle,fontsize=13)
ax_mBm.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_mBm.legend(fontsize=10)
ax_mBm.grid(True, alpha=0.4)

fig.suptitle('Implied Volatility Smiles: Constant vs Time-Varying $H$', fontsize=15, y=1.02)
plt.tight_layout()

filename = f'rbergomi_smiles_comparison_T{T_analysis_0_5}y.png'
plt.savefig(filename, dpi=600, bbox_inches='tight')
print(f"Plot saved as '{filename}'")
plt.show()
plt.close(fig)

###############################################################################

print("--- Starting Analysis for T = 1.0 Year ---")

# --- Parameters for T=1.0 Year ---
T_analysis_1_0 = 1.0
xi_fixed = 0.04
eta_fixed = 1.8
rho_fixed = -0.8
S0_fixed = 1.0 
n_steps_1_0 = int(252 * T_analysis_1_0)
n_paths_analysis = 250000 
moneyness_range = np.linspace(0.6, 1.3, 30)

# Specifications for H (constant)
H_values = [0.49, 0.40, 0.30, 0.20, 0.10] 
colors_constant = ['blue', 'red', 'green', 'orange', 'purple']

# Specification for H (time-varying)
H_specs_mBm = [
    {
        "name": "Linear",
        "label": r"$H(t)=0.05+0.30t$",
        "func": lambda t: np.clip(0.05 + 0.30 * t, 1e-4, 0.4999),
        "color": "blue"
    },
    {
        "name": "Logistic",
        "label": r"$H(t)=0.05+0.40/(1+e^{-10(t-0.5)})$",
        "func": lambda t: np.clip(0.05 + 0.40 / (1 + np.exp(-10 * (t - 0.5))), 1e-4, 0.4999),
        "color": "red"
    },
    {
        "name": "Quadratic",
        "label": r"$H(t)=0.15+0.45t+0.25t^2$",
        "func": lambda t: np.clip(0.15 + 0.45 * t + 0.25 * t**2, 1e-4, 0.4999),
        "color": "green"
    },
    {
        "name": "Sinusoidal",
        "label": r"$H(t)=0.25+0.225\sin(2\pi t/T)$",
        "func": lambda t: np.clip(0.25 + 0.225 * np.sin((2 * np.pi * t) / T_analysis_1_0), 1e-4, 0.4999),
        "color": "orange"
    },
]

# Step 1: Simulate constant-H smiles 
print("Simulating constant-H smiles...")
all_smiles_constant = {}
for H_val in H_values:
    all_smiles_constant[H_val] = generate_smile(
        H=H_val, rho=rho_fixed, eta=eta_fixed, xi=xi_fixed,
        S0=S0_fixed, T=T_analysis_1_0, n_steps=n_steps_1_0,
        n_paths=n_paths_analysis, moneyness=moneyness_range)

# Step 2: Simulate mBm smiles 
print("Simulating mBm time-varying H(t) smiles...")

all_smiles_mBm = {}
for spec in H_specs_mBm:
    print(f"Running mBm spec: {spec['name']}")
    all_smiles_mBm[spec["name"]] = generate_smile_mBm(H_func = spec["func"], rho = rho_fixed, eta = eta_fixed,
        xi = xi_fixed, S0 = S0_fixed, T = T_analysis_1_0, n_steps = n_steps_1_0, n_paths  = n_paths_analysis,
        moneyness = moneyness_range)

# Step 3: Plotting
fig, (ax_const, ax_mBm) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

subtitle = (rf'$T={T_analysis_1_0}\ \mathrm{{years}},\ 'rf'\xi={xi_fixed},\ \eta={eta_fixed},\ \rho={rho_fixed}$')

# Constant H 
for i, H_val in enumerate(H_values):
    ax_const.plot(moneyness_range, all_smiles_constant[H_val], color = colors_constant[i], lw = 1.8, label = rf'$H={H_val:.2f}$')

ax_const.set_title('Constant Hurst Parameter\n' + subtitle, fontsize=13)
ax_const.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_const.set_ylabel(rf'Implied Volatility $\sigma_{{\mathrm{{BS}}}}(K,T={T_analysis_1_0})$', fontsize=13)
ax_const.legend(fontsize=11)
ax_const.grid(True, alpha=0.4)

# Time-varying H(t)
for spec in H_specs_mBm:
    ax_mBm.plot(moneyness_range, all_smiles_mBm[spec["name"]], color = spec["color"], lw = 1.8, label = spec["label"])

ax_mBm.set_title('Time-Varying Hurst Function $H(t)$\n' + subtitle,fontsize=13)
ax_mBm.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_mBm.legend(fontsize=10)
ax_mBm.grid(True, alpha=0.4)

fig.suptitle('Implied Volatility Smiles: Constant vs Time-Varying $H$', fontsize=15, y=1.02)
plt.tight_layout()

filename = f'rbergomi_smiles_comparison_T{T_analysis_1_0}y.png'
plt.savefig(filename, dpi=600, bbox_inches='tight')
print(f"Plot saved as '{filename}'")
plt.show()
plt.close(fig)

###############################################################################

print("--- Starting Analysis for T = 2.0 Years ---")

# --- Parameters for T=2.0 ---
T_analysis_2_0 = 2.0
xi_fixed = 0.04
eta_fixed = 1.8
rho_fixed = -0.8
S0_fixed = 1.0 
n_steps_2_0 = int(252 * T_analysis_2_0)
n_paths_analysis = 250000 
moneyness_range = np.linspace(0.6, 1.3, 30)

# Specifications for H (constant)
H_values = [0.49, 0.40, 0.30, 0.20, 0.10] 
colors_constant = ['blue', 'red', 'green', 'orange', 'purple']

# Specification for H (time-varying)
H_specs_mBm = [
    {
        "name": "Linear",
        "label": r"$H(t)=0.05+0.30t$",
        "func": lambda t: np.clip(0.05 + 0.30 * t, 1e-4, 0.4999),
        "color": "blue"
    },
    {
        "name": "Logistic",
        "label": r"$H(t)=0.05+0.40/(1+e^{-10(t-0.5)})$",
        "func": lambda t: np.clip(0.05 + 0.40 / (1 + np.exp(-10 * (t - 0.5))), 1e-4, 0.4999),
        "color": "red"
    },
    {
        "name": "Quadratic",
        "label": r"$H(t)=0.15+0.45t+0.25t^2$",
        "func": lambda t: np.clip(0.15 + 0.45 * t + 0.25 * t**2, 1e-4, 0.4999),
        "color": "green"
    },
    {
        "name": "Sinusoidal",
        "label": r"$H(t)=0.25+0.225\sin(2\pi t/T)$",
        "func": lambda t: np.clip(0.25 + 0.225 * np.sin((2 * np.pi * t) / T_analysis_2_0), 1e-4, 0.4999),
        "color": "orange"
    },
]

# Step 1: Simulate constant-H smiles 
print("Simulating constant-H smiles...")
all_smiles_constant = {}
for H_val in H_values:
    all_smiles_constant[H_val] = generate_smile(
        H=H_val, rho=rho_fixed, eta=eta_fixed, xi=xi_fixed,
        S0=S0_fixed, T=T_analysis_2_0, n_steps=n_steps_2_0,
        n_paths=n_paths_analysis, moneyness=moneyness_range)

# Step 2: Simulate mBm smiles 
print("Simulating mBm time-varying H(t) smiles...")

all_smiles_mBm = {}
for spec in H_specs_mBm:
    print(f"Running mBm spec: {spec['name']}")
    all_smiles_mBm[spec["name"]] = generate_smile_mBm(H_func = spec["func"], rho = rho_fixed, eta = eta_fixed,
        xi = xi_fixed, S0 = S0_fixed, T = T_analysis_2_0, n_steps = n_steps_2_0, n_paths  = n_paths_analysis,
        moneyness = moneyness_range)

# Step 3: Plotting
fig, (ax_const, ax_mBm) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

subtitle = (rf'$T={T_analysis_2_0}\ \mathrm{{years}},\ 'rf'\xi={xi_fixed},\ \eta={eta_fixed},\ \rho={rho_fixed}$')

# Constant H 
for i, H_val in enumerate(H_values):
    ax_const.plot(moneyness_range, all_smiles_constant[H_val], color = colors_constant[i], lw = 1.8, label = rf'$H={H_val:.2f}$')

ax_const.set_title('Constant Hurst Parameter\n' + subtitle, fontsize=13)
ax_const.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_const.set_ylabel(rf'Implied Volatility $\sigma_{{\mathrm{{BS}}}}(K,T={T_analysis_2_0})$', fontsize=13)
ax_const.legend(fontsize=11)
ax_const.grid(True, alpha=0.4)

# Time-varying H(t)
for spec in H_specs_mBm:
    ax_mBm.plot(moneyness_range, all_smiles_mBm[spec["name"]], color = spec["color"], lw = 1.8, label = spec["label"])

ax_mBm.set_title('Time-Varying Hurst Function $H(t)$\n' + subtitle,fontsize=13)
ax_mBm.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_mBm.legend(fontsize=10)
ax_mBm.grid(True, alpha=0.4)

fig.suptitle('Implied Volatility Smiles: Constant vs Time-Varying $H$', fontsize=15, y=1.02)
plt.tight_layout()

filename = f'rbergomi_smiles_comparison_T{T_analysis_2_0}y.png'
plt.savefig(filename, dpi=600, bbox_inches='tight')
print(f"Plot saved as '{filename}'")
plt.show()
plt.close(fig)

###############################################################################

print("--- Starting Analysis for T = 5.0 Years ---")

# --- Parameters for T=5.0 ---
T_analysis_5_0 = 5.0
xi_fixed = 0.04
eta_fixed = 1.8
rho_fixed = -0.8
S0_fixed = 1.0 
n_steps_5_0 = int(252 * T_analysis_5_0)
n_paths_analysis = 250000 
moneyness_range = np.linspace(0.6, 1.3, 30)

# Specifications for H (constant)
H_values = [0.49, 0.40, 0.30, 0.20, 0.10] 
colors_constant = ['blue', 'red', 'green', 'orange', 'purple']

# Specification for H (time-varying)
H_specs_mBm = [
    {
        "name": "Linear",
        "label": r"$H(t)=0.05+0.30t$",
        "func": lambda t: np.clip(0.05 + 0.30 * t, 1e-4, 0.4999),
        "color": "blue"
    },
    {
        "name": "Logistic",
        "label": r"$H(t)=0.05+0.40/(1+e^{-10(t-0.5)})$",
        "func": lambda t: np.clip(0.05 + 0.40 / (1 + np.exp(-10 * (t - 0.5))), 1e-4, 0.4999),
        "color": "red"
    },
    {
        "name": "Quadratic",
        "label": r"$H(t)=0.15+0.45t+0.25t^2$",
        "func": lambda t: np.clip(0.15 + 0.45 * t + 0.25 * t**2, 1e-4, 0.4999),
        "color": "green"
    },
    {
        "name": "Sinusoidal",
        "label": r"$H(t)=0.25+0.225\sin(2\pi t/T)$",
        "func": lambda t: np.clip(0.25 + 0.225 * np.sin((2 * np.pi * t) / T_analysis_5_0), 1e-4, 0.4999),
        "color": "orange"
    },
]

# Step 1: Simulate constant-H smiles 
print("Simulating constant-H smiles...")
all_smiles_constant = {}
for H_val in H_values:
    all_smiles_constant[H_val] = generate_smile(
        H=H_val, rho=rho_fixed, eta=eta_fixed, xi=xi_fixed,
        S0=S0_fixed, T=T_analysis_5_0, n_steps=n_steps_5_0,
        n_paths=n_paths_analysis, moneyness=moneyness_range)

# Step 2: Simulate mBm smiles 
print("Simulating mBm time-varying H(t) smiles...")

all_smiles_mBm = {}
for spec in H_specs_mBm:
    print(f"Running mBm spec: {spec['name']}")
    all_smiles_mBm[spec["name"]] = generate_smile_mBm(H_func = spec["func"], rho = rho_fixed, eta = eta_fixed,
        xi = xi_fixed, S0 = S0_fixed, T = T_analysis_5_0, n_steps = n_steps_5_0, n_paths  = n_paths_analysis,
        moneyness = moneyness_range)

# Step 3: Plotting
fig, (ax_const, ax_mBm) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

subtitle = (rf'$T={T_analysis_5_0}\ \mathrm{{years}},\ 'rf'\xi={xi_fixed},\ \eta={eta_fixed},\ \rho={rho_fixed}$')

# Constant H 
for i, H_val in enumerate(H_values):
    ax_const.plot(moneyness_range, all_smiles_constant[H_val], color = colors_constant[i], lw = 1.8, label = rf'$H={H_val:.2f}$')

ax_const.set_title('Constant Hurst Parameter\n' + subtitle, fontsize=13)
ax_const.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_const.set_ylabel(rf'Implied Volatility $\sigma_{{\mathrm{{BS}}}}(K,T={T_analysis_5_0})$', fontsize=13)
ax_const.legend(fontsize=11)
ax_const.grid(True, alpha=0.4)

# Time-varying H(t)
for spec in H_specs_mBm:
    ax_mBm.plot(moneyness_range, all_smiles_mBm[spec["name"]], color = spec["color"], lw = 1.8, label = spec["label"])

ax_mBm.set_title('Time-Varying Hurst Function $H(t)$\n' + subtitle,fontsize=13)
ax_mBm.set_xlabel(r'Moneyness $(K/F)$', fontsize=13)
ax_mBm.legend(fontsize=10)
ax_mBm.grid(True, alpha=0.4)

fig.suptitle('Implied Volatility Smiles: Constant vs Time-Varying $H$', fontsize=15, y=1.02)
plt.tight_layout()

filename = f'rbergomi_smiles_comparison_T{T_analysis_5_0}y.png'
plt.savefig(filename, dpi=600, bbox_inches='tight')
print(f"Plot saved as '{filename}'")
plt.show()
plt.close(fig)

###############################################################################
# Term Structure of ATM Forward Skew for different H
###############################################################################

# --- mBm specifications (shared across both analyses) ---
mBm_colors = ['blue', 'red', 'green', 'orange']
H_specs_mBm = [
    {
        "name"  : "Linear",
        "label" : r"$H(t)=0.05+0.30t$",
        "func"  : lambda t: np.clip(0.05 + 0.30 * t, 1e-4, 0.4999),
        "color" : "blue"
    },
    {
        "name"  : "Logistic",
        "label" : r"$H(t)=0.05+0.40/(1+e^{-10(t-0.5)})$",
        "func"  : lambda t: np.clip(0.05 + 0.40 / (1 + np.exp(-10 * (t - 0.5))), 1e-4, 0.4999),
        "color" : "red"
    },
    {
        "name"  : "Quadratic",
        "label" : r"$H(t)=0.15+0.45t+0.25t^2$",
        "func"  : lambda t: np.clip(0.15 + 0.45 * t + 0.25 * t**2, 1e-4, 0.4999),
        "color" : "green"
    },
    {
        "name"  : "Sinusoidal",
        "label" : r"$H(t)=0.25+0.225\sin(2\pi t/5)$",
        "func"  : lambda t: np.clip(0.25 + 0.225 * np.sin((2 * np.pi * t) / 5), 1e-4, 0.4999),
        "color" : "orange"
    },
]

def _simulate_hybrid_components_mBm_vectorised(H_func, n_steps, T, n_paths):
    """
    Vectorised frozen-H hybrid scheme for mBm. Generates n_paths simultaneously rather than one at a time.
    This function is a vectorised form of the function "_simulate_hybrid_components_mBm"
    
    Returns:
        Y            : shape (n_paths, n_steps+1)
        W1_increments: shape (n_paths, n_steps)
    """
    dt               = T / n_steps
    t_grid           = np.linspace(0, T, n_steps + 1)
    n_steps_per_year = int(n_steps / T) if T > 0 else n_steps

    # Storage — all paths at once
    W1_increments   = np.zeros((n_paths, n_steps))
    W2_increments   = np.zeros((n_paths, n_steps))
    convolution_sum = np.zeros((n_paths, n_steps))
    Y               = np.zeros((n_paths, n_steps + 1))

    # --- Step 1: Draw (W1_i, W2_i) for all paths at each step ---
    # n_paths draws from bivariate normal — one matrix multiply per step
    for i in range(n_steps):
        t_i       = t_grid[i + 1]
        H_i       = H_func(t_i)
        var_W1    = dt
        var_W2    = dt ** (2 * H_i) / (2 * H_i)
        cov_W1_W2 = dt ** (H_i + 0.5) / (H_i + 0.5)
        cov_mat   = np.array([[var_W1, cov_W1_W2],
                               [cov_W1_W2, var_W2]])

        # Draw n_paths bivariate normals at once        # <-- KEY CHANGE
        vecs = np.random.multivariate_normal(
            [0, 0], cov_mat, size=n_paths
        )
        W1_increments[:, i] = vecs[:, 0]
        W2_increments[:, i] = vecs[:, 1]

    # --- Step 2: Convolution — vectorised dot product across paths ---
    for i in range(1, n_steps):
        t_i     = t_grid[i + 1]
        H_i     = H_func(t_i)
        alpha_i = H_i - 0.5

        k        = np.arange(2, i + 2)
        base     = (k ** (alpha_i+1) - (k-1) ** (alpha_i+1)) / (alpha_i+1)
        b_k_star = np.abs(base) ** (1 / alpha_i)
        coeffs   = (b_k_star / n_steps_per_year) ** alpha_i  # shape (i,)

        past_W1 = W1_increments[:, :i][:, ::-1]  # shape (n_paths, i)
        length  = min(len(coeffs), past_W1.shape[1])

        # Matrix-vector product: (n_paths, length) @ (length,) -> (n_paths,)
        convolution_sum[:, i] = past_W1[:, :length] @ coeffs[:length]

    # --- Step 3: Assemble Y ---
    for i in range(n_steps):
        H_i = H_func(t_grid[i + 1])
        Y[:, i + 1] = np.sqrt(2 * H_i) * (
            W2_increments[:, i] + convolution_sum[:, i]
        )

    return Y, W1_increments

def simulate_rbergomi_paths_mBm_batch(H_func, rho, eta, xi, S0, T,
                                       n_steps, n_paths):
    """
    Simulates n_paths simultaneously for the mBm rough Bergomi model.
    Returns terminal stock prices S_T of shape (n_paths,).
    """
    Y_paths, W1_inc = _simulate_hybrid_components_mBm_vectorised(
        H_func, n_steps, T, n_paths
    )
    dt        = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)

    # Time-varying Ito correction
    H_vals     = np.array([H_func(t) for t in time_grid])
    correction = 0.5 * eta**2 * time_grid ** (2 * H_vals)  # shape (n_steps+1,)

    # Variance process — shape (n_paths, n_steps+1)
    V = xi * np.exp(eta * Y_paths - correction[np.newaxis, :])

    # Stock price
    dW_perp = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
    dB      = rho * W1_inc + np.sqrt(1 - rho**2) * dW_perp

    V_t            = np.maximum(V[:, :-1], 1e-12)
    log_increments = -0.5 * V_t * dt + np.sqrt(V_t) * dB
    log_S_T        = np.log(S0) + np.sum(log_increments, axis=1)

    return np.exp(log_S_T)

def calculate_atm_forward_skew_structure_mBm_fast(H_func, rho, eta, xi, S0, maturity_grid, n_paths_skew):
    """
    Fast ATM skew term structure for mBm using vectorised batch simulation.
    """
    print("  Computing mBm skew term structure (vectorised)...")
    skew_structure = []
    epsilon        = 0.001
    n_maturities   = len(maturity_grid)

    for i, T in enumerate(maturity_grid):
        print(f"\r    Maturity {i+1}/{n_maturities} (T={T:.2f})...", end="")
        n_steps = int(252 * T)
        if n_steps == 0:
            skew_structure.append(np.nan)
            continue

        K_up   = S0 * (1 + epsilon)
        K_down = S0 * (1 - epsilon)

        # All n_paths in one call                       
        S_T = simulate_rbergomi_paths_mBm_batch(
            H_func, rho, eta, xi, S0, T, n_steps, n_paths_skew
        )

        price_up   = np.mean(np.maximum(S_T - K_up,   0))
        price_down = np.mean(np.maximum(S_T - K_down, 0))

        iv_up   = find_implied_vol(price_up,   S0, K_up,   T, 0, 0, 'c')
        iv_down = find_implied_vol(price_down, S0, K_down, T, 0, 0, 'c')

        if iv_up is not None and iv_down is not None:
            skew = abs(
                (iv_up - iv_down)
                / (np.log(K_up / S0) - np.log(K_down / S0))
            )
            skew_structure.append(skew)
        else:
            skew_structure.append(np.nan)

    print()
    return np.array(skew_structure)

def calculate_atm_forward_skew_structure_mBm(H_func, rho, eta, xi, S0, maturity_grid, n_paths_skew):
    """
    ATM forward skew term structure for the mBm rough Bergomi model.
    Identical logic to calculate_atm_forward_skew_structure but uses
    simulate_mBm_rbergomi_path as the simulation engine.
    """
    print(f"  Computing mBm skew term structure...")
    skew_structure = []
    epsilon = 0.001
    n_maturities = len(maturity_grid)

    for i, T in enumerate(maturity_grid):
        print(f"\r    Maturity {i+1}/{n_maturities} (T={T:.2f})...", end="")
        n_steps = int(252 * T)
        if n_steps == 0:
            skew_structure.append(np.nan)
            continue

        K_up   = S0 * (1 + epsilon)
        K_down = S0 * (1 - epsilon)

        # Collect terminal prices
        S_T = np.zeros(n_paths_skew)
        for j in range(n_paths_skew):
            _, S_path, _ = simulate_rbergomi_path_mBm(H_func=H_func, rho=rho, eta=eta, xi=xi, S0=S0, T=T, n_steps=n_steps)
            S_T[j] = S_path[-1]

        price_up   = np.mean(np.maximum(S_T - K_up,   0))
        price_down = np.mean(np.maximum(S_T - K_down, 0))

        iv_up   = find_implied_vol(price_up,   S0, K_up,   T, 0, 0, 'c')
        iv_down = find_implied_vol(price_down, S0, K_down, T, 0, 0, 'c')

        if iv_up is not None and iv_down is not None:
            skew = abs((iv_up - iv_down) / (np.log(K_up / S0) - np.log(K_down / S0)))
            skew_structure.append(skew)
        else:
            skew_structure.append(np.nan)

    print()
    return np.array(skew_structure)

###############################################################################
# Analysis 1: Constant H 
###############################################################################

print("\n Analysis 1: Constant H...")

# Fixed Parameters 
eta_fixed            = 1.8
rho_fixed            = -0.8
xi_fixed             = 0.04
S0_fixed             = 1.0
maturity_grid        = np.linspace(0.05, 5, 50)
H_values_1      = [0.05, 0.10, 0.20, 0.30, 0.40]
colors_1        = ['red', 'blue', 'green', 'purple', 'orange']
n_paths_skew_1  = 250000
H_values_2      = [0.02, 0.04, 0.06, 0.08, 0.10]
colors_2        = ['darkred', 'red', 'orangered', 'orange', 'gold']
n_paths_skew_2  = 250000


all_skews_1 = {}
total_start_time_1 = time.time()

for H_val in H_values_1:
    all_skews_1[H_val] = calculate_atm_forward_skew_structure(
        H=H_val, rho=rho_fixed, eta=eta_fixed, xi=xi_fixed, S0=S0_fixed,
        maturity_grid=maturity_grid, n_paths_skew=n_paths_skew_1
    )
print(f"\nTotal simulation time for Analysis 1: {(time.time() - total_start_time_1)/60:.2f} minutes.")

# --- Plotting for Analysis 1 ---
plt.figure(figsize=(10, 7))
for i, H_val in enumerate(H_values_1):
    plt.plot(maturity_grid, all_skews_1[H_val], label=f'H={H_val:.2f}', color=colors_1[i])

plt.title(r'Term Structure of ATM Forward Skew - Constant Hurst Parameters' + '\n' + rf'$\eta={eta_fixed}$, $\rho={rho_fixed}$')
plt.xlabel('$\\tau$ (Years)')
plt.ylabel('ATM Forward Skew $\\psi(\\tau)$')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.savefig('atm_skew_H_05_40.png', dpi=600)
plt.show()
plt.close()

all_skews_2 = {}
total_start_time_2 = time.time()

for H_val in H_values_2:
    all_skews_2[H_val] = calculate_atm_forward_skew_structure(
        H=H_val, rho=rho_fixed, eta=eta_fixed, xi=xi_fixed, S0=S0_fixed,
        maturity_grid=maturity_grid, n_paths_skew=n_paths_skew_2
    )
print(f"\nTotal simulation time for Analysis 2: {(time.time() - total_start_time_2)/60:.2f} minutes.")

plt.figure(figsize=(10, 7))
for i, H_val in enumerate(H_values_2):
    plt.plot(maturity_grid, all_skews_2[H_val], label=rf'$H={H_val:.2f}$', color=colors_2[i])

plt.title(r'Term Structure of ATM Forward Skew - Constant Hurst Parameters' + '\n' + rf'$\eta={eta_fixed}$, $\rho={rho_fixed}$')
plt.xlabel(r'$\tau$ (Years)')
plt.ylabel(r'ATM Forward Skew $\psi(\tau)$')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.savefig('atm_skew_H_002_010.png', dpi=600)
plt.show()
plt.close()

###############################################################################
# Analysis 2: Time Varying H 
###############################################################################

print("\n Analysis 2: Time Varying H...")

# Fixed Parameters 
eta_fixed            = 1.8
rho_fixed            = -0.8
xi_fixed             = 0.04
S0_fixed             = 1.0
maturity_grid        = np.linspace(0.05, 5, 50)
colors_3        = ['blue', 'darkred', 'purple', 'green', 'gold']
n_paths_skew_3  = 250000


# mBm H(t)
all_skews_mBm = {}
t0_3m = time.time()
for spec in H_specs_mBm:
    print(f"  mBm spec: {spec['name']}...")
    all_skews_mBm[spec["name"]] = \
        calculate_atm_forward_skew_structure_mBm_fast(H_func=spec["func"], rho=rho_fixed, eta=eta_fixed,
            xi=xi_fixed, S0=S0_fixed, maturity_grid=maturity_grid, n_paths_skew=n_paths_skew_3)
        
print(f"mBm done in {(time.time()-t0_3m)/60:.2f} min")


plt.figure(figsize=(10, 7))
for spec in H_specs_mBm:
    plt.plot(maturity_grid, all_skews_mBm[spec["name"]], color=spec["color"], lw=1.8, label=spec["label"])


plt.title(r'Term Structure of ATM Forward Skew - Time Varying H(t)' + '\n' + rf'$\eta={eta_fixed}$, $\rho={rho_fixed}$')
plt.xlabel(r'$\tau$ (Years)')
plt.ylabel(r'ATM Forward Skew $\psi(\tau)$')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.savefig('atm_skew_H_003_010.png', dpi=600)
plt.show()
plt.close()


###############################################################################
# H(t) reference plot — shows what each function looks like over [0,5]
###############################################################################

t_ref = np.linspace(0, 5, 500)
fig_h, ax_h = plt.subplots(figsize=(9, 4))

for spec in H_specs_mBm:
    H_vals = [spec["func"](t) for t in t_ref]
    ax_h.plot(t_ref, H_vals, color=spec["color"], lw=1.8, label=spec["label"])

# Dashed reference lines for constant H values
for H_val, col in zip(H_values_1, colors_1):
    ax_h.axhline(H_val, color=col, lw=0.9, ls='--', alpha=0.5, label=rf"$H={H_val:.2f}$ (const)")

ax_h.set_xlabel(r"$t$ (Years)", fontsize=13)
ax_h.set_ylabel(r"$H(t)$", fontsize=13)
ax_h.set_ylim(0, 0.55)
ax_h.set_title(r"Hurst Functions over $[0, 5]$ Years", fontsize=13)
ax_h.legend(fontsize=9, ncol=2)
ax_h.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("H_functions_skew_reference.png", dpi=600, bbox_inches="tight")
print("Saved: H_functions_skew_reference.png")
plt.show()
plt.close(fig_h)

###############################################################################
# Approximate ATM Forward Skew — Constant H vs mBm H(t)
###############################################################################

# Fixed Parameters 
rho_fixed       = -0.8
eta_fixed       = 1.8
xi_fixed        = 0.04
sigma_bar_fixed = np.sqrt(xi_fixed)

tau_days              = [2, 5, 10, 15, 20, 30]
trading_days_per_year = 252.0
tau_years             = [d / trading_days_per_year for d in tau_days]
H_range               = np.linspace(0.01, 0.5, 400)

# mBm specifications 
H_specs_mBm_skew = [
    {
        "name"  : "Linear",
        "label" : r"$H(t)=0.05+0.30t$",
        "func"  : lambda t: np.clip(0.05 + 0.30 * t, 1e-4, 0.4999),
        "color" : "blue"
    },
    {
        "name"  : "Logistic",
        "label" : r"$H(t)=0.05+0.40/(1+e^{-10(t-0.5)})$",
        "func"  : lambda t: np.clip(0.05 + 0.40 / (1 + np.exp(-10 * (t - 0.5))), 1e-4, 0.4999),
        "color" : "green"
    },
    {
        "name"  : "Quadratic",
        "label" : r"$H(t)=0.15+0.45t+0.25t^2$",
        "func"  : lambda t: np.clip(0.15 + 0.45 * t + 0.25 * t**2, 1e-4, 0.4999),
        "color" : "purple"
    },
    {
        "name"  : "Sinusoidal",
        "label" : r"$H(t)=0.25+0.225\sin(2\pi t/5)$",
        "func"  : lambda t: np.clip(0.25 + 0.225 * np.sin((2 * np.pi * t) / 5), 1e-4, 0.4999),
        "color" : "orange"
    },
]

###############################################################################
# Figure 1: Skew vs H for constant H 
###############################################################################

fig1, axes1 = plt.subplots(3, 2, figsize=(12, 15))
axes1 = axes1.flatten()

for i, tau_day in enumerate(tau_days):
    ax  = axes1[i]
    tau = tau_years[i]

    skew_values    = np.abs(approximate_skew(H_range, tau, rho_fixed, eta_fixed, sigma_bar_fixed))
    max_skew_index = np.nanargmax(skew_values)
    H_max = H_range[max_skew_index]

    ax.plot(H_range, skew_values, 'r-', lw=1.8, label='Approximate Skew (constant $H$)')
    ax.axvline(x=H_max, color='b', ls='--', label=f'Max at $H={H_max:.3f}$')
    ax.set_title(rf'$\tau$ = {int(tau_day)} Business Days')
    ax.set_xlabel(r'$H$')
    ax.set_ylabel(r'ATM Forward Skew $\psi(\tau)$')
    ax.legend(fontsize=9)
    ax.grid(True, ls='--', lw=0.5)
    ax.set_xlim(0, 0.55)

fig1.suptitle(rf'Approximate ATM Forward Skew vs. $H$ (Constant $H$)' '\n' rf'$\rho={rho_fixed}$, $\eta={eta_fixed}$, $\xi={xi_fixed}$', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('atm_skew_constant_H_6maturities.png', dpi=600)
print("Saved: atm_skew_constant_H_6maturities.png")
plt.show()
plt.close(fig1)

###############################################################################
# Figure 2: mBm version — for each maturity tau, display the constant-H curve
#           and indicate the skew implied by each H(t) spec at H(tau)
###############################################################################
fig2, axes2 = plt.subplots(3, 2, figsize=(12, 15))
axes2 = axes2.flatten()

for i, tau_day in enumerate(tau_days):
    ax  = axes2[i]
    tau = tau_years[i]

    # Constant-H skew curve 
    skew_values = np.abs(approximate_skew(H_range, tau, rho_fixed, eta_fixed, sigma_bar_fixed))
    ax.plot(H_range, skew_values, 'r-', lw=1.8, alpha=0.5, label='Constant $H$ skew curve')

    #  Evaluate H_func(tau), read off skew 
    for spec in H_specs_mBm_skew:
        H_at_tau   = spec["func"](tau)           # local H at this maturity
        skew_at_H  = float(np.abs(approximate_skew(np.array([H_at_tau]), tau, rho_fixed, eta_fixed, sigma_bar_fixed)))

        # Vertical line at H(tau) on the constant-H curve
        ax.axvline(x=H_at_tau, color=spec["color"], ls='--', lw=1.2, alpha=0.7)
        # Dot where (H(tau), skew(H(tau))) sits on the curve
        ax.scatter(H_at_tau, skew_at_H, color=spec["color"], s=60, zorder=5, label=rf"{spec['label']}: $H(\tau)={H_at_tau:.3f}$")

    ax.set_title(rf'$\tau$ = {int(tau_day)} Business Days')
    ax.set_xlabel(r'$H$')
    ax.set_ylabel(r'ATM Forward Skew $\psi(\tau)$')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, ls='--', lw=0.5)
    ax.set_xlim(0, 0.55)

fig2.suptitle(
    rf'Approximate ATM Forward Skew — mBm $H(t)$ Specifications' '\n' rf'Dots show $\psi(\tau)$ at $H(\tau)$ for each $H(t)$ spec'
    '\n' rf'$\rho={rho_fixed}$, $\eta={eta_fixed}$, $\xi={xi_fixed}$', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.985])
plt.savefig('atm_skew_mBm_H_6maturities.png', dpi=600)
print("Saved: atm_skew_mBm_H_6maturities.png")
plt.show()
plt.close(fig2)

###############################################################################
# Figure 3: Skew vs maturity tau — shows how skew evolves across maturities
#           for each mBm H(t) spec vs constant-H baselines
###############################################################################
tau_fine = np.linspace(0.005, 30/252, 400)   # 0 to 30 business days

fig3, ax3 = plt.subplots(figsize=(11, 6))

# Constant H baselines
for H_const, col in zip([0.05, 0.10, 0.20], ['silver', 'grey', 'black']):
    skew_const = np.abs(approximate_skew(np.full_like(tau_fine, H_const), tau_fine, rho_fixed, eta_fixed, sigma_bar_fixed))
    ax3.plot(tau_fine * 252, skew_const, color=col, lw=1.2, ls='-', label=rf'Const $H={H_const:.2f}$')

# mBm specs: skew at H(tau) for each tau
for spec in H_specs_mBm_skew:
    skew_mBm = np.array([float(np.abs(approximate_skew(np.array([spec["func"](t)]), t, rho_fixed, eta_fixed, sigma_bar_fixed))) for t in tau_fine])
    ax3.plot(tau_fine * 252, skew_mBm, color=spec["color"], lw=1.8, ls="--", label=spec["label"])

# Mark the six specific maturities
for tau_day in tau_days:
    ax3.axvline(tau_day, color='lightgrey', lw=0.8, ls='--')

ax3.set_xlabel(r'$\tau$ (Business Days)', fontsize=13)
ax3.set_ylabel(r'ATM Forward Skew $\psi(\tau)$', fontsize=13)
ax3.set_title( r'Approximate ATM Forward Skew vs Maturity — mBm $H(t)$ Specifications'
    '\n' rf'$\rho={rho_fixed}$, $\eta={eta_fixed}$, $\xi={xi_fixed}$', fontsize=13)
ax3.legend(fontsize=9, ncol=2)
ax3.grid(True, alpha=0.4)
ax3.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('atm_skew_mBm_vs_maturity.png', dpi=600)
print("Saved: atm_skew_mBm_vs_maturity.png")
plt.show()
plt.close(fig3)

###############################################################################
# Implied vol for eta and rho
###############################################################################

# These parameters are kept constant across both analyses to isolate the effects of rho and eta.
common_params = {
    'T': 2.0,          # Time to maturity in years
    'H': 0.1,          # Hurst parameter, controlling the "roughness" of volatility
    'xi': 0.04,       # Initial forward variance (flat curve assumption)
    'S0': 1.0,         # Initial stock price, normalized to 1 for easy moneyness calculation
    'n_paths': 250000 # Number of Monte Carlo paths for simulation accuracy
}

# --- Different rho ---
print("--- Generating Smiles for different rho ---")
eta_fixed_a = 1.8
moneyness_range_a = np.linspace(0.6, 1.4, 40)
rho_values = [-0.9, -0.7, -0.5, -0.3, -0.1]
colors_a = ['b', 'r', 'g', 'c', 'm']

# Iterate through each specified rho value to generate a separate smile curve.
plt.figure(figsize=(10,7))
for i, rho_val in enumerate(rho_values):
    n_steps = int(252 * common_params['T']) # Daily steps
    
    # Call the simulation function with the current rho value and other fixed parameters.
    implied_vols = generate_smile(
        H=common_params['H'], rho=rho_val, eta=eta_fixed_a, xi=common_params['xi'], 
        S0=common_params['S0'], T=common_params['T'], n_steps=n_steps, 
        n_paths=common_params['n_paths'], moneyness=moneyness_range_a
    )
    # Plot the resulting implied volatility smile on the first subplot.
    plt.plot(moneyness_range_a, implied_vols, label=f'$\\rho={rho_val:.1f}$', color=colors_a[i])

# Set the title and labels to match the thesis figure.
plt.title(f"Implied Volatility Smiles for Different $\\rho$; T={common_params['T']} years, $\\xi={common_params['xi']}$, $\\eta={eta_fixed_a}$, H={common_params['H']}", fontsize=12)
plt.xlabel(r'Moneyness $(K/F)$', fontsize=10)
plt.ylabel(f'Implied Volatility $\\sigma_{{BS}}(K, T={common_params["T"]})$', fontsize=10)
plt.legend(title=r'Correlation $\rho$')
plt.grid(True)


# --- Plot (b): Different eta ---
# This block shows the effect of varying the volatility-of-volatility parameter eta.
# Eta primarily controls the curvature (smile) of the volatility surface.
print("\n--- Generating Smiles for different eta ---")
rho_fixed_b = -0.8
moneyness_range_b = np.linspace(0.6, 1.5, 40)
eta_values = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
colors_b = ['orange','purple','blue', 'red', 'green', 'c', 'm']

# Iterate through each specified eta value to generate a separate smile curve.
plt.figure(figsize=(10,7))
for i, eta_val in enumerate(eta_values):
    n_steps = int(252 * common_params['T'])
    
    # Call the simulation function with the current eta value and other fixed parameters.
    implied_vols = generate_smile(
        H=common_params['H'], rho=rho_fixed_b, eta=eta_val, xi=common_params['xi'], 
        S0=common_params['S0'], T=common_params['T'], n_steps=n_steps, 
        n_paths=common_params['n_paths'], moneyness=moneyness_range_b)
    plt.plot(moneyness_range_b, implied_vols, label=f'$\\eta={eta_val:.2f}$', color=colors_b[i])


plt.title(f"Implied Volatility Smiles for Different $\\eta$; T={common_params['T']} years, $\\xi={common_params['xi']}$, $\\rho={rho_fixed_b}$, H={common_params['H']}", fontsize=12)
plt.xlabel(r'Moneyness $(K/F)$', fontsize=10)
plt.ylabel(f'Implied Volatility $\\sigma_{{BS}}(K, T={common_params["T"]})$', fontsize=10)
plt.legend(title=r'Vol-of-Vol $\eta$')
plt.grid(True)

# Adjust layout to prevent titles from overlapping and show the final plot.
plt.tight_layout()
plt.savefig('rBergomi_rho_eta_smiles.png', dpi=600)
print("Plot saved successfully as 'rBergomi_rho_eta_smiles.png'")
plt.show()
plt.close()

###############################################################################

if __name__ == '__main__':
    # --- Parameters ---
    T_analysis = 1.0
    H_analysis = 0.1
    rho_eta_product = -1.44
    xi_analysis = 0.04
    S0_analysis = 1.0
    
    rho_values_to_test = [-0.90, -0.85, -0.80, -0.75, -0.70]

    # --- Simulation quality ---
    n_steps_analysis = int(252 * T_analysis)
    n_paths_analysis = 250000 

    # --- Moneyness and Strikes ---
    # The moneyness range is set to be between 0.6 and 1.4.
    moneyness_range_k_f = np.linspace(0.6, 1.4, 50)

    # --- Simulation Loop ---
    all_smiles_analysis = {}
    for rho_val in rho_values_to_test:
        # Key constraint: rho * eta = -1.71
        eta_val = rho_eta_product / rho_val
        
        # Calling the 'generate_smile' function
        smile_data = generate_smile(
            H=H_analysis, rho=rho_val, eta=eta_val, xi=xi_analysis, S0=S0_analysis,
            T=T_analysis, n_steps=n_steps_analysis, n_paths=n_paths_analysis,
            moneyness=moneyness_range_k_f
        )
        all_smiles_analysis[rho_val] = smile_data

    # --- Plot Results ---
    plt.figure(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(rho_values_to_test)))

    for i, rho_val in enumerate(rho_values_to_test):
        plt.plot(moneyness_range_k_f, all_smiles_analysis[rho_val], color=colors[i], lw=2, label=fr'$\rho={rho_val:.2f}$')

    title_text = (f'Smiles with fixed product $\\rho \\times \\eta = {rho_eta_product}$ and different $\\rho$\n'
                  fr'$T={T_analysis}$ year, $H={H_analysis}$, $\xi={xi_analysis}$')
    plt.title(title_text, fontsize=14)
    plt.xlabel(r'Moneyness $(K/F)$', fontsize=14)
    plt.ylabel(f'Implied Volatility $\\sigma_{{BS}}(K, T)$', fontsize=12)
    plt.legend(title=fr'Correlation $\rho$') 
    plt.grid(True)
    plt.savefig('smiles_fixed_product.png', dpi=600)
    print("Plot saved successfully as 'smiles_fixed_product.png'")
    plt.show()
    plt.close()
    
    
















    

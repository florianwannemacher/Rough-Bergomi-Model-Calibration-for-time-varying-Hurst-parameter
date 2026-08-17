import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm
from scipy.optimize import brentq, minimize, differential_evolution
from scipy.interpolate import interp1d, PchipInterpolator
from numpy.fft import rfft, irfft
import heapq

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12
})

# ==============================================================================
# UTILITY FUNCTIONS (BSM, IV Solver, Rates)
# ==============================================================================

def bsm_price(S, K, T, r, q, sigma, flag):
    """
    Calculates the Black-Scholes-Merton option price.
    """
    # Handle edge cases T=0 or sigma=0
    if T <= 1e-6 or sigma <= 1e-6:
        if flag == 'c':
            return np.maximum(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        else:
            return np.maximum(0, K * np.exp(-r * T) - S * np.exp(-q * T))
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if flag == 'c':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def find_implied_vol(target_price, S, K, T, r, q, flag):
    """
    Numerically solves for the implied volatility using Brent's method.
    """
    # Basic check for arbitrage bounds (price must be >= intrinsic value)
    intrinsic_value = bsm_price(S, K, T, r, q, 1e-6, flag)
    if target_price < intrinsic_value:
        # If the MC price is slightly below intrinsic due to noise, we might clip it, 
        # but often it's better to return NaN to signal an issue in the cost function.
        return np.nan

    objective = lambda sigma: bsm_price(S, K, T, r, q, sigma, flag) - target_price
    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-6, rtol=1e-6, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan

def get_risk_free_rate_data():
    """Provides US Treasury data (Example Data)."""
    # In a real scenario, this would fetch live data or read from a dynamic file.
    treasury_data = {
        'Maturity_str': ['1-Month', '1.5-Months', '2-Months' ,'3-Months', '4-Months', '6-Months', '1-Year', '2-Years', '3-Years', '5-Years', '7-Years', '10-Years', '20-Years', '30-Years'],
        'Rate_str': ['3.76 %', '3.86 %', '3.90 %', '3.90 %', '4.02 %', '4.07 %', '4.09 %', '4.26 %', '4.31 %', '4.35 %', '4.47 %', '4.61 %', '5.11 %', '5.09 %']
    }
    return pd.DataFrame(treasury_data)

def create_risk_free_rate_interpolator(treasury_df):
    """Creates a continuous risk-free rate function (r_func)."""
    maturity_map = {
        '1-Month': 1/12, '1.5-Months': 1.5/12, '2-Months': 2/12, '3-Months': 3/12, '4-Months': 4/12, 
        '6-Months': 6/12, '1-Year': 1.0, '2-Years': 2.0, '3-Years': 3.0, '5-Years': 5.0, 
        '7-Years': 7.0, '10-Years': 10.0, '20-Years': 20.0, '30-Years': 30.0
    }
    treasury_df = treasury_df.copy()
    treasury_df['T'] = treasury_df['Maturity_str'].map(maturity_map)
    treasury_df['Rate'] = treasury_df['Rate_str'].str.replace(' %', '', regex=False).astype(float) / 100
    # Linear interpolation with flat extrapolation is generally robust
    return interp1d(treasury_df['T'].values, treasury_df['Rate'].values, kind='linear', fill_value="extrapolate")

# ==============================================================================
# FORWARD VARIANCE BOOTSTRAPPING FUNCTIONS
# ==============================================================================

def extract_atm_volatility_curve(surface_df, S0):
    """
    Extracts the At-the-Money (ATM Forward) volatility curve.
    """
    print("\n--- Extracting ATM Volatility Curve from Market Data ---")
    df = surface_df.copy()
    
    # Calculate Forward Price F = S0 * exp((r-q)*T)
    df['forward_price'] = S0 * np.exp((df['risk_free_rate'] - df['dividend_yield']) * df['time_to_maturity'])
    
    # Calculate distance from ATM Forward
    df['atm_distance'] = np.abs(df['strike'] - df['forward_price'])
    
    # Select the option closest to the forward price for each maturity
    atm_indices = df.groupby('time_to_maturity')['atm_distance'].idxmin()
    
    atm_curve = df.loc[atm_indices]
    
    atm_curve_formatted = atm_curve[['time_to_maturity', 'implied_vol']].rename(
        columns={'time_to_maturity': 'T', 'implied_vol': 'market_iv'}
    ).sort_values('T')
    
    print(f"Extracted {len(atm_curve_formatted)} ATM points.")
    return atm_curve_formatted

def create_xi_from_market_atm(market_atm_curve_df):
    """
    Bootstraps the forward variance curve (xi_func) from ATM implied volatilities.
    """
    print("\n--- Creating market-consistent xi_func from ATM Volatilities ---")
    
    df = market_atm_curve_df.sort_values('T').copy()

    # Ensure T=0 exists
    if not df.empty and not np.isclose(df['T'].iloc[0], 0.0):
        initial_point = pd.DataFrame({'T': [0.0], 'market_iv': [df['market_iv'].iloc[0]]})
        df = pd.concat([initial_point, df]).reset_index(drop=True)
    
    # 1. Total Variance (W = IV^2 * T)
    df['total_variance'] = df['T'] * (df['market_iv']**2)
    
    # 2. Forward Variance (xi = dW/dT) using discrete differences
    df['forward_variance'] = (df['total_variance'].diff()) / (df['T'].diff())
    
    # 3. Fill T=0 (Instantaneous variance)
    if not df.empty:
       df.iat[0, df.columns.get_loc('forward_variance')] = df['market_iv'].iat[0]**2
    
    # 4. Ensure non-negative forward variances (Arbitrage constraint)
    df['forward_variance'] = df['forward_variance'].clip(lower=1e-6)
    
    # 5. Create interpolation function (xi_func) with flat extrapolation
    new_xi_func = interp1d(
        df['T'].values,
        df['forward_variance'].values,
        kind='linear', # Linear is often more stable than cubic for bootstrapped data
        bounds_error=False,
        fill_value=(df['forward_variance'].iloc[0], df['forward_variance'].iloc[-1])
    )

    print("Market-consistent xi_func created successfully.")
    return new_xi_func

# ==============================================================================
# DATA PREPARATION AND GRID SELECTION
# ==============================================================================

def prepare_full_surface_and_rates(file_path, constant_dividend_yield=0.0115):
    """
    Reads, cleans, and processes the raw options data file.
    """
    print("--- Reading and processing raw options data... ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None, None, None
        
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Filter for liquid options with valid prices
    df_filtered = df[(df['bid'] > 0) & (df['ask'] > 0) & (df['trade_volume'] > 10)].copy()
    
    if df_filtered.empty:
        print("Error: No valid options found after initial filtering.")
        return None, None, None, None, None

    # Time to Maturity Calculation
    df_filtered['quote_date'] = pd.to_datetime(df_filtered['quote_datetime'])
    df_filtered['expiration'] = pd.to_datetime(df_filtered['expiration'])
    df_filtered['time_to_maturity'] = (df_filtered['expiration'] - df_filtered['quote_date']).dt.days / 365.0
    # Exclude very short maturities
    df_filtered = df_filtered[df_filtered['time_to_maturity'] > (7 / 365.0)].reset_index(drop=True)
    
    if df_filtered.empty:
        print("Error: No options found after maturity filtering.")
        return None, None, None, None, None

    # Determine Spot and Date
    spot_price = (df_filtered['underlying_bid'].iloc[0] + df_filtered['underlying_ask'].iloc[0]) / 2
    quote_date = df_filtered['quote_date'].iloc[0]
    
    # Rates (r_func and q_func)
    treasury_df = get_risk_free_rate_data()
    r_func = create_risk_free_rate_interpolator(treasury_df)
    q_val = constant_dividend_yield
    q_func = lambda T: q_val
    
    df_filtered['risk_free_rate'] = df_filtered['time_to_maturity'].apply(r_func)
    df_filtered['dividend_yield'] = q_val
    df_filtered['mid_price'] = (df_filtered['bid'] + df_filtered['ask']) / 2
    
    # Calculate Market Implied Volatility
    df_filtered['implied_vol'] = df_filtered.apply(
        lambda row: find_implied_vol(row['mid_price'], spot_price, row['strike'], row['time_to_maturity'], row['risk_free_rate'], row['dividend_yield'], row['option_type'].lower()),
        axis=1
    )
    
    final_surface_df = df_filtered.dropna(subset=['implied_vol']).copy()
    
    print(f"Data processing complete. Found {len(final_surface_df)} valid options. S0={spot_price:.2f}")
    return spot_price, quote_date, final_surface_df, r_func, q_func

def create_calibration_grid(full_surface_df, spot_price, num_options=50):
    """
    Selects a representative subset of options for calibration and displays the summary.
    """
    print(f"\n--- Selecting approx. {num_options} representative options for calibration... ---")
    
    # Calculate moneyness (K/S)
    if 'moneyness' not in full_surface_df.columns:
        # Use .loc to ensure we modify the DataFrame correctly
       full_surface_df.loc[:, 'moneyness'] = full_surface_df['strike'] / spot_price
    
    # Define target grid
    target_maturities_years = np.array([1/12, 3/12, 6/12, 1, 2, 5])
    target_moneyness_levels = np.array([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25])

    grid_points = pd.DataFrame([{'T': t, 'M': m} for t in target_maturities_years for m in target_moneyness_levels])

    # Find closest options
    selected_indices = set()
    for _, point in grid_points.iterrows():
        if full_surface_df.empty:
            continue
        # Calculate Euclidean distance in (T, M) space
        distances = np.sqrt(
            (full_surface_df['time_to_maturity'] - point['T'])**2 +
            (full_surface_df['moneyness'] - point['M'])**2
        )
        # Ensure distances are valid before finding minimum
        if not distances.empty and not distances.isnull().all():
            closest_idx = distances.idxmin()
            selected_indices.add(closest_idx)

    if not selected_indices:
        print("Warning: Could not select any options for the calibration grid.")
        return pd.DataFrame()

    calibration_df = full_surface_df.loc[list(selected_indices)].reset_index(drop=True)
    
    # Display Summary
    print("\n--- Summary of Options Selected for Calibration ---")
    display_cols = [
        'expiration', 'strike', 'option_type', 'time_to_maturity',
        'moneyness', 'implied_vol'
    ]
    display_df = calibration_df[display_cols].copy()

    # Formatting
    display_df['expiration'] = display_df['expiration'].dt.strftime('%Y-%m-%d')
    display_df['T (Years)'] = display_df['time_to_maturity'].round(4)
    display_df['Moneyness'] = display_df['moneyness'].round(3)
    display_df['Market IV (%)'] = (display_df['implied_vol'] * 100).round(2)

    final_display_cols = ['expiration', 'T (Years)', 'strike', 'option_type', 'Moneyness', 'Market IV (%)']

    # Display full table
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(display_df[final_display_cols].sort_values(by=['T (Years)', 'strike']).reset_index(drop=True))

    print(f"\nSelection complete. {len(calibration_df)} unique options selected.")
    return calibration_df

# ==============================================================================
# ROUGH BERGOMI MODEL (Hybrid Scheme Implementation)
# ==============================================================================

class rBergomi:
    """
    A fully vectorized rBergomi simulation engine using the Hybrid Scheme.
    """
    def __init__(self, n_steps, T, H):
        # Initialization (H, T, time grid, dt)
        self.T = float(T)
        self.H = float(H)
        self.alpha = self.H - 0.5
        self.n_steps = int(n_steps)
        self.dt = self.T / self.n_steps if self.n_steps > 0 else 0
        self.time_grid = np.linspace(0, T, self.n_steps + 1)
        
        # Pre-compute covariance matrix for the Hybrid Scheme (W1, W2)
        var_W1 = self.dt
        var_W2 = (self.dt**(2 * self.H)) / (2 * self.H) if self.H > 0 else 0
        cov_W1_W2 = (self.dt**(self.H + 0.5)) / (self.H + 0.5) if self.H > -0.5 else 0
        self.cov_matrix = np.array([[var_W1, cov_W1_W2], [cov_W1_W2, var_W2]])

        # Pre-compute gamma_k coefficients for convolution
        if self.n_steps > 1:
            k = np.arange(2, self.n_steps + 1)
            if np.abs(self.alpha) > 1e-9:
                base = (k**(self.alpha + 1) - (k - 1)**(self.alpha + 1)) / (self.alpha + 1)
                # Ensure numerical stability for negative alpha
                b_k = np.sign(base) * np.abs(base)**(1 / self.alpha)
                self.gamma_k = (b_k * self.dt)**self.alpha
            else: # H=0.5 case (Standard Brownian Motion)
                self.gamma_k = np.zeros_like(k, dtype=float)
        else:
            self.gamma_k = np.array([])

    def generate_paths(self, N_paths, rho, eta, xi_func, S0, r_func, q_func, seed=None):
        """Generates asset price paths."""
        
        # Allow setting a seed for reproducibility (important for SLSQP)
        if seed is not None:
            np.random.seed(seed)

        if self.n_steps == 0:
            return np.full((N_paths, 1), S0)

        # Basic check for positive semi-definiteness (numerical stability)
        # If the matrix is near singular, regularization might be needed.
        if np.linalg.cond(self.cov_matrix) > 1/np.finfo(self.cov_matrix.dtype).eps:
             # Handle near-singular matrix by slightly adjusting correlation if needed
             corr = self.cov_matrix[0,1] / np.sqrt(self.cov_matrix[0,0] * self.cov_matrix[1,1])
             if abs(corr) > 1.0:
                clipped_corr = np.clip(corr, -1.0, 1.0)
                self.cov_matrix[0,1] = self.cov_matrix[1,0] = clipped_corr * np.sqrt(self.cov_matrix[0,0] * self.cov_matrix[1,1])

        # Generate correlated Brownian increments (W1, W2)
        try:
            dW_vectors = np.random.multivariate_normal([0, 0], self.cov_matrix, size=(N_paths, self.n_steps))
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"Warning: Covariance matrix issue during path generation (H={self.H}): {e}")
            return None

        W1_increments = dW_vectors[:, :, 0]
        W2_increments = dW_vectors[:, :, 1]
        
        # Generate orthogonal component (W_perp)
        dW_perp = np.random.randn(N_paths, self.n_steps) * np.sqrt(self.dt)

        # Simulate the Riemann-Liouville process Y using FFT convolution
        Y = np.zeros((N_paths, self.n_steps + 1))
        if np.any(self.gamma_k) and np.abs(self.alpha) > 1e-9:
            # FFT setup
            gamma_k_padded = np.zeros(self.n_steps)
            if len(self.gamma_k) > 0:
               gamma_k_padded[1:] = self.gamma_k
            fft_len = 2 * self.n_steps
            
            # Convolution via FFT
            gamma_k_fft = rfft(gamma_k_padded, fft_len)
            W1_fft = rfft(W1_increments, fft_len, axis=1)
            
            convolution_sum = irfft(gamma_k_fft * W1_fft, fft_len, axis=1)[:, :self.n_steps]
            Y[:, 1:] = np.sqrt(2 * self.H) * (W2_increments + convolution_sum)
        else: # H=0.5 case
            Y[:, 1:] = np.sqrt(2 * self.H) * W2_increments
        
        # Forward variance curve evaluation
        forward_variances = xi_func(self.time_grid)
        
        # Variance process V(t) (Wick exponential)
        exponent_term = eta * Y - 0.5 * eta**2 * self.time_grid**(2 * self.H)
        # Clip exponent to prevent numerical overflow/underflow
        exponent_term_clipped = np.clip(exponent_term, -30, 30) 
        V = forward_variances * np.exp(exponent_term_clipped)
        
        # Asset price process S(t) using Log-Euler scheme
        S = np.zeros((N_paths, self.n_steps + 1))
        S[:, 0] = S0
        # Ensure variance is strictly positive
        V_sliced = np.maximum(V[:, :-1], 1e-12)
        
        # Time-dependent rates
        time_grid_sliced = self.time_grid[:-1]
        r_t = r_func(time_grid_sliced)
        q_t = q_func(time_grid_sliced)
        
        # Construct correlated asset Brownian motion (dB)
        dB = rho * W1_increments + np.sqrt(1 - rho**2) * dW_perp
        
        # Accumulate log-returns
        log_increments = (r_t - q_t - 0.5 * V_sliced) * self.dt + np.sqrt(V_sliced) * dB
        log_S_paths = np.log(S0) + np.cumsum(log_increments, axis=1)
        
        S[:, 1:] = np.exp(log_S_paths)
        
        return S

# ==============================================================================
# CALIBRATION ENGINE (Enhanced with Top N Tracking and Dual Methods)
# ==============================================================================

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

def simulate_rbergomi_paths_mBm_batch(H_func, rho, eta, xi, S0, T, n_steps, n_paths, r_func=None, q_func=None):
    """
    Simulates n_paths simultaneously for the mBm rough Bergomi model.
    Returns terminal stock prices S_T of shape (n_paths,).
    """
    Y_paths, W1_inc = _simulate_hybrid_components_mBm_vectorised(
        H_func, n_steps, T, n_paths)
    
    dt        = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)

    # Time-varying Ito correction
    H_vals     = np.array([H_func(t) for t in time_grid])
    correction = 0.5 * eta**2 * time_grid ** (2 * H_vals)  # shape (n_steps+1,)

    # Variance process — shape (n_paths, n_steps+1)
    V = xi * np.exp(eta * Y_paths - correction[np.newaxis, :])

    # Stock price
    dW_perp = np.sqrt(dt) * np.random.randn(n_paths, n_steps)
    dB = rho * W1_inc + np.sqrt(1 - rho**2) * dW_perp

    V_t = np.maximum(V[:, :-1], 1e-12)
    
    if r_func is not None and q_func is not None:
        time_grid_mid = time_grid[:-1]
        r_t = np.array([r_func(t) for t in time_grid_mid])
        q_t = np.array([q_func(t) for t in time_grid_mid])
        drift = (r_t - q_t - 0.5 * V_t) * dt
        
    else:
        drift = -0.5 * V_t * dt   # fallback: r=q=0

    log_increments = drift + np.sqrt(V_t) * dB
    log_S_T        = np.log(S0) + np.sum(log_increments, axis=1)
    return np.exp(log_S_T)

class rBergomi_Calibrator:
    def __init__(self, calibration_df, S0, r_func, q_func, xi_func):
        self.S0 = S0
        self.market_df = calibration_df
        self.r_func, self.q_func, self.xi_func = r_func, q_func, xi_func
        
        self.market_data_dict = {
            (row['strike'], row['time_to_maturity']): row['implied_vol']
            for _, row in self.market_df.iterrows()
        }
        self.options_grid = list(self.market_data_dict.keys())
        self.num_options = len(self.options_grid)
        print(f"\nCalibrator initialized with {self.num_options} options.")
        
        # Default settings
        self.n_paths = 50000
        self.n_steps_per_year = 100
        self.use_static_seed = False # Controls seeding in path generation

        # NEW: Tracking mechanism initialization
        self.top_results = []
        self.iteration_count = 0

    def _reset_tracking(self):
        # Reset tracking before a new calibration run
        self.top_results = []
        self.iteration_count = 0

    def _update_top_results(self, mse, params):
        # Update the list of top 10 results
        # We append the result and then sort/truncate. Since the cost function 
        # is expensive, the overhead of sorting this small list is negligible.
        self.top_results.append((mse, params))
        self.top_results.sort(key=lambda x: x[0]) # Sort by MSE (ascending)
        if len(self.top_results) > 10:
            self.top_results = self.top_results[:10] # Keep only top 10

    def _price_options_grid(self, H, rho, eta):
            """
            Prices the calibration grid options using Monte Carlo.
            """
            if not self.options_grid:
                return None
                
            max_maturity = max(maturity for _, maturity in self.options_grid)
            n_steps = int(self.n_steps_per_year * max_maturity)
            if n_steps == 0:
                return None
    
            model = rBergomi(n_steps=n_steps, T=max_maturity, H=H)
            
            # Determine seed: Static for SLSQP, Dynamic (None) for DE
            seed = 42 if self.use_static_seed else None
    
            # Generate paths (Note: Batching is omitted here for brevity, assuming memory is sufficient for n_paths)
            S_paths = model.generate_paths(self.n_paths, rho, eta, self.xi_func, self.S0, self.r_func, self.q_func, seed=seed)
    
            if S_paths is None:
                return None
    
            # Calculate prices and IVs for the grid
            model_implied_vols = {}
            for option_key in self.options_grid:
                strike, maturity = option_key
                r_option = self.r_func(maturity)
                q_option = self.q_func(maturity)
                
                # Find the correct time index for maturity
                time_idx = int(round(n_steps * (maturity / max_maturity)))
                if time_idx >= S_paths.shape[1]:
                    time_idx = S_paths.shape[1] - 1
                
                if time_idx < 0: continue
    
                S_T = S_paths[:, time_idx]
                
                # Determine option type based on forward
                forward_price = self.S0 * np.exp((r_option - q_option) * maturity)
                option_type = 'c' if strike >= forward_price else 'p'
                
                # Calculate payoff
                if option_type == 'c':
                    payoffs = np.maximum(S_T - strike, 0)
                else:
                    payoffs = np.maximum(strike - S_T, 0)
                
                # Discounted price
                discounted_price = np.mean(payoffs) * np.exp(-r_option * maturity)
                
                # Back out IV
                iv = find_implied_vol(discounted_price, self.S0, strike, maturity, r_option, q_option, flag=option_type)
                model_implied_vols[option_key] = iv
                
            return model_implied_vols


    def _price_options_grid_mBm(self, H_func, rho, eta):
        """
        Prices the calibration grid options using the simulation method for multifractional Brownian motion.
        """
        if not self.options_grid:
            return None
        
        if self.use_static_seed:                        
            np.random.seed(42)  
        
        max_maturity = max(maturity for _, maturity in self.options_grid)
        n_steps = int(self.n_steps_per_year * max_maturity)
        if n_steps == 0:
            return None

        # Use vectorised multifractional Brownian motion batch simulator
        model_implied_vols = {}
        for option_key in self.options_grid:
            
            strike, maturity = option_key
            r_option = self.r_func(maturity)
            q_option = self.q_func(maturity)
            n_steps_opt = int(self.n_steps_per_year * maturity)
            
            if n_steps_opt == 0:
                continue
    
            # Batch simulation for this maturity
            S_T = simulate_rbergomi_paths_mBm_batch(H_func = H_func, rho = rho, eta = eta, xi = self.xi_func(maturity), S0 = self.S0,
                T = maturity, n_steps = n_steps_opt, n_paths = self.n_paths, r_func  = self.r_func, q_func  = self.q_func)
    
            forward_price = self.S0 * np.exp((r_option - q_option) * maturity)
            
            option_type = 'c' if strike >= forward_price else 'p'
            
            payoffs = (np.maximum(S_T - strike, 0) if option_type == 'c' else np.maximum(strike - S_T, 0))
            
            price = np.mean(payoffs) * np.exp(-r_option * maturity)
            iv = find_implied_vol(price, self.S0, strike, maturity, r_option, q_option, flag=option_type)
            model_implied_vols[option_key] = iv
    
        return model_implied_vols

    def cost_function_fBm(self, params):
            """
            The objective function (MSE). Tracks top 10 results during the run.
            """
            self.iteration_count += 1
            H, rho, eta = params
            
            # Safety constraints (though optimizers usually handle bounds)
            if H <= 0.001 or H >= 0.999 or eta <= 0.01 or rho <= -0.999 or rho >= 0.999:
                 return 1e10
    
            # Print progress
            print(f"Iter: {self.iteration_count:04d} | Testing: H={H:.4f}, rho={rho:.4f}, eta={eta:.4f}...", end="")
    
            model_vols = self._price_options_grid(H, rho, eta)
            if model_vols is None:
                print(" -> MSE: Error (Pricing Failed)")
                return 1e10
    
            total_squared_error = 0.0
            valid_options = 0
            for option, market_iv in self.market_data_dict.items():
                model_iv = model_vols.get(option, np.nan)
                
                if np.isnan(model_iv) or np.isnan(market_iv):
                    # Apply a penalty for failures (e.g., if MC price violates arbitrage bounds)
                    total_squared_error += 0.5**2 # Penalty equivalent to 50 vol points error
                else:
                    total_squared_error += (model_iv - market_iv)**2
                    valid_options += 1
            
            if valid_options == 0:
                 print(" -> MSE: Error (No valid options priced)")
                 return 1e10
    
            # Calculate Mean Squared Error (MSE)
            mse = total_squared_error / self.num_options
            
            # NEW: Update top results tracking
            self._update_top_results(mse, {'H': H, 'rho': rho, 'eta': eta})
                    
            print(f" -> MSE: {mse:.8f}")
            return mse

    def cost_function_mBm(self, H_func_factory, param_names):
        """
        Returns a cost function closure for an arbitrary H_func specification.

        Parameters:
        
        H_func_factory : callable(shape_params, rho, eta) -> H_func; Given the shape parameters, returns the H_func callable.
            e.g. for linear: lambda params, r, e: (lambda t: clip(params[0]+params[1]*t))
        param_names : list of str; Names of the shape parameters for display/tracking.
            e.g. ['beta0', 'beta1'] for linear

        """
        
        def cost_fn(params):
            self.iteration_count += 1
            shape_params = params[:-2]
            rho, eta     = params[-2], params[-1]
    
            # Basic validity
            if eta <= 0.01 or rho <= -0.999 or rho >= 0.999:
                return 1e10
    
            # Construct H_func from current shape parameters
            H_func = H_func_factory(shape_params)
    
            # Check H_func stays in (0, 0.5) on a coarse grid
            t_check = np.linspace(0, 5, 20)
            H_check = np.array([H_func(t) for t in t_check])
            if np.any(H_check <= 0) or np.any(H_check >= 0.5):
                return 1e10
    
            # Progress display
            shape_str = ', '.join(
                f"{n}={v:.4f}" for n, v in zip(param_names, shape_params)
            )
            print(f"Iter {self.iteration_count:04d} | {shape_str}, "
                  f"rho={rho:.4f}, eta={eta:.4f}...", end="")
    
            model_vols = self._price_options_grid_mBm(H_func, rho, eta)
            if model_vols is None:
                print(" -> Error")
                return 1e10
    
            total_sq_err = 0.0
            valid = 0
            for opt, market_iv in self.market_data_dict.items():
                model_iv = model_vols.get(opt, np.nan)
                if np.isnan(model_iv) or np.isnan(market_iv):
                    total_sq_err += 0.5**2
                else:
                    total_sq_err += (model_iv - market_iv)**2
                    valid += 1
    
            if valid == 0:
                return 1e10
    
            mse = total_sq_err / self.num_options
    
            # Track result with named parameters
            param_dict = dict(zip(param_names, shape_params))
            param_dict.update({'rho': rho, 'eta': eta})
            self._update_top_results(mse, param_dict)
    
            print(f" -> MSE: {mse:.8f}")
            return mse
    
        return cost_fn
        
    def calibrate_slsqp(self, x0, bounds=None):
        """
        Starts the calibration process using the SLSQP (Local Optimizer).
        """
        self._reset_tracking()
        # SLSQP needs a static seed for stable finite difference gradient estimation
        self.use_static_seed = True

        if bounds is None:
            bounds = [(0.01, 0.49), (-0.99, -0.01), (0.5, 5.5)]
            
        print("\n--- Starting Local Calibration (SLSQP) ---")
        print(f"Settings: N_Paths={self.n_paths}, Steps/Year={self.n_steps_per_year}, Static Seed=True")
        start_time = time.time()
        
        # Call SLSQP. 'eps' is increased to help navigate the noisy MC surface.
        result = minimize(
            self.cost_function,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            options={'disp': False, 'ftol': 1e-6, 'maxiter': 100, 'eps': 1e-4}
        )
        
        print(f"\n--- SLSQP Calibration finished in {(time.time() - start_time)/60:.2f} minutes ---")
        if result.success:
            print("Optimal parameters found:")
            print(f"H = {result.x[0]:.4f}, rho = {result.x[1]:.4f}, eta = {result.x[2]:.4f}")
            print(f"Final Error (MSE) = {result.fun:.8f}")
        else:
            print("Calibration failed:", result.message)
            
        self.use_static_seed = False # Reset seed preference
        return result

    def calibrate_global(self, bounds=None, seed=42, maxiter=50, popsize=15):
        """
        Starts the calibration process using Differential Evolution (Global Optimizer).
        """
        self._reset_tracking()
        # DE benefits from dynamic seeding (stochastic exploration)
        self.use_static_seed = False

        if bounds is None:
            bounds = [(0.01, 0.49), (-0.99, -0.10), (0.5, 5.5)]
            
        print("\n--- Starting Global Calibration (Differential Evolution) ---")
        print(f"Settings: N_Paths={self.n_paths}, Steps/Year={self.n_steps_per_year}, MaxIter={maxiter}, PopSize={popsize}")
        start_time = time.time()
        
        # Call Differential Evolution
        result = differential_evolution(
            self.cost_function,
            bounds=bounds,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            mutation=(0.5, 1.0), # Controls the 'jumps'
            recombination=0.7,
            seed=seed,
            disp=False, # We handle printing in cost_function
            polish=True # Uses L-BFGS-B at the end to refine the best result
        )
        
        print(f"\n--- Differential Evolution finished in {(time.time() - start_time)/60:.2f} minutes ---")
        
        # Display Final Result
        if result.success:
            print("\n--- Best Result Found (Converged) ---")
        else:
            # This often happens when maxiter is reached, which is expected
            print("\n--- Best Result Found (Max Iterations Reached) ---")
            print("Calibration message:", result.message)

        print(f"H = {result.x[0]:.4f}, rho = {result.x[1]:.4f}, eta = {result.x[2]:.4f}")
        print(f"Final Error (MSE) = {result.fun:.8f}")

        # NEW: Display the Top 10 results found during the search
        self.display_top_results()
            
        return result
    
    def display_top_results(self):
        """Displays top 10 results tracked during the optimization for any parameter set."""
        if not self.top_results:
            print("No results tracked yet.")
            return
    
        print("\n--- Top 10 Parameter Sets Found ---")
    
        # Infer column names from first result
        first_params = self.top_results[0][1]
        col_names    = list(first_params.keys())
    
        # Header
        header = f"{'Rank':<6}|{'MSE':<12}|" + "|".join(f"{n:<10}" for n in col_names)
        
        print(header)
        print("-" * len(header))
    
        for rank, (mse, params) in enumerate(self.top_results, 1):
            row = f"{rank:<6}|{mse:<12.8f}|" + "|".join(f"{params[n]:<10.4f}" for n in col_names)
            print(row)
            
        print("-" * len(header))

# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_single_maturity_smile(spot_price, full_surface_df, r_func, q_func, xi_func, calib_params, target_days):
    """
    Compares and plots the market and model implied volatilities for a specific maturity.
    """
    print(f"\n--- Starting Plot Generation for {target_days}-Day Maturity ---")
    
    # 1: Filter Maturity
    target_T_years = target_days / 365.0
    unique_maturities = full_surface_df['time_to_maturity'].unique()
    
    if len(unique_maturities) == 0:
        return

    closest_maturity = unique_maturities[np.argmin(np.abs(unique_maturities - target_T_years))]
    smile_df = full_surface_df[full_surface_df['time_to_maturity'] == closest_maturity].copy()
    
    if smile_df.empty:
        return

    # 2: Calculate Moneyness
    T_val = smile_df['time_to_maturity'].iloc[0]
    r_val = r_func(T_val)
    q_val = q_func(T_val)
    forward_price = spot_price * np.exp((r_val - q_val) * T_val)
    
    smile_df['forward_moneyness'] = smile_df['strike'] / forward_price
    smile_df = smile_df.sort_values('forward_moneyness').reset_index(drop=True)
    smile_df.dropna(subset=['implied_vol'], inplace=True)
    
    print(f"Closest maturity found: {closest_maturity*365:.1f} days (T={T_val:.4f}).")

    # 3: Model Simulation (High precision for plotting)
    H, rho, eta = calib_params['H'], calib_params['rho'], calib_params['eta']
    
    # Increased paths for visualization to handle deep OTM options better
    n_paths =  250000 
    n_steps = max(50, int(100 * T_val))
    
    print(f"Starting high-precision simulation ({n_paths} paths, {n_steps} steps)...")
    model = rBergomi(n_steps=n_steps, T=T_val, H=H)
    # Use a fixed seed for reproducible plots
    S_paths = model.generate_paths(n_paths, rho, eta, xi_func, spot_price, r_func, q_func, seed=123)
    
    if S_paths is None:
        print("Plotting failed due to simulation error.")
        return
        
    S_T = S_paths[:, -1]
    print("Simulation complete.")

    # 4: Calculate Model IVs
    model_ivs = []
    for _, row in smile_df.iterrows():
        K = row['strike']
        r_row = r_func(T_val)
        q_row = q_func(T_val)
        fwd_mny = row['forward_moneyness']

        option_type = 'c' if fwd_mny >= 1.0 else 'p'
        payoffs = np.maximum(S_T - K, 0) if option_type == 'c' else np.maximum(K - S_T, 0)
        discounted_price = np.mean(payoffs) * np.exp(-r_row * T_val)
        # IV calculation handles potential MC price=0 (returns NaN)
        iv = find_implied_vol(discounted_price, spot_price, K, T_val, r_row, q_row, flag=option_type)
        model_ivs.append(iv)
        
    smile_df['model_iv'] = model_ivs
    
    # 5: Generate Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        smile_df['forward_moneyness'],
        smile_df['implied_vol'],
        'o-',
        color='royalblue',
        markersize=4,
        label=r'SPX Implied Volatility'
    )
    
    # Filter out NaN model IVs (where MC price was 0 for Deep OTM options)
    valid_model_data = smile_df.dropna(subset=['model_iv'])
    if not valid_model_data.empty:
        ax.plot(
            valid_model_data['forward_moneyness'],
            valid_model_data['model_iv'],
            'x--',
            color='darkred',
            label=rf'Rough Bergomi Implied Volatility'
        )
    else:
        print("Warning: No valid model volatilities to plot.")

    ax.set_title(
        rf'SPX Implied Volatility and Rough Bergomi Smile (Maturity = {T_val * 365:.0f} days)'
        '\n'
        rf'Calibrated Parameters: '
        rf'$H = {H:.4f},\ \rho = {rho:.4f},\ \eta = {eta:.4f}$'
    )
    ax.set_xlabel(r'Moneyness $(K/F)$')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel(r'Implied Volatility $\sigma_{\mathrm{BS}}(K,T)$')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    filename = f"Smile_T{target_days}_H{H:.4f}_Rho{rho:.4f}_Eta{eta:.4f}.png"
    plt.savefig(filename, dpi=600)
    print(f"Plot saved as '{filename}'.")
    plt.show()

def plot_single_maturity_smile_mBm(spot_price, full_surface_df, r_func, q_func, xi_func,
                                    calib_params, H_func_factory, target_days):
    """
    Compares and plots the market and model implied volatilities for a specific maturit for multifractional H(t).
    
    Parameters:
    -----------
    calib_params   : dict — must contain 'rho', 'eta', and all shape params
    H_func_factory : callable(shape_params) -> H_func
                     same factory used during calibration
    target_days    : int — target maturity in calendar days
    """
    print(f"\n--- Plot: {target_days}-Day Maturity (mBm) ---")

    # --- Step 1: Find closest maturity ---
    target_T = target_days / 365.0
    unique_T = full_surface_df['time_to_maturity'].unique()
    closest_T = unique_T[np.argmin(np.abs(unique_T - target_T))]
    smile_df  = full_surface_df[full_surface_df['time_to_maturity'] == closest_T].copy()

    if smile_df.empty:
        print("No data for this maturity.")
        return

    # --- Step 2: Reconstruct H_func from calibrated shape params ---
    rho = calib_params['rho']
    eta = calib_params['eta']

    # All keys except rho and eta are shape params
    shape_param_names = [k for k in calib_params if k not in ('rho', 'eta')]
    shape_params      = [calib_params[k] for k in shape_param_names]
    H_func            = H_func_factory(shape_params)

    # --- Step 3: Compute moneyness ---
    T_val   = smile_df['time_to_maturity'].iloc[0]
    r_val   = r_func(T_val)
    q_val   = q_func(T_val)
    forward = spot_price * np.exp((r_val - q_val) * T_val)

    smile_df['forward_moneyness'] = smile_df['strike'] / forward
    smile_df = smile_df.sort_values('forward_moneyness').reset_index(drop=True)
    smile_df.dropna(subset=['implied_vol'], inplace=True)

    print(f"Closest maturity: {closest_T*365:.1f} days (T={T_val:.4f})")

    # --- Step 4: Simulate ---
    n_paths = 250000
    n_steps = max(50, int(100 * T_val))
    print(f"Simulating {n_paths} paths, {n_steps} steps...")

    S_T = simulate_rbergomi_paths_mBm_batch(H_func = H_func, rho = rho, eta = eta, xi = xi_func(T_val),
        S0 = spot_price, T = T_val, n_steps = n_steps, n_paths = n_paths, r_func = r_func, q_func = q_func)

    # --- Step 5: Back out model IVs ---
    model_ivs = []
    for _, row in smile_df.iterrows():
        K   = row['strike']
        fmn = row['forward_moneyness']
        opt = 'c' if fmn >= 1.0 else 'p'
        payoffs = (np.maximum(S_T - K, 0) if opt == 'c' else np.maximum(K - S_T, 0))
        price = np.mean(payoffs) * np.exp(-r_val * T_val)
        iv    = find_implied_vol(price, spot_price, K, T_val, r_val, q_val, flag=opt)
        model_ivs.append(iv)

    smile_df['model_iv'] = model_ivs

    # --- Step 6: Build title string from shape params ---
    shape_str = ', '.join(f"{n}={v:.4f}" for n, v in zip(shape_param_names, shape_params))
    param_str = shape_str + rf', $\rho={rho:.4f}$, $\eta={eta:.4f}$'

    # --- Step 7: Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(smile_df['forward_moneyness'], smile_df['implied_vol'], 'o-', color='royalblue', markersize=4, label='Market IV')

    valid = smile_df.dropna(subset=['model_iv'])
    if not valid.empty:
        ax.plot(valid['forward_moneyness'], valid['model_iv'], 'x--', color='darkred', label='mBm Rough Bergomi IV')

    ax.set_title(rf'SPX Implied Volatility — mBm Rough Bergomi' '\n' rf'Maturity = {T_val*365:.0f} days | {param_str}')
    ax.set_xlabel(r'Moneyness $(K/F)$')
    ax.set_ylabel(r'Implied Volatility $\sigma_{\mathrm{BS}}(K,T)$')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    filename = (f"mBm_smile_T{target_days}" + "_".join(f"{n}{v:.4f}" for n, v in zip(shape_param_names, shape_params)) + f"_rho{rho:.4f}_eta{eta:.4f}.png")
    plt.savefig(filename, dpi=600)
    print(f"Saved: {filename}")
    plt.show()
    plt.close(fig)

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

def generate_smile(H, rho, eta, xi, S0, T, n_steps, n_paths, moneyness, r_func=None, q_func=None):
    """Generates a single implied volatility smile for a given set of rBergomi parameters."""
    print(f"Starting simulation for H = {H:.2f}...")
    start_time = time.time()
    
    # Wrap scalar xi as a callable for generate_paths
    xi_func_wrap = lambda t: xi

    # Use provided rate functions or default to zero
    r_func_use = r_func if r_func is not None else lambda t: 0.0
    q_func_use = q_func if q_func is not None else lambda t: 0.0

    # Use r and q at maturity T for IV inversion
    r_val = float(r_func_use(T)) 
    q_val = float(q_func_use(T))
    
    model = rBergomi(n_steps=n_steps, T=T, H=H)
    S_paths = model.generate_paths(n_paths, rho, eta, xi_func_wrap, S0, r_func_use, q_func_use)
    S_T = S_paths[:, -1] # Get the terminal stock prices
    
    implied_vols = []
    strike_range = moneyness * S0
    
    for K in strike_range:
        # Use Out-of-the-Money (OTM) options to reduce estimator noise.
        option_type = 'c' if K >= S0 else 'p'
        
        # Calculate the option price via Monte Carlo average of payoffs
        model_price = np.mean(np.maximum(S_T - K, 0) if option_type == 'c' else np.maximum(K - S_T, 0))
        
        # Back out the implied volatility from the model price
        iv = find_implied_vol(model_price, S0, K, T, r=r_val, q=q_val, flag=option_type)
        implied_vols.append(iv)
        
    print(f"Finished H = {H:.2f} in {(time.time() - start_time):.2f} seconds.")
    return implied_vols

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
    
def plot_model_vs_market_multi_maturity(
        spot_price, final_surface_df, r_func, q_func, xi_func,
        fbm_params, mbm_params_list, target_days_list):
    """
    For each maturity in target_days_list, plots:
    - Market smile
    - fBm rough Bergomi smile
    - Each mBm specification smile
    Side by side for direct comparison.
    """
    n_maturities = len(target_days_list)
    fig, axes = plt.subplots(1, n_maturities,
                              figsize=(7*n_maturities, 6),
                              sharey=True)
    if n_maturities == 1:
        axes = [axes]

    moneyness_range = np.linspace(0.8, 1.2, 25)

    for ax, target_days in zip(axes, target_days_list):
        T = target_days / 365.0
        r_val = r_func(T)
        q_val = q_func(T)

        # Market smile
        tol = 5 / 365.0
        market_slice = final_surface_df[
            np.abs(final_surface_df['time_to_maturity'] - T) < tol
        ].copy()
        market_slice['mny'] = market_slice['strike'] / spot_price
        market_slice = market_slice.sort_values('mny')
        ax.plot(market_slice['mny'], market_slice['implied_vol'],
                'ko-', markersize=3, lw=1.2, label='Market', zorder=5)

        # fBm model smile
        H_fbm = fbm_params['H']
        smile_fbm = generate_smile(
            H=H_fbm, rho=fbm_params['rho'],
            eta=fbm_params['eta'], xi=xi_func(T),
            S0=spot_price, T=T,
            n_steps=max(20, int(100*T)),
            n_paths=250000, moneyness=moneyness_range,
            r_func=r_func,     
            q_func=q_func      
        )
        ax.plot(moneyness_range, smile_fbm, 'b--',
                lw=1.5, label=rf'fBm $H={H_fbm:.2f}$')

        # mBm model smiles
        colors = ['red', 'green', 'orange', 'purple']
        for (spec_label, spec_func, mbm_p), col in zip(mbm_params_list, colors):
            smile_mbm = generate_smile_mBm(
                H_func=spec_func, rho=mbm_p['rho'], eta=mbm_p['eta'],
                xi=xi_func(T), S0=spot_price, T=T, n_steps=max(20, int(100*T)), n_paths=250000,
                moneyness=moneyness_range, r_func = r_func, q_func = q_func)
            
            ax.plot(moneyness_range, smile_mbm,
                    color=col, lw=1.5, label=spec_label)

        ax.set_title(f'T = {target_days} days')
        ax.set_xlabel(r'Moneyness $(K/F)$')
        ax.grid(True, alpha=0.4)
        if ax == axes[0]:
            ax.set_ylabel(r'Implied Volatility $\sigma_{\mathrm{BS}}$')
        ax.legend(fontsize=8)

    fig.suptitle('Market vs fBm vs mBm — Implied Volatility Smiles\n'
                 '28 July 2026 SPX Options', fontsize=14)
    plt.tight_layout()
    plt.savefig('market_vs_model_smiles.png', dpi=600)
    plt.show()
    
def plot_mse_comparison(results_dict):
    """
    Bar chart of calibration MSE for each H(t) specification.
    results_dict: {'fBm H=0.08': 0.00123, 'Linear H(t)': 0.00089, ...}
    """
    labels = list(results_dict.keys())
    mses   = [v * 10000 for v in results_dict.values()]  # convert to bps^2

    colors = ['steelblue'] + ['coral'] * (len(labels) - 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, mses, color=colors, edgecolor='black', lw=0.8)
    ax.bar_label(bars, fmt='{:.2f}', padding=3, fontsize=11)

    ax.set_ylabel(r'MSE ($\times 10^{-4}$ vol$^2$)')
    ax.set_title('Calibration MSE by Model Specification\n'
                 '28 July 2026 SPX Options')
    ax.grid(True, axis='y', alpha=0.4)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('mse_comparison.png', dpi=600)
    plt.show()

# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_two_stage(calibrator, cost_fn, bounds, spec_name, de_paths=50000, slsqp_paths=200000, de_maxiter=50, de_popsize=15):
    """
    Two-stage calibration:
    Stage 1 — DE with moderate paths for global search
    Stage 2 — SLSQP refinement from DE best point with high paths
    Returns the better of the two results.
    """
    print(f"\n{'='*60}")
    print(f"  Calibrating: {spec_name}")
    print(f"{'='*60}")

    # ── Stage 1: Differential Evolution ──────────────────────────
    print(f"\n  Stage 1: Differential Evolution "
          f"(n_paths={de_paths}, maxiter={de_maxiter})...")

    calibrator.n_paths        = de_paths
    calibrator.use_static_seed = False   # dynamic seed for DE

    t0 = time.time()
    result_de = differential_evolution(
        cost_fn,
        bounds      = bounds,
        strategy    = 'best1bin',
        maxiter     = de_maxiter,
        popsize     = de_popsize,
        tol         = 1e-5,
        mutation    = (0.5, 1.0),
        recombination = 0.7,
        seed        = 42,
        polish      = False,    # manual SLSQP polish below
        disp        = False)
    
    t_de = (time.time() - t0) / 60

    print(f"\n  DE finished in {t_de:.2f} min")
    print(f"  DE best:  params={result_de.x}, MSE={result_de.fun:.8f}")

    # ── Stage 2: SLSQP refinement ─────────────────────────────────
    print(f"\n  Stage 2: SLSQP refinement "
          f"(n_paths={slsqp_paths}, x0=DE solution)...")

    calibrator.n_paths         = slsqp_paths
    calibrator.use_static_seed = True    # fixed seed for stable gradients

    t0 = time.time()
    result_slsqp = minimize(
        cost_fn,
        x0      = result_de.x,          # warm start from DE
        method  = 'SLSQP',
        bounds  = bounds,
        options = {
            'ftol'   : 1e-7,
            'maxiter': 50,
            'eps'    : 5e-4,             # larger eps reduces gradient noise
            'disp'   : False})
    
    t_slsqp = (time.time() - t0) / 60

    print(f"\n  SLSQP finished in {t_slsqp:.2f} min")
    print(f"  SLSQP: params={result_slsqp.x}, MSE={result_slsqp.fun:.8f}")

    calibrator.use_static_seed = False   # reset

    # ── Pick better result ────────────────────────────────────────
    if result_slsqp.fun < result_de.fun:
        print(f"\n  -> SLSQP improved DE result "
              f"({result_de.fun:.8f} -> {result_slsqp.fun:.8f})")
        best = result_slsqp
        best_stage = "SLSQP"
    else:
        print(f"\n  -> DE result retained "
              f"(SLSQP did not improve: "
              f"{result_slsqp.fun:.8f} vs {result_de.fun:.8f})")
        best = result_de
        best_stage = "DE"

    print(f"\n  Best ({best_stage}): MSE={best.fun:.8f}")
    return best, result_de, result_slsqp

# =============================================================================
# MAIN EXECUTION BLOCK FOR CALIBRATION AND PLOTTING
# =============================================================================
if __name__ == '__main__':

    # --- Load data ---
    spot_price, quote_date, final_surface_df, r_func, q_func = \
        prepare_full_surface_and_rates('UnderlyingOptionsEODCalcs_2026-07-28.csv',
            constant_dividend_yield=0.0115)
    if spot_price is None:
        exit()
        
    atm_curve_df = extract_atm_volatility_curve(final_surface_df, spot_price)
    xi_func = create_xi_from_market_atm(atm_curve_df)
    calib_df = create_calibration_grid(final_surface_df, spot_price, num_options=50)
    calibrator = rBergomi_Calibrator(calib_df, spot_price, r_func, q_func, xi_func)

    calibrator.n_steps_per_year = 100

    ###########################################################################
    # 1. fBm baseline calibration
    ###########################################################################
    
    bounds_fbm = [(0.01, 0.49), (-0.99, -0.01), (0.50, 5.50)]
    calibrator._reset_tracking()
    
    best_fbm, de_fbm, slsqp_fbm = calibrate_two_stage(
        calibrator  = calibrator,
        cost_fn     = calibrator.cost_function_fBm,
        bounds      = bounds_fbm,
        spec_name   = "fBm constant H",
        de_paths    = 50000, 
        slsqp_paths = 200000 
    )
    
    fbm_params = {
        'H'  : best_fbm.x[0],
        'rho': best_fbm.x[1],
        'eta': best_fbm.x[2],
        'MSE': best_fbm.fun
    }
    
    print(f"\nfBm final: H={fbm_params['H']:.4f}, "
          f"rho={fbm_params['rho']:.4f}, "
          f"eta={fbm_params['eta']:.4f}, "
          f"MSE={fbm_params['MSE']:.8f}")
    
    ###########################################################################
    # 2. mBm factories
    ###########################################################################
    
    def linear_factory(p):
            b0, b1 = p[0], p[1]
            return lambda t: np.clip(b0 + b1*t, 1e-4, 0.4999)
    
    def logistic_factory(p):
            H_min, H_max, gamma = p[0], p[1], p[2]
            return lambda t: np.clip(
                H_min + (H_max-H_min)/(1+np.exp(-gamma*(t-0.5))), 1e-4, 0.4999)
    
    def quadratic_factory(p):
            b0, b1, b2 = p[0], p[1], p[2]
            return lambda t: np.clip(b0 + b1*t + b2*t**2, 1e-4, 0.4999)
    
    def sinusoidal_factory(p):
            a0, a1, period = p[0], p[1], p[2]
            return lambda t: np.clip(
                a0 + a1*np.sin(2*np.pi*t/period), 1e-4, 0.4999)
    
    mBm_specs_calibration = [
        {
            "name"   : "Linear $H(t)$",
            "factory": linear_factory,
            "p_names": ['beta0', 'beta1'],
            "bounds" : [(0.01,0.40),(-0.20,0.30),(-0.99,-0.01),(0.50,5.50)],
            "color"  : "red"
        },
        {
            "name"   : "Logistic $H(t)$",
            "factory": logistic_factory,
            "p_names": ['H_min','H_max','gamma'],
            "bounds" : [(0.01,0.20),(0.10,0.49),(1.0,20.0),
                        (-0.99,-0.01),(0.50,5.50)],
            "color"  : "green"
        },
        {
            "name"   : "Quadratic $H(t)$",
            "factory": quadratic_factory,
            "p_names": ['b0','b1','b2'],
            "bounds" : [(0.01,0.35),(-0.20,0.20),(-0.10,0.10),
                        (-0.99,-0.01),(0.50,5.50)],
            "color"  : "orange"
        },
        {
            "name"   : "Sinusoidal $H(t)$",
            "factory": sinusoidal_factory,
            "p_names": ['alpha0','alpha1','period'],
            "bounds" : [(0.05,0.40),(0.01,0.20),(0.5,5.0),
                        (-0.99,-0.01),(0.50,5.50)],
            "color"  : "purple"
        },
    ]
    
    ###########################################################################
    # 3. mBm calibrations — build mbm_results dict
    ###########################################################################
    
    mbm_results = {}
    
    for spec in mBm_specs_calibration:
        print(f"\nCalibrating: {spec['name']}...")
        calibrator._reset_tracking()
    
        cost_fn = calibrator.cost_function_mBm(
            spec['factory'], param_names=spec['p_names'])
        
        best, de_res, slsqp_res = calibrate_two_stage(
            calibrator  = calibrator,
            cost_fn     = cost_fn,
            bounds      = spec['bounds'],
            spec_name   = spec['name'],
            de_paths    = 50000, 
            slsqp_paths = 200000) 
    
        shape_params = best.x[:-2]
        rho_cal      = best.x[-2]
        eta_cal      = best.x[-1]
    
        mbm_results[spec['name']] = {
            'params' : {
                **dict(zip(spec['p_names'], shape_params)),
                'rho': rho_cal,
                'eta': eta_cal
            },
            'H_func' : spec['factory'](shape_params),
            'MSE'    : best.fun,
            'MSE_DE' : de_res.fun,        
            'MSE_SLSQP': slsqp_res.fun,
            'color'  : spec['color'],
            'label'  : spec['name'],
            'factory': spec['factory'],
            'p_names': spec['p_names'],
        }
    
        p_str = ', '.join(
            f"{n}={v:.4f}"
            for n, v in zip(spec['p_names'], shape_params))
        
        print(f"\n{spec['name']} final: {p_str}, "
              f"rho={rho_cal:.4f}, eta={eta_cal:.4f}, "
              f"MSE={best.fun:.8f}")
        
        calibrator.display_top_results()
        
    ###########################################################################
    # 4. Plot multi-maturity comparison
    ###########################################################################   
    
    mbm_params_list = [
        (res['label'], res['H_func'],
            {'rho': res['params']['rho'], 'eta': res['params']['eta']})
        for res in mbm_results.values()]
    
    plot_model_vs_market_multi_maturity(
        spot_price       = spot_price,
        final_surface_df = final_surface_df,
        r_func           = r_func,
        q_func           = q_func,
        xi_func          = xi_func,
        fbm_params       = fbm_params,
        mbm_params_list  = mbm_params_list,
        target_days_list = [21, 60, 180, 360]
    )
    
    ###########################################################################
    # 5. Individual smile plots for each spec and maturity
    ###########################################################################
    for days in [21, 60, 180, 360]:
        plot_single_maturity_smile(
            spot_price      = spot_price,
            full_surface_df = final_surface_df,
            r_func          = r_func,
            q_func          = q_func,
            xi_func         = xi_func,
            calib_params    = fbm_params,
            target_days     = days
        )
    
    for name, res in mbm_results.items():
        shape_vals = [
            res['params'][n] for n in res['p_names']
        ]
        for days in [21, 60, 180, 360]:
            plot_single_maturity_smile_mBm(
                spot_price      = spot_price,
                full_surface_df = final_surface_df,
                r_func          = r_func,
                q_func          = q_func,
                xi_func         = xi_func,
                calib_params    = res['params'],
                H_func_factory  = res['factory'],
                target_days     = days
            )
    
    ###########################################################################
    # 6. Comparison Table
    ###########################################################################
    
    print("\n--- DE vs SLSQP MSE Comparison ---")
    print(f"{'Spec':<25}|{'DE MSE':<14}|{'SLSQP MSE':<14}|{'Winner':<8}")
    print("-" * 65)
    for name, res in mbm_results.items():
        winner = ("SLSQP" if res['MSE_SLSQP'] < res['MSE_DE']
                  else "DE")
        print(f"{name:<25}|{res['MSE_DE']:<14.8f}"
              f"|{res['MSE_SLSQP']:<14.8f}|{winner:<8}")
    







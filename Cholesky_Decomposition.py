import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

###############################################################################
# Different Hurst functions
###############################################################################

H_SPECS = [
    {
        "name": "Sinusoidal",
        "label": "$H(t) = 0.5 + 0.45\\sin(3\\pi t)$",
        "func": lambda t: np.clip(0.5 + 0.45 * np.sin(3 * np.pi * t), 1e-4, 0.9999)
    },
    {
        "name": "Constant (Rough Path)",
        "label": "$H(t) = 0.1$",
        "func": lambda t: np.clip(0.1, 1e-4, 0.9999)
    },
    {
        "name": "Constant (Smooth Path)",
        "label": "$H(t) = 0.75$",
        "func": lambda t: np.clip(0.75, 1e-4, 0.9999)
    },
    {
        "name": "Linear",
        "label": "$H(t) = 0.18 + 0.5t$",
        "func": lambda t: np.clip(0.18 + 0.5 * t, 1e-4, 0.9999)
    },
    {
        "name": "Quadratic",
        "label": "$H(t) = 0.55 - 1.95t + 2.15t^2$",
        "func": lambda t: np.clip(0.55 - 1.95*t + 2.15*t**2, 1e-4, 0.9999)
    },
    {
        "name": "Cubic",
        "label": "$H(t) = 0.115917 - 0.652677t + 7.006683t^2 - 6.398797t^3$",
        "func": lambda t: np.clip(0.115917 - 0.652677 * t + 7.006683 * t**2 - 6.398797 * t**3, 1e-4, 0.9999)
    },
    {
        "name": "Logistic",
        "label": "$H(t) = 0.05 + 0.80/(1+e^{-10(t-0.5)})$",
        "func": lambda t: np.clip(0.05 + 0.80 / (1 + np.exp(-20 * (t - 0.5))), 1e-4, 0.9999)
    },
    {
        "name": "Exponential Decay",
        "label": "$H(t) = 0.25 + 0.7e^{-6t}$",
        "func": lambda t: np.clip(0.25 + 0.7 * np.exp(-6 * t), 1e-4, 0.9999)
    },
    {
        "name": "Piecewise Constant",
        "label": "$H(t) = 0.1 \\cdot \\mathbf{1}_{t \\leq 0.4} "
                 "+ 0.4 \\cdot \\mathbf{1}_{0.4 < t \\leq 0.7}+ 0.75 \\cdot \\mathbf{1}_{t > 0.7}$",
        "func": lambda t: 0.1 if t <= 0.4 else 0.4 if t <= 0.7 else 0.75
    },
]

###############################################################################
# Covariance kernel 
###############################################################################

def cov_mBm(t, s, H_func):
    """
    Covariance of the Volterra mBm:
    
        Cov(B(t), B(s)) = sqrt(2H(t)) * sqrt(2H(s))
                                    * integral_0^{min(t,s)} (t-u)^{H(t)-0.5}
                                        (s-u)^{H(s)-0.5} du
    
    Computed via numerical quadrature since no closed form exists
    for general H(t).
    """
    if t == 0 or s == 0:
        return 0.0

    Ht = H_func(t)
    Hs = H_func(s)
    upper = min(t, s)

    def integrand(u):
        # Small buffer to avoid singularity at u = upper limit
        dt = t - u
        ds = s - u
        if dt <= 0 or ds <= 0:
            return 0.0
        return (dt ** (Ht - 0.5)) * (ds ** (Hs - 0.5))

    result, _ = quad(integrand, 0, upper, limit=200,
                     points=[upper * 0.99])  # hint near singularity
    return np.sqrt(2 * Ht) * np.sqrt(2 * Hs) * result

###############################################################################
# Covariance matrix 
###############################################################################

def build_covariance_matrix(t_grid, H_func):
    """
    Builds the n x n covariance matrix C where
    C[i,j] = Cov(B(t_i), B(t_j)).
    
    Since C is symmetric we only compute the upper triangle
    and mirror it, halving the number of quadrature calls.
    """
    n = len(t_grid)
    C = np.zeros((n, n))
    
    print(f"  Building {n}x{n} covariance matrix ({n*(n+1)//2} entries)...")
    for i in range(n):
        for j in range(i, n):   # upper triangle only
            C[i, j] = cov_mBm(t_grid[i], t_grid[j], H_func)
            C[j, i] = C[i, j]  # exploit symmetry
    return C

###############################################################################
# Cholesky simulation 
###############################################################################

def simulate_mBm_cholesky(T, n_steps, n_paths, H_func,
                           seed=42, include_zero=True):
    """
    Simulates paths of the Volterra multifractional Brownian motion
    using the Cholesky decomposition method.

    Parameters:
    -----------
    T          : float  — time horizon
    n_steps    : int    — number of interior time steps
    n_paths    : int    — number of sample paths to generate
    H_func : callable — Hurst function H:[0,T]->(0,0.5)
    seed       : int    — random seed for reproducibility
    include_zero : bool — whether to prepend B(0)=0

    Returns:
    --------
    t_grid : np.ndarray, shape (n_steps,)
    paths  : np.ndarray, shape (n_paths, n_steps)
    """
    # Time grid — exclude t=0 since B(0)=0 exactly
    t_grid = np.linspace(T / n_steps, T, n_steps)

    # Build covariance matrix
    C = build_covariance_matrix(t_grid, H_func)

    # Regularise for numerical stability — adds a small diagonal
    # perturbation to ensure positive definiteness
    epsilon = 1e-10
    C += epsilon * np.eye(len(t_grid))

    # Cholesky decomposition: C = L L^T
    print("  Computing Cholesky factorisation...")
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Covariance matrix is not positive definite. "
            "Try increasing epsilon or checking H_func."
        )

    # Generate n_paths independent standard normal vectors
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(len(t_grid), n_paths))

    # Paths: each column of L @ Z is one sample path
    # Shape: (n_steps, n_paths) -> transpose to (n_paths, n_steps)
    paths = (L @ Z).T

    if include_zero:
        # Prepend B(0) = 0 for all paths
        zeros = np.zeros((n_paths, 1))
        paths = np.concatenate([zeros, paths], axis=1)
        t_grid = np.concatenate([[0.0], t_grid])

    return t_grid, paths

###############################################################################
# Plot 
###############################################################################

def plot_single_spec(ax_path, ax_hurst, t_grid, paths, H_func, label):
    
    colors = plt.cm.tab10(np.linspace(0, 1, paths.shape[0]))
    
    for i, path in enumerate(paths):
        ax_path.plot(t_grid, path, color=colors[i], lw=1.2, alpha=0.85, label=f"Path {i+1}")
        
    ax_path.axhline(0, color="black", lw=0.5, ls="--")
    ax_path.set_ylabel("$B^{H(\\cdot)}_t$", fontsize=11)
    ax_path.set_title(label, fontsize=12)
    ax_path.legend(fontsize=8, loc="upper left")
    ax_path.grid(True, alpha=0.3)

    H_vals = [H_func(t) for t in t_grid]
    ax_hurst.plot(t_grid, H_vals, color="black", lw=1.5)
    ax_hurst.set_xlabel("$t$", fontsize=11)
    ax_hurst.set_ylabel("$H(t)$", fontsize=11)
    ax_hurst.set_ylim(0, 1.0)
    ax_hurst.grid(True, alpha=0.3)

###############################################################################
# Main 
###############################################################################

if __name__ == "__main__":

    T       = 1.0
    n_steps = 1000
    n_paths = 3
    seed    = 42

    for idx, spec in enumerate(H_SPECS):
        print(f"[{idx+1}/{len(H_SPECS)}] Simulating: {spec['name']}...")

        t_grid, paths = simulate_mBm_cholesky(
            T       = T,
            n_steps = n_steps,
            n_paths = n_paths,
            H_func  = spec["func"],
            seed    = seed
        )

        # One figure per specification
        fig, (ax_path, ax_hurst) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(10, 7),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        plot_single_spec(
            ax_path  = ax_path,
            ax_hurst = ax_hurst,
            t_grid   = t_grid,
            paths    = paths,
            H_func   = spec["func"],
            label    = spec["label"]
        )

        plt.suptitle(
            f"mBm Sample Paths — {spec['name']}",
            fontsize=14
        )
        plt.tight_layout()

        # Save each figure separately
        filename = f"mBm_{spec['name'].replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.show()

        print(f"    Saved: {filename}")
        print(f"    H(0) = {spec['func'](0):.3f}, "
              f"H(T) = {spec['func'](T):.3f}")

    print("Done — all specifications plotted.")


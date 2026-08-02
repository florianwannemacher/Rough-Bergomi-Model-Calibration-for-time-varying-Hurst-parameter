# Rough-Bergomi-Model-Calibration-for-time-varying-Hurst-parameter

This repository contains the implementation and calibration of the Rough Bergomi model and its extensions, including a multifractional Rough Bergomi approach, using multifractional Brownian motion to drive volatility. The work was conducted as part of a thesis project focused on modeling stochastic volatility with roughness features and calibrating these models to market data.

# Repository Structure
* Cholesky_Decomposition.py : Generates paths of multifractional Brownian motion using the Cholesky Decomposition method
* Implied Vol_multifractional_rBergomi.py : Generates implied volatility smiles for different C^1 specifications of H(t) under the rough Bergomi framework
* Implied Vol_multifractional_rBergomi_polynomials.py : Generates implied volatility smiles for different polynomial (up to degree 3) specifications of H(t) under the rough Bergomi framework
* Data_Preprocessing_and_IV_calculations.py :  Generates implied volatility surfaces and compares them with real market data.
* Effects_of_H_eta_rho_on_implied_vol : Extension of the rBergomi model to allow different C^1 specifications of H(t); effects of H(t), rho, eta on implied volatility smiles 
* rBergomi_calibration.py : Calibration routines combining global optimization (DE) with local refinement (SLSQP).

# Information
**Thesis Name:** Modeling Rough Volatility with Time-Varying Hurst Parameter in the Rough Bergomi Framework
**Author:** Florian Wannemacher
**Institution:** University of Edinburgh
**Program:** Financial Modelling and Optimization MSc
**Year:** 2026

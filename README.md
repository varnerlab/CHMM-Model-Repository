# HMMs-withJumps

This Julia project provides a framework for simulating stock market returns using a hybrid modeling approach that combines Hidden Markov Models (HMMs) with a Poisson jump process. It leverages daily ticker data including Open, High, Low, Close (OHLC), Volume, and Volume Weighted Average Price (VWAP).

## Overview

Financial markets exhibit complex behaviors such as volatility clustering, heavy-tailed distributions, and regime-dependent dynamics, making classical models inadequate. This project addresses these challenges by:

* Implementing discrete-state HMMs to model distinct market regimes (e.g., favorable, unfavorable market conditions).
* Integrating a Poisson jump process to account for sudden, significant market movements.
* Using empirical data calibration for realistic modeling of state transitions.

## Features

* **Modular pipeline** for constructing HMM-based return models for any given ticker.
* Simulation capabilities reflecting real-world statistical features including:

  * Volatility clustering
  * Heavy-tailed return distributions
  * Absence of autocorrelation
* Robust validation framework using Kolmogorov-Smirnov tests.

## Data

The current implementation is validated with a comprehensive dataset covering:

* Over 400 U.S.-listed equities and ETFs
* Historical data spanning 2,515 trading days
* Representative tickers such as NVDA, AAPL, and SPY

## Prerequisites and Dependencies

Ensure you have Julia (version 1.9 or higher recommended) installed. Install required packages by running:

```julia
using Pkg
Pkg.add([
    "CSV", "DataFrames", "Distributions", "StatsBase", "Plots", "HMMBase", 
    "JLD2", "FileIO", "Colors", "StatsPlots", "HypothesisTests", "Distances", 
    "PQPolygonSDK", "Dates", "HTTP", "JSON3", "VLQuantitativeFinancePackage", "LinearAlgebra", "Statistics"
])
```

## Running Simulations

For interactive use and experimentation, a Jupyter notebook (`simulation_notebook.ipynb`) is also provided.

To run simulations, execute the provided script:

```julia
include("include.jl")
```

Adjust parameters in the configuration file as needed for specific tickers and simulation scenarios.

## Validation

In-sample and out-of-sample validations were performed using standard statistical tests. Results confirmed high accuracy in replicating historical data properties.

## Contributing

Contributions to improve functionality or extend capabilities are welcome. Please create pull requests with clear explanations of proposed enhancements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Disclaimer

This project is intended **for research and educational purposes only**.  
It does **not** constitute financial advice, investment recommendations, or trading strategies.  

The models and simulations provided are simplified representations of financial markets and may not capture all real-world complexities.  
Users are solely responsible for any decisions made using this code.

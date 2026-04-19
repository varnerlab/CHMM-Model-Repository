# ========================================================================================= #
# Pricing.jl: Option Pricing via CHMM Regime-Switching Volatility
# ========================================================================================= #


# --- ANALYTICAL BENCHMARK ---------------------------------------------------- #

"""
    black_scholes(contract::MyEuropeanOptionContract, sigma::Float64) -> Float64

Analytical Black-Scholes price for a European option.
Used as a benchmark and for implied volatility inversion.
"""
function black_scholes(contract::MyEuropeanOptionContract, sigma::Float64)::Float64

    S0 = contract.S0; K = contract.K; T = contract.T; r = contract.r;

    d1 = (log(S0 / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T));
    d2 = d1 - sigma * sqrt(T);

    Φ = Normal(0.0, 1.0);

    if contract.is_call
        return S0 * cdf(Φ, d1) - K * exp(-r * T) * cdf(Φ, d2);
    else
        return K * exp(-r * T) * cdf(Φ, -d2) - S0 * cdf(Φ, -d1);
    end
end


# --- PRIVATE: PATH SIMULATORS ----------------------------------------------- #

"""
    _simulate_chmm_price_path(model::MyCHMMPricingModel, contract::MyEuropeanOptionContract) -> Float64

Private: Simulates one GBM price path driven by regime-switching volatility.
Returns the terminal price S(T).
"""
function _simulate_chmm_price_path(model::MyCHMMPricingModel, contract::MyEuropeanOptionContract)::Float64

    n_steps = ceil(Int, contract.T * model.n_steps_per_year);
    dt = contract.T / n_steps;

    # Simulate hidden state path using the HMM functor
    s0 = rand(model.start_distribution);
    state_path = model.hmm(s0, n_steps);

    # GBM with regime-switching vol
    S = contract.S0;
    for t in 1:n_steps
        σ_t = model.volatility_map[state_path[t]];
        Z = randn();
        S = S * exp((contract.r - σ_t^2 / 2) * dt + σ_t * sqrt(dt) * Z);
    end

    return S;
end


# --- PUBLIC: PRICING FUNCTIONS ----------------------------------------------- #

"""
    price(model::MyCHMMPricingModel, contract::MyEuropeanOptionContract) -> MyPricingResult

Prices a European option using CHMM regime-switching Monte Carlo.
Simulates n_paths price paths, computes discounted payoffs, returns mean ± std error.
"""
function price(model::MyCHMMPricingModel, contract::MyEuropeanOptionContract)::MyPricingResult

    payoffs = zeros(model.n_paths);

    for i in 1:model.n_paths
        S_T = _simulate_chmm_price_path(model, contract);

        if contract.is_call
            payoffs[i] = max(S_T - contract.K, 0.0);
        else
            payoffs[i] = max(contract.K - S_T, 0.0);
        end
    end

    # Discount to present value
    discount = exp(-contract.r * contract.T);
    discounted = payoffs .* discount;

    p = mean(discounted);
    se = std(discounted) / sqrt(model.n_paths);

    return MyPricingResult(p, se, model.n_paths, discounted);
end


# --- IMPLIED VOLATILITY ----------------------------------------------------- #

"""
    implied_volatility(contract::MyEuropeanOptionContract, market_price::Float64;
        tol::Float64=1e-6, max_iter::Int64=100) -> Float64

Finds the Black-Scholes implied volatility that matches the given market price,
using bisection on the interval [1e-4, 5.0].
"""
function implied_volatility(contract::MyEuropeanOptionContract, market_price::Float64;
    tol::Float64=1e-6, max_iter::Int64=100)::Float64

    σ_lo = 1e-4; σ_hi = 5.0;

    for _ in 1:max_iter
        σ_mid = (σ_lo + σ_hi) / 2;
        bs_price = black_scholes(contract, σ_mid);
        err = bs_price - market_price;

        if abs(err) < tol
            return σ_mid;
        elseif err > 0
            σ_hi = σ_mid;
        else
            σ_lo = σ_mid;
        end
    end

    return (σ_lo + σ_hi) / 2;
end


# --- SURFACE GENERATION ----------------------------------------------------- #

"""
    implied_vol_surface(model::AbstractPricingModel, S0::Float64, r::Float64,
        strikes::Array{Float64,1}, expiries::Array{Float64,1};
        is_call::Bool=true) -> Array{Float64,2}

Computes the implied volatility surface by pricing a grid of options under the
given model and inverting each price through Black-Scholes.
Returns a matrix of implied vols (strikes × expiries).
"""
function implied_vol_surface(model::AbstractPricingModel, S0::Float64, r::Float64,
    strikes::Array{Float64,1}, expiries::Array{Float64,1};
    is_call::Bool=true)::Array{Float64,2}

    nk = length(strikes); nt = length(expiries);
    iv_matrix = zeros(nk, nt);

    for (j, T_val) in enumerate(expiries)
        for (i, K_val) in enumerate(strikes)
            contract = build(MyEuropeanOptionContract, (
                S0=S0, K=K_val, T=T_val, r=r, is_call=is_call));

            result = price(model, contract);
            iv_matrix[i, j] = implied_volatility(contract, result.price);
        end
    end

    return iv_matrix;
end

# ----------------------------------------------------------------------------- #

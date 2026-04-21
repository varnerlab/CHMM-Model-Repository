# ========================================================================================= #
# Pricing.jl: Black-Scholes benchmark and implied-volatility inversion
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


# ----------------------------------------------------------------------------- #

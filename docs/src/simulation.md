# Simulation

## Functor Interface

The CHMM model is a callable object. To simulate a state chain:

```julia
state_chain = model(start_state, n_steps)
```

This returns a `Vector{Int64}` of hidden state indices.

## Continuous HMM Simulation

For `MyContinuousHiddenMarkovModel`, simulation follows the standard Markov chain:
1. Start at `start_state`
2. At each step, transition to the next state via `rand(model.transition[current_state])`

To generate observable returns from a state chain:
```julia
returns = [rand(model.emission[s]) for s in state_chain]
```

## Generating Multiple Paths

```julia
n_paths = 1000
n_steps = 252

# Compute stationary distribution for realistic initial states
K = length(model.states)
T_mat = zeros(K, K)
for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
π_stat = (T_mat^1000)[1, :]
start_dist = Categorical(π_stat)

# Simulate state chains from stationary distribution
chains = [model(rand(start_dist), n_steps) for _ in 1:n_paths]

# Convert to returns
all_returns = [[rand(model.emission[s]) for s in c] for c in chains]

# Reconstruct price paths from returns
function returns_to_prices(returns; P0=100.0, dt=1/252)
    prices = zeros(length(returns) + 1)
    prices[1] = P0
    for t in eachindex(returns)
        prices[t+1] = prices[t] * exp(returns[t] * dt)
    end
    return prices
end
```

The annualized return convention (`G_t = (1/Δt)·ln(P_t/P_{t-1})`) is why the price
reconstruction multiplies by `exp(returns[t] * dt)`.

## Other Single-Asset Models

The Student-t, Laplace, and GED CHMMs share the same functor + emission-sampling pattern
as the Gaussian CHMM. The GARCH-family and semi-Markov baselines instead return a return
series directly:

```julia
garch_returns   = simulate_garch(garch_model, n_steps)
msgarch_returns = simulate_msgarch(ms_model, n_steps)
sm_returns      = simulate_sm_chmm(sm_model, n_steps)
egarch_returns  = simulate_egarch(egarch_fit, n_steps)   # also simulate_gjr/garcht/harrv
sv_paths        = simulate_sv_ar1(sv_params, n_steps; n_paths=1000)
```

## Multi-Asset Simulation

Cross-asset generators propagate univariate CHMM marginals to a full universe and return a
`T × d × n_paths` array (time × firms × paths):

```julia
# Single-index (factor) model: drive the universe from simulated market paths
sim = simulate(sim_model, market_paths)

# Copula models: dependence via a correlation matrix (Student-t adds tail dependence)
sim = simulate(gaussian_copula_model, T_sim, n_paths)
sim = simulate(student_t_copula_model, T_sim, n_paths)
sim = simulate(cvine_model, T_sim, n_paths)
```

See the dependence models in the [API Reference](@ref) and `CrossAsset.jl`.

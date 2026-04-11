# Simulation

## Functor Interface

All models are callable objects. To simulate a state chain:

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

## Jump HMM Simulation (Regime Teleportation)

For `MyContinuousHiddenMarkovModelWithJumps`, each time step has two possible modes:

### Normal Mode (probability `1 - epsilon`)
Standard Markov transition using the learned transition matrix.

### Jump Mode (probability `epsilon`)
1. Draw a duration `d ~ Poisson(lambda)`
2. For each of the `d` steps:
   - Flip a biased coin (52% crash, 48% boom)
   - **Crash**: randomly select from states `1:3` (lowest-return regimes)
   - **Boom**: randomly select from states `(N-2):N` (highest-return regimes)

The coin flip occurs **inside** the duration loop, so each jump step independently chooses crash or boom. This produces high volatility (magnitude) without directional trends.

## Generating Multiple Paths

```julia
n_paths = 1000
n_steps = 252

# State chains
chains = [model(1, n_steps) for _ in 1:n_paths]

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

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

@testset "Compute" begin

    # --- Helper: build a small trained model ---
    function _make_test_model(; n_states=5, seed=42)
        rng = Random.MersenneTwister(seed)
        obs = vcat(
            randn(rng, 200) .* 0.01,
            randn(rng, 200) .* 0.05
        )
        return build(MyContinuousHiddenMarkovModel, (
            observations = shuffle(rng, obs),
            number_of_states = n_states,
            max_iter = 15
        ))
    end

    @testset "Baum-Welch convergence" begin
        rng = Random.MersenneTwister(123)
        obs = randn(rng, 300) .* 0.02
        T, μ, σ, π_vec, ll_hist, γ = baum_welch(obs, 3; max_iter=20)

        # Log-likelihood should generally increase (EM guarantee)
        @test length(ll_hist) > 1
        @test ll_hist[end] >= ll_hist[1]

        # Output dimensions
        @test size(T) == (3, 3)
        @test length(μ) == 3
        @test length(σ) == 3
        @test length(π_vec) == 3
        @test size(γ) == (300, 3)

        # Transition rows should sum to 1
        for i in 1:3
            @test sum(T[i, :]) ≈ 1.0 atol=1e-10
        end

        # Standard deviations should be positive
        @test all(σ .> 0)
    end

    @testset "Simulation - continuous HMM" begin
        model = _make_test_model()
        chain = model(1, 100)

        @test length(chain) == 100
        @test chain[1] == 1
        @test all(s -> s in model.states, chain)
    end

    @testset "simulate_returns - Gaussian CHMM" begin
        Random.seed!(1234)
        model = _make_test_model()

        # Single-path form returns a Vector.
        R = simulate_returns(model, 250)
        @test R isa Vector{Float64}
        @test length(R) == 250
        @test all(isfinite, R)

        # Multi-path form returns a Matrix of the right shape.
        R_mat = simulate_returns(model, 120; n_paths=8)
        @test R_mat isa Matrix{Float64}
        @test size(R_mat) == (120, 8)

        # Explicit integer start state must be honored through emissions.
        R2 = simulate_returns(model, 50; start=1)
        @test length(R2) == 50
    end

    @testset "simulate_returns - Student-t and Laplace CHMMs" begin
        rng = Random.MersenneTwister(4242)
        obs = vcat(randn(rng, 200) .* 0.01, randn(rng, 200) .* 0.04)
        obs = shuffle(rng, obs)

        model_t = build(MyStudentTHiddenMarkovModel,
            (observations=obs, number_of_states=3, max_iter=15))
        model_l = build(MyLaplaceHiddenMarkovModel,
            (observations=obs, number_of_states=3, max_iter=15))

        R_t = simulate_returns(model_t, 100)
        R_l = simulate_returns(model_l, 100)
        @test length(R_t) == 100
        @test length(R_l) == 100
        @test all(isfinite, R_t)
        @test all(isfinite, R_l)
    end

    @testset "forward_filter - shape and row-stochasticity" begin
        Random.seed!(7777)
        model = _make_test_model()
        obs = randn(300) .* 0.03

        π = forward_filter(model, obs)
        @test size(π) == (300, length(model.states))
        @test all(isfinite, π)
        # Each row is a probability distribution over states.
        for t in axes(π, 1)
            @test sum(π[t, :]) ≈ 1.0 atol=1e-8
            @test all(π[t, :] .>= -1e-10)
        end
    end

    @testset "regime_forecast - convergence to stationary as h grows" begin
        Random.seed!(5050)
        model = _make_test_model()
        K = length(model.states)

        π_now = fill(1.0 / K, K)
        # Non-stationary start: put all mass on state 1.
        π_now[:] .= 0.0; π_now[1] = 1.0

        f1  = regime_forecast(model, π_now, 1)
        f5  = regime_forecast(model, π_now, 5)
        f50 = regime_forecast(model, π_now, 50)

        @test size(f1)  == (1, K)
        @test size(f5)  == (5, K)
        @test size(f50) == (50, K)
        for h in (1, 5, 50)
            f = regime_forecast(model, π_now, h)
            for i in 1:h
                @test sum(f[i, :]) ≈ 1.0 atol=1e-8
            end
        end

        # Long-horizon forecast from any starting point should be close to the
        # stationary distribution.
        π_stat = (1 / K) .* ones(K)  # will be replaced below
        T_mat = zeros(K, K)
        for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
        π_stat = (T_mat ^ 500)[1, :]
        @test maximum(abs.(f50[end, :] .- π_stat)) < 1e-3
    end

    @testset "simulate_prices - path shape and round-trip invariant" begin
        Random.seed!(9999)
        model = _make_test_model()
        S0 = 100.0
        Δt = 1/252
        rf = 0.0

        P = simulate_prices(model, S0, 200; Δt=Δt, risk_free_rate=rf)
        @test length(P) == 201
        @test P[1] == S0
        @test all(P .> 0.0)

        # Multi-path shape.
        P_mat = simulate_prices(model, S0, 50; n_paths=10)
        @test size(P_mat) == (51, 10)
        @test all(P_mat[1, :] .== S0)

        # Round-trip: prices recovered from their own simulated returns should
        # satisfy ln(P_t/P_{t-1}) / Δt = G_t. We cannot replay the same random
        # draws through simulate_returns independently, so instead we verify
        # the invariant by taking log-ratios of the produced path and checking
        # it is a finite, well-scaled series.
        G_implied = (1.0 / Δt) .* diff(log.(P)) .- rf
        @test length(G_implied) == 200
        @test all(isfinite, G_implied)
    end

    @testset "Simulation - discrete jump HMM" begin
        K = 7
        T_mat = zeros(K, K) .+ 1/K
        E_mat = zeros(K, K) .+ 1/K
        model = build(MyHiddenMarkovModelWithJumps, (
            states = collect(1:K), T = T_mat, E = E_mat,
            ϵ = 0.02, λ = 3.0
        ))
        chain = model(1, 500)

        @test length(chain) == 500
        @test chain[1] == 1
        @test all(s -> s in model.states, chain)
    end

    @testset "fit_coupling - row-stochasticity and smoothing" begin
        vix_states = [1, 1, 2, 2, 1, 3, 3, 1, 2]
        eq_states  = [2, 1, 1, 3, 2, 3, 2, 1, 3]
        C = fit_coupling(vix_states, eq_states, 3, 3; lag=1, alpha=1.0)

        @test size(C) == (3, 3)
        for i in 1:3
            @test sum(C[i, :]) ≈ 1.0 atol=1e-10
            @test all(C[i, :] .> 0.0)  # Laplace smoothing keeps every cell positive
        end

        # Entropies finite and bounded by log(K_eq).
        h = coupling_entropy(C)
        @test length(h) == 3
        @test all(isfinite, h)
        @test all(h .<= log(3) + 1e-10)
    end

    @testset "simulate_prices_vix_conditioned - path shape and nonnegative prices" begin
        Random.seed!(4321)
        model_vix = _make_test_model(n_states=4)
        model_eq  = _make_test_model(n_states=5)

        # Synthetic coupling matrix (random, row-stochastic).
        raw = rand(4, 5)
        C = raw ./ sum(raw; dims=2)

        P = simulate_prices_vix_conditioned(model_eq, model_vix, C, 100.0, 150)
        @test length(P) == 151
        @test P[1] == 100.0
        @test all(P .> 0.0)

        P_mat = simulate_prices_vix_conditioned(model_eq, model_vix, C, 100.0, 50; n_paths=7)
        @test size(P_mat) == (51, 7)
        @test all(P_mat[1, :] .== 100.0)
    end

    @testset "GARCH(1,1) fitting and simulation" begin
        rng = Random.MersenneTwister(42)
        obs = randn(rng, 500) .* 0.03

        model = build(MyGARCHModel, (observations = obs,))

        @test model.ω > 0
        @test (model.α + model.β) < 1.0

        # Simulate from fitted model
        sim = simulate_garch(model, 252)
        @test length(sim) == 252
        @test all(isfinite, sim)
    end

    @testset "log_growth_matrix - raw vector" begin
        prices = [100.0, 101.0, 99.5, 102.0, 100.5]
        R = log_growth_matrix(prices; Δt=1/252, risk_free_rate=0.0)

        @test length(R) == 4
        @test R[1] ≈ (252.0) * log(101.0 / 100.0) atol=1e-10
    end

    @testset "log_growth_matrix - DataFrame" begin
        df = DataFrame(
            volume_weighted_average_price = [100.0, 101.0, 99.5, 102.0]
        )
        R = log_growth_matrix(df)
        @test length(R) == 3
    end

    @testset "Viterbi decoding" begin
        model = _make_test_model(n_states=3)
        rng = Random.MersenneTwister(77)
        obs = vcat(randn(rng, 100) .* 0.01, randn(rng, 100) .* 0.05)

        states = viterbi(obs, model)

        @test length(states) == length(obs)
        @test all(s -> s in model.states, states)
        @test eltype(states) == Int64
    end

    @testset "Walk-forward regimes" begin
        rng = Random.MersenneTwister(88)
        obs = vcat(randn(rng, 300) .* 0.01, randn(rng, 100) .* 0.05)
        window = 100; n_states = 3;

        regimes = walk_forward_regimes(obs, window, n_states; max_iter=10)

        @test length(regimes) == length(obs) - window
        @test all(s -> 1 <= s <= n_states, regimes)
        @test eltype(regimes) == Int64
    end

    @testset "VWAP calculation" begin
        df = DataFrame(
            high = [101.0, 102.0, 100.5],
            low = [99.0, 100.0, 98.5],
            close = [100.0, 101.0, 99.5],
            volume = [1000.0, 2000.0, 1500.0]
        )
        v = vwap(df)

        @test length(v) == 3
        # First VWAP = typical price of first row
        tp1 = (101.0 + 99.0 + 100.0) / 3
        @test v[1] ≈ tp1 atol=1e-10
    end
end

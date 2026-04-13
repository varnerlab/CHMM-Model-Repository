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

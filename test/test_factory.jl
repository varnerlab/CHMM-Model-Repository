@testset "Factory" begin

    # --- Helper: generate synthetic observations ---
    function _make_synthetic_observations(; n=500, seed=42)
        rng = Random.MersenneTwister(seed)
        obs = vcat(
            randn(rng, n ÷ 2) .* 0.01,
            randn(rng, n ÷ 2) .* 0.05
        )
        return shuffle(rng, obs)
    end
    # _make_synthetic_observations is called inside nested @testset blocks below

    @testset "Build MyContinuousHiddenMarkovModel" begin
        obs = _make_synthetic_observations()
        model = build(MyContinuousHiddenMarkovModel, (
            observations = obs,
            number_of_states = 3,
            max_iter = 10
        ))

        @test model isa MyContinuousHiddenMarkovModel
        @test length(model.states) == 3
        @test length(model.emission) == 3
        @test length(model.transition) == 3
        @test length(model.log_likelihood_history) > 0

        for s in model.states
            @test model.emission[s] isa Normal
        end

        for s in model.states
            @test model.transition[s] isa Categorical
        end
    end

    @testset "Build MyHiddenMarkovModel (discrete)" begin
        K = 3
        T = [0.7 0.2 0.1; 0.1 0.8 0.1; 0.2 0.1 0.7]
        E = [0.6 0.3 0.1; 0.1 0.6 0.3; 0.3 0.1 0.6]

        model = build(MyHiddenMarkovModel, (
            states = collect(1:K),
            T = T,
            E = E
        ))

        @test model isa MyHiddenMarkovModel
        @test length(model.states) == K
        @test length(model.transition) == K
        @test length(model.emission) == K
    end

    @testset "Build MyHiddenMarkovModelWithJumps" begin
        K = 7
        T = zeros(K, K) .+ 1/K
        E = zeros(K, K) .+ 1/K

        model = build(MyHiddenMarkovModelWithJumps, (
            states = collect(1:K),
            T = T,
            E = E,
            ϵ = 0.02,
            λ = 3.0
        ))

        @test model isa MyHiddenMarkovModelWithJumps
        @test model.ϵ == 0.02
        @test model.λ == 3.0
        @test model.jump_distribution isa Poisson
        @test length(model.states) == K
    end

    @testset "Build MyGARCHModel" begin
        rng = Random.MersenneTwister(42)
        obs = randn(rng, 500) .* 0.03

        model = build(MyGARCHModel, (observations = obs,))

        @test model isa MyGARCHModel
        @test model.ω > 0
        @test model.α ≥ 0
        @test model.β ≥ 0
        @test (model.α + model.β) < 1.0
        @test length(model.σ2_history) == 500
        @test isfinite(model.log_likelihood)
    end
end

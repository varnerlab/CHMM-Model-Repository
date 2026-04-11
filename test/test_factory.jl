@testset "Factory" begin

    # --- Helper: generate synthetic observations ---
    function _make_synthetic_observations(; n=500, seed=42)
        rng = Random.MersenneTwister(seed)
        # Two-regime synthetic data: low-vol and high-vol
        obs = vcat(
            randn(rng, n ÷ 2) .* 0.01,    # regime 1: low volatility
            randn(rng, n ÷ 2) .* 0.05      # regime 2: high volatility
        )
        return shuffle(rng, obs)  # mix regimes
    end

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

        # Emission distributions should be Normal
        for s in model.states
            @test model.emission[s] isa Normal
        end

        # Transition distributions should be Categorical
        for s in model.states
            @test model.transition[s] isa Categorical
        end
    end

    @testset "Build MyContinuousHiddenMarkovModelWithJumps" begin
        obs = _make_synthetic_observations()
        base = build(MyContinuousHiddenMarkovModel, (
            observations = obs,
            number_of_states = 5,
            max_iter = 10
        ))

        model = build(MyContinuousHiddenMarkovModelWithJumps, (
            base_model = base,
            epsilon = 0.02,
            lambda = 2.0
        ))

        @test model isa MyContinuousHiddenMarkovModelWithJumps
        @test model.ϵ == 0.02
        @test model.λ == 2.0
        @test model.jump_distribution isa Poisson
        @test length(model.states) == 5
        @test model.emission == base.emission
        @test model.transition == base.transition
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
end

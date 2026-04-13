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

end

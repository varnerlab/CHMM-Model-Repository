@testset "Visualize" begin

    @testset "plot_acf_comparison returns a plot" begin
        rng = Random.MersenneTwister(99)
        observed = randn(rng, 500)
        simulated = randn(rng, 500)

        p = plot_acf_comparison(observed, simulated, "Test ACF", 1; L=50)
        @test p isa Plots.Plot
    end

    @testset "plot_acf_comparison with absolute values" begin
        rng = Random.MersenneTwister(99)
        observed = randn(rng, 500)
        simulated = randn(rng, 500)

        p = plot_acf_comparison(observed, simulated, "Test |ACF|", 1; is_absolute=true, L=50)
        @test p isa Plots.Plot
    end

    @testset "plot_regime_overlay returns a plot" begin
        dates = collect(1:100)
        prices = cumsum(randn(Random.MersenneTwister(42), 100)) .+ 100.0
        states = rand(Random.MersenneTwister(42), 1:3, 100)

        p = plot_regime_overlay(dates, prices, states, "TEST")
        @test p isa Plots.Plot
    end

    @testset "plot_emission_pdfs returns a plot" begin
        rng = Random.MersenneTwister(42)
        obs = vcat(randn(rng, 200) .* 0.01, randn(rng, 200) .* 0.05)
        model = build(MyContinuousHiddenMarkovModel, (
            observations = shuffle(rng, obs),
            number_of_states = 3,
            max_iter = 10
        ))

        p = plot_emission_pdfs(model, "TEST")
        @test p isa Plots.Plot
    end
end

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
end

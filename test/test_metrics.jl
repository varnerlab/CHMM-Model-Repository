@testset "Multistart Baum-Welch" begin
    # (2026-07-16 sixth review, findings 1-2: high-state fits need optimizer
    # evidence — per-start diagnostics, convergence flags, best-LL selection.)
    rng = Random.MersenneTwister(31)
    obs = vcat(randn(rng, 250) .* 0.01, randn(rng, 250) .* 0.04)[randperm(rng, 500)]
    T, μ, σ, πv, llh, γ, diags = baum_welch_multistart(obs, 3;
        n_starts = 3, max_iter = 400, tol = 1e-4, seed = 7)
    @test length(diags) == 3
    @test llh[end] ≈ maximum(d.ll for d in diags) atol = 1e-10
    @test llh[end] ≈ maximum(llh) atol = 1e-8
    for d in diags
        @test d.converged == (abs(d.final_increment) < 1e-4)
        @test d.n_evals <= 401
    end
    @test size(γ) == (500, 3)
    @test all(isfinite, llh)
end

@testset "Truncated discrete Pareto ML update" begin
    # (2026-07-16 sixth review, finding 4: the previous α = 1/E_w[log d] update
    # is the continuous untruncated formula, not the truncated-discrete MLE.)
    D = 30
    α_true = 1.2
    p = exp.(truncated_pareto_logpmf(α_true, D))
    @test sum(p) ≈ 1.0 atol = 1e-12
    w = 1000.0 .* p                      # exact expected counts under α_true
    α̂ = fit_truncated_pareto_alpha(w, D)
    @test abs(α̂ - α_true) < 1e-3         # exact-counts MLE recovers the truth

    ℓ(α) = sum(w[d] * truncated_pareto_logpmf(α, D)[d] for d in 1:D)
    # α̂ maximizes ℓ over a fine grid
    grid = 0.1:0.005:6.0
    @test ℓ(α̂) >= maximum(ℓ.(grid)) - 1e-6
    # the old continuous untruncated formula is NOT the maximizer here
    α_old = 1.0 / (sum(w[d] * log(Float64(d)) for d in 1:D) / sum(w))
    @test abs(α_old - α_true) > 0.05
    @test ℓ(α̂) > ℓ(α_old) + 1e-3
end

@testset "hsmm core uses the library censored truncated-Pareto ML update" begin
    core = read(joinpath(@__DIR__, "..", "runners", "baselines", "hsmm_core.jl"), String)
    @test occursin("fit_truncated_pareto_alpha(w[s, :], D", core)
    @test occursin("censored_counts", core)
    @test !occursin("1.0 / max(log_d_mean", core)
    runner = read(joinpath(@__DIR__, "..", "runners", "baselines", "run_hsmm_ml.jl"), String)
    @test occursin("hsmm_core.jl", runner)
end

@testset "Metrics utilities" begin

    @testset "_auc degenerate label sets return 0.5" begin
        # (2026-07-16 fifth review, code finding 8: `np == 0 || nn == 0 && return 0.5`
        # short-circuited so an empty POSITIVE class fell through to a 0/0 division.)
        scores = [0.1, 0.4, 0.35, 0.8, 0.65]
        @test _auc(scores, zeros(Int, 5)) == 0.5   # no positives
        @test _auc(scores, ones(Int, 5)) == 0.5    # no negatives
    end

    @testset "_auc separable and random labelings" begin
        y = [0, 0, 0, 1, 1, 1]
        @test _auc([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], y) == 1.0   # perfectly separable
        @test _auc([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], y) == 0.0   # perfectly inverted
        @test _auc([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], y) == 0.5   # all tied
    end
end

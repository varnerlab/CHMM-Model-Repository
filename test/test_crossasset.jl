@testset "CrossAsset CRN" begin

    # --- Helper: small correlated fixture with CHMM-N marginals and both copulas ---
    # (2026-07 final-rerun review, finding 1: the strict-CRN simulate methods
    # that back the seed-uncertainty panel need automated regression coverage.)
    function _make_crn_fixture(; d=3, T_obs=300)
        rng = Random.MersenneTwister(2026)
        L = [1.0 0.0 0.0; 0.6 0.8 0.0; 0.3 0.4 0.866]
        R = 0.02 .* (randn(rng, T_obs, d) * L')
        tickers = ["A$(j)" for j in 1:d]
        marginals = Vector{AbstractMarkovModel}(undef, d)
        for j in 1:d
            marginals[j] = build(MyContinuousHiddenMarkovModel, (
                observations = R[:, j],
                number_of_states = 2,
                max_iter = 10
            ))
        end
        gauss = build(MyGaussianCopulaModel, (returns = R, tickers = tickers, marginals = marginals))
        tcop = build(MyStudentTCopulaModel, (returns = R, tickers = tickers, marginals = marginals))
        return gauss, tcop
    end

    gauss_cop, t_cop = _make_crn_fixture()
    T_sim, n_paths, crn_seed = 120, 4, 90210

    @testset "CRN simulate is bitwise reproducible" begin
        for model in (gauss_cop, t_cop)
            a = simulate(model, T_sim, n_paths, crn_seed)
            b = simulate(model, T_sim, n_paths, crn_seed)
            @test a == b
        end
    end

    @testset "CRN simulate shape and finiteness" begin
        for model in (gauss_cop, t_cop)
            out = simulate(model, T_sim, n_paths, crn_seed)
            @test size(out) == (T_sim, 3, n_paths)
            @test all(isfinite, out)
        end
    end

    @testset "strict CRN: copulas share marginal draws, mixing changes ranks only" begin
        out_g = simulate(gauss_cop, T_sim, n_paths, crn_seed)
        out_t = simulate(t_cop, T_sim, n_paths, crn_seed)
        # Rank reordering permutes the same marginal values, so for every
        # asset and path the two copulas must produce the same multiset of
        # observations under a common crn_seed ...
        for p in 1:n_paths, j in 1:3
            @test sort(out_g[:, j, p]) == sort(out_t[:, j, p])
        end
        # ... while the Student-t chi-square mixing stream changes the
        # ordering (the copula ranks), so the tensors are not identical.
        @test out_g != out_t
    end

    @testset "different CRN seeds give different draws" begin
        for model in (gauss_cop, t_cop)
            @test simulate(model, T_sim, n_paths, crn_seed) !=
                  simulate(model, T_sim, n_paths, crn_seed + 1)
        end
    end

    @testset "t-copula profile log-likelihood is finite" begin
        # Guards the logabsgamma-based constant in _tcopula_profile_loglik
        # (replacement for the deprecated lgamma; 2026-07-15 review).
        rng = Random.MersenneTwister(7)
        U = clamp.(rand(rng, 100, 3), 1e-6, 1.0 - 1e-6)
        Σ = [1.0 0.3 0.2; 0.3 1.0 0.4; 0.2 0.4 1.0]
        for ν in (2.5, 6.0, 30.0)
            ll = _tcopula_profile_loglik(U, Σ, ν)
            @test isfinite(ll)
        end
    end

    @testset "seed_uncertainty artifact row reproduces (nonoverlap, seed 20260501)" begin
        # Runner-level smoke test: refit the non-overlapping basket exactly as
        # runners/cross_asset/run_seed_uncertainty.jl does (the fits are
        # deterministic given the data) and recompute one stored artifact row.
        assets = ["MSFT", "UNH", "BAC", "CAT", "PG", "XOM"]
        train_dataset = MyPortfolioDataSet()["dataset"]
        oos_dataset = MyOutOfSamplePortfolioDataSet()["dataset"]
        max_days = nrow(train_dataset["AAPL"])
        dataset = Dict{String,DataFrame}()
        for (t, data) in train_dataset
            if nrow(data) == max_days
                dataset[t] = data
            end
        end
        Δt = 1 / 252
        cols_is = [log_growth_matrix(dataset, t; Δt = Δt, risk_free_rate = 0.0) for t in assets]
        cols_oos = [log_growth_matrix(oos_dataset, t; Δt = Δt, risk_free_rate = 0.0) for t in assets]
        n_is = minimum(length.(cols_is))
        n_oos = minimum(length.(cols_oos))
        R_is = hcat([c[1:n_is] for c in cols_is]...)
        R_oos = hcat([c[1:n_oos] for c in cols_oos]...)

        marginals = Vector{AbstractMarkovModel}(undef, length(assets))
        for j in 1:length(assets)
            marginals[j] = build(MyContinuousHiddenMarkovModel, (
                observations = R_is[:, j],
                number_of_states = 3,
                max_iter = 60
            ))
        end
        gc = build(MyGaussianCopulaModel, (returns = R_is, tickers = assets, marginals = marginals))
        tc = build(MyStudentTCopulaModel, (returns = R_is, tickers = assets, marginals = marginals))

        mae_t = correlation_reproduction(R_oos, simulate(tc, n_oos, 200, 20260501)).offdiag_mae
        mae_g = correlation_reproduction(R_oos, simulate(gc, n_oos, 200, 20260501)).offdiag_mae

        csv_path = joinpath(@__DIR__, "..", "results", "cross_asset", "seed_uncertainty.csv")
        rows = filter(l -> startswith(l, "nonoverlap,20260501,"), readlines(csv_path))
        @test length(rows) == 1
        parts = split(rows[1], ",")
        @test round(mae_t, digits = 6) == parse(Float64, parts[3])
        @test round(mae_g, digits = 6) == parse(Float64, parts[4])
    end
end

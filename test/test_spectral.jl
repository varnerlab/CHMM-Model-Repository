include(joinpath(@__DIR__, "..", "runners", "spectral", "spectral_common.jl"))
include(joinpath(@__DIR__, "..", "runners", "spectral", "acf_capacity_common.jl"))
include(joinpath(@__DIR__, "..", "runners", "spectral", "exp_mode_common.jl"))

@testset "Spectral grouped components" begin

    # Cyclic-mixture transition matrix: T = 0.7 I + 0.3 P (P a 3-cycle) has one
    # unit eigenvalue and one complex-conjugate pair, so the grouped
    # decomposition must produce exactly ONE damped-oscillatory component.
    P = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]
    T = 0.7 * Matrix(1.0I, 3, 3) + 0.3 * P
    π̄ = fill(1/3, 3)              # doubly stochastic → uniform stationary law
    m = [1.0, 2.0, 4.0]           # per-state E|G|
    M = m.^2 .+ 2.0               # per-state EG² (> m² so σ² > 0)

    σ²_G, comps, κ_V, recon_err = _spectral_components(T, π̄, m, M)

    @test length(comps) == 1
    @test comps[1].kind == :pair
    @test comps[1].theta > 0.0
    # Signed real contribution: ρ(1) must equal the direct matrix formula.
    μ_G = π̄' * m
    ρ1_direct = (m' * Diagonal(π̄) * T * m - μ_G^2) / σ²_G
    @test comps[1].contrib.t1 ≈ ρ1_direct atol=1e-10
    # Full-curve reconstruction against matrix powers.
    @test recon_err < 1e-8

    # Real-spectrum control: a symmetric transition matrix has all-real
    # eigenvalues → K - 1 real components and the same reconstruction property.
    T2 = [0.90 0.08 0.02; 0.08 0.84 0.08; 0.02 0.08 0.90]
    π̄2 = (T2^2000)[1, :]
    σ²2, comps2, _, recon2 = _spectral_components(T2, π̄2, m, M)
    @test length(comps2) == 2
    @test all(c -> c.kind == :real, comps2)
    @test recon2 < 1e-8

    # Summary invariants: shares in (0, 1], budget ≥ |ρ(1)|.
    s2 = _component_summary(comps2)
    @test 0.0 < s2.dom_share <= 1.0
    @test s2.budget >= abs(s2.rho1) - 1e-12
    @test 1 <= s2.n_for_95 <= 2
end

@testset "ACF-targeted realizable HMM optimizer" begin
    # Parametrization roundtrip and validity.
    K = 4
    rng = MersenneTwister(7)
    Λ = 2.0 .* Matrix(1.0I, K, K) .+ randn(rng, K, K)
    T0 = exp.(Λ .- maximum(Λ, dims=2)); T0 ./= sum(T0, dims=2)
    μ0 = randn(rng, K); σ0 = exp.(0.3 .* randn(rng, K))
    θ = _pack_acf_params(T0, μ0, σ0)
    T1, μ1, σ1 = _unpack_acf_params(θ, K)
    @test maximum(abs.(T1 .- T0)) < 1e-12
    @test μ1 ≈ μ0 atol=1e-12
    @test σ1 ≈ σ0 atol=1e-12
    @test all(abs.(sum(T1, dims=2) .- 1.0) .< 1e-12)
    @test all(T1 .> 0.0)
    @test all(σ1 .> 0.0)

    # Unchecked stationary solve agrees with the checked one.
    @test _stationary_pi_unchecked(T0) ≈ _stationary_pi(T0) atol=1e-10

    # Vector-recursion population ACF agrees with the matrix-power formula.
    m0, M0 = _folded_moments(μ0, σ0)
    π̄0 = _stationary_pi(T0)
    @test _population_acf(T0, π̄0, m0, M0, 252) ≈ _theoretical_acf(T0, π̄0, m0, M0, 252) atol=1e-12

    # Degenerate points return the finite penalty, never NaN/throw.
    θ_deg = _pack_acf_params(fill(1/3, 3, 3), ones(3), fill(1e-9, 3))  # |G| ≈ 1 a.s. → σ²_G ≈ 0
    v_deg = _acf_objective(θ_deg, fill(0.1, 252), 3)
    @test isfinite(v_deg)
    @test v_deg == 1e6
    θ_nan = copy(θ); θ_nan[1] = NaN
    @test _acf_objective(θ_nan, fill(0.1, 252), K) == 1e6

    # Recovery: the optimizer must reproduce a target that IS an HMM ACF.
    K3 = 3
    T_true = [0.97 0.02 0.01; 0.015 0.97 0.015; 0.01 0.02 0.97]
    μ_true = zeros(K3); σ_true = [0.5, 1.0, 2.0]
    m_t, M_t = _folded_moments(μ_true, σ_true)
    π̄_t = _stationary_pi(T_true)
    ρ_true = _theoretical_acf(T_true, π̄_t, m_t, M_t, 252)
    fit = fit_acf_hmm(ρ_true, K3; n_starts=6, seed=11, R=nothing, n_iter=4000)
    @test fit.sse < 1e-5
    @test fit.near_mae < 1e-3
    @test all(abs.(sum(fit.T, dims=2) .- 1.0) .< 1e-10)
    @test all(fit.π̄ .>= 0.0)
    # Per-start diagnostics: Adam returns the best-so-far, never worse than the start.
    @test length(fit.diagnostics) == 6
    @test all(d.sse_final <= d.sse_init + 1e-12 for d in fit.diagnostics)
end

@testset "Unrestricted exponential-mode diagnostic core" begin
    maxlag = 252
    comps, bases = _build_dictionary(200, maxlag)
    grid = _lambda_grid(200)

    # Signed recovery: a positive and a NEGATIVE real mode. Component interaction
    # can tilt the greedy pick to a neighboring grid atom, so recovery is near-
    # exact (grid/refinement resolution), far below sample-ACF error scales.
    λa, λb = grid[40], grid[120]
    ρ_signed = [0.3 * λa^τ - 0.12 * (-λb)^τ for τ in 1:maxlag]
    sel, a, fit = _fit_m_modes(ρ_signed, 2, comps, bases, maxlag)
    @test maximum(abs.(fit .- ρ_signed)) < 2e-3
    @test any(c -> c[1] == :neg, sel)

    # Damped-oscillatory recovery: a conjugate-pair shape at an on-grid angle
    # costs 2 modes and must be reproduced at m = 2.
    λc = grid[100]; θc = π/4
    ρ_osc = [λc^τ * (0.4 * cos(θc*τ) + 0.1 * sin(θc*τ)) for τ in 1:maxlag]
    sel2, a2, fit2 = _fit_m_modes(ρ_osc, 2, comps, bases, maxlag)
    @test maximum(abs.(fit2 .- ρ_osc)) < 1e-8
    @test sel2[1][1] == :osc

    # Budget accounting: at m = 1 an oscillatory component (cost 2) is not
    # selectable, so the fit falls back to a single real mode.
    sel3, _, _ = _fit_m_modes(ρ_osc, 1, comps, bases, maxlag)
    @test length(sel3) == 1
    @test sel3[1][1] in (:pos, :neg)

    # HMM-shape coverage: the theoretical ACF of a genuine 3-state HMM with two
    # real modes is inside the class at m = 2 (lambda continuum via refinement).
    T2 = [0.90 0.08 0.02; 0.08 0.84 0.08; 0.02 0.08 0.90]
    π̄2 = _stationary_pi(T2)
    m_e = [1.0, 2.0, 4.0]; M_e = m_e.^2 .+ 2.0
    ρ_hmm = _theoretical_acf(T2, π̄2, m_e, M_e, maxlag)
    _, _, fit4 = _fit_m_modes(ρ_hmm, 2, comps, bases, maxlag)
    @test mean(abs.(fit4 .- ρ_hmm)) < 5e-4

    # Component serialization: coefficient count matches the mode structure.
    s = _comp_string(sel2, a2)
    @test startswith(s, "osc:")
    @test length(split(split(s, "|")[1], ":")) == 5
end

@testset "Frontier marginal helpers and nondominated set" begin
    π̄ = [0.5, 0.3, 0.2]; μ = [-0.5, 0.2, 1.0]; σ = [0.6, 1.0, 2.0]
    # quantile inverts the mixture CDF
    for q in (0.01, 0.25, 0.5, 0.9, 0.99)
        x = _mixture_quantile(π̄, μ, σ, q)
        @test abs(_mixture_cdf(x, π̄, μ, σ) - q) < 1e-9
    end
    # CvM of the mixture against its own quantile grid is ~0
    n = 400
    xq = [_mixture_quantile(π̄, μ, σ, (i - 0.5) / n) for i in 1:n]
    @test _marginal_cvm(π̄, μ, σ, xq) < 1e-12
    # λ = 0 joint objective equals the pure ACF objective; λ = Inf is CvM only
    rng = MersenneTwister(3)
    Λ = 2.0 .* Matrix(1.0I, 3, 3) .+ randn(rng, 3, 3)
    T0 = exp.(Λ .- maximum(Λ, dims=2)); T0 ./= sum(T0, dims=2)
    θ = _pack_acf_params(T0, μ, σ)
    ρ̂ = fill(0.05, 252)
    @test _joint_objective(θ, ρ̂, xq, 0.0, 3) ≈ _acf_objective(θ, ρ̂, 3) atol=1e-12
    @test _joint_objective(θ, ρ̂, xq, Inf, 3) ≈
          _marginal_cvm(_stationary_pi_unchecked(T0), μ, σ, xq) atol=1e-12
    # nondominated set on a hand-built configuration (minimize both coordinates)
    pts = [(1.0, 5.0), (2.0, 4.0), (3.0, 3.0), (2.5, 4.5), (1.5, 6.0)]
    nd = _nondominated(pts)
    @test (1 in nd) && (2 in nd) && (3 in nd)
    # 4 = (2.5, 4.5) is dominated by 2 = (2.0, 4.0); 5 = (1.5, 6.0) by 1 = (1.0, 5.0).
    @test !(4 in nd) && !(5 in nd)
end

@testset "ACF-capacity certificates reload and verify" begin
    cert_dir = joinpath(@__DIR__, "..", "results", "hmm_acf_capacity")
    files = filter(f -> endswith(f, ".jld2"), readdir(cert_dir))
    @test length(files) == 62                       # 31 tickers x K in {3, 18}
    csv = readlines(joinpath(@__DIR__, "..", "results", "diagnostics", "hmm_acf_capacity.csv"))
    rows = Dict{Tuple{String,Int},Vector{SubString{String}}}()
    for ln in csv[2:end]
        p = split(ln, ",")
        rows[(String(p[1]), parse(Int, p[3]))] = p
    end
    for fl in files
        d = JLD2.load(joinpath(cert_dir, fl))
        T = d["T"]; π̄ = d["pi"]; μ = d["mu"]; σ = d["sigma"]
        @test all(abs.(sum(T, dims=2) .- 1.0) .< 1e-10)   # valid stochastic matrix
        @test all(T .> 0.0)
        @test all(σ .> 0.0)
        @test maximum(abs.(vec(π̄' * T) .- π̄)) < 1e-8      # stationarity
        m, M = _folded_moments(μ, σ)
        ρ = _population_acf(T, π̄, m, M, length(d["rho_fit"]))
        @test maximum(abs.(ρ .- d["rho_fit"])) < 1e-10     # stored curve reproduces
        @test sum(abs2, ρ .- d["rho_target"]) ≈ d["sse"] rtol = 1e-8
        p = rows[(d["ticker"], d["K"])]
        @test parse(Float64, p[8]) ≈ d["sse"] atol = 5e-7
        @test parse(Float64, p[9]) ≈ d["near_mae"] atol = 5e-7
        # attainability bookkeeping: winner no worse than any start's initial
        # value, including the likelihood-seeded start
        @test all(d["sse"] <= dg.sse_init + 1e-12 for dg in d["diagnostics"])
        @test d["sse"] <= d["sse_mlseed"] + 1e-12
    end
end

@testset "Frontier certificates reload and verify" begin
    frt_dir = joinpath(@__DIR__, "..", "results", "hmm_acf_frontier")
    files = filter(f -> endswith(f, ".jld2"), readdir(frt_dir))
    @test length(files) == 31
    # published multistart likelihood fits: ll_best per (ticker, K=3) from the
    # spectral cross-ticker artifact (printed at 4 decimals)
    spectral_ll = Dict{String,Float64}()
    for ln in readlines(joinpath(@__DIR__, "..", "results", "diagnostics",
                                 "spectral_rank_cross_ticker_fits.csv"))[2:end]
        p = split(ln, ",")
        parse(Int, p[2]) == 3 && (spectral_ll[String(p[1])] = parse(Float64, p[7]))
    end
    for fl in files
        d = JLD2.load(joinpath(frt_dir, fl))
        ρ̂ = d["rho_target"]; xq = d["xq"]
        arms = filter(k -> startswith(k, "arm_"), collect(keys(d)))
        @test length(arms) == 9                     # 8 lambda arms + pure-marginal
        for a in arms
            w = d[a]
            T = w["T"]; π̄ = w["pi"]; μ = w["mu"]; σ = w["sigma"]
            @test all(abs.(sum(T, dims=2) .- 1.0) .< 1e-10)
            @test all(T .> 0.0)
            @test maximum(abs.(vec(π̄' * T) .- π̄)) < 1e-8
            m, M = _folded_moments(μ, σ)
            ρ = _population_acf(T, π̄, m, M, length(ρ̂))
            @test maximum(abs.(ρ .- w["rho_fit"])) < 1e-10
            @test mean(abs.(ρ[1:63] .- ρ̂[1:63])) ≈ w["near"] atol = 1e-10
            @test _marginal_cvm(π̄, μ, σ, xq) ≈ w["cvm"] rtol = 1e-8
            # the saved winning objective value is the weighted objective at the
            # saved model: J = ACF_SSE + lambda_s * CvM (CvM alone for the
            # pure-marginal arm), recomputed here from the stored metrics
            λs = w["lambda_scaled"]
            J = isfinite(λs) ? w["acf_sse"] + λs * w["cvm"] : w["cvm"]
            @test J ≈ w["objective_value"] rtol = 1e-8
            θ = _pack_acf_params(T, μ, σ)
            @test _joint_objective(θ, ρ̂, xq, λs, 3) ≈ w["objective_value"] rtol = 1e-6
        end
        # ml_multi comparator: the published converged multistart likelihood fit,
        # verified as a model and tied back to the spectral cross-ticker artifact
        mm = d["ml_multi"]
        T = mm["T"]; π̄ = mm["pi"]; μ = mm["mu"]; σ = mm["sigma"]
        @test all(abs.(sum(T, dims=2) .- 1.0) .< 1e-10)
        @test all(σ .> 0.0)
        @test maximum(abs.(vec(π̄' * T) .- π̄)) < 1e-8
        m, M = _folded_moments(μ, σ)
        ρ = _population_acf(T, π̄, m, M, length(ρ̂))
        @test maximum(abs.(ρ .- mm["rho_fit"])) < 1e-10
        @test mean(abs.(ρ[1:63] .- ρ̂[1:63])) ≈ mm["near"] atol = 1e-10
        @test _marginal_cvm(π̄, μ, σ, xq) ≈ mm["cvm"] rtol = 1e-8
        @test mm["ll_best"] ≈ spectral_ll[d["ticker"]] atol = 1e-3
    end
    # regret CSV is a pure function of the main CSV (rebuildable via --summary-only)
    diag_dir = joinpath(@__DIR__, "..", "results", "diagnostics")
    main = Dict{Tuple{String,String},Tuple{Float64,Float64}}()
    tickers = Set{String}()
    for ln in readlines(joinpath(diag_dir, "hmm_acf_frontier.csv"))[2:end]
        p = split(ln, ",")
        main[(String(p[1]), String(p[2]))] = (parse(Float64, p[3]), parse(Float64, p[6]))
        push!(tickers, String(p[1]))
    end
    sweep_arms = ["0", "0.1", "0.3", "1", "3", "10", "30", "100", "marginal"]
    reg_lines = readlines(joinpath(diag_dir, "hmm_acf_frontier_regret.csv"))
    @test reg_lines[1] == "ticker,arm,acf_regret,cvm_regret,max_regret,both_le_1p5"
    @test length(reg_lines) == 1 + length(sweep_arms) * length(tickers)
    for ln in reg_lines[2:end]
        p = split(ln, ",")
        t = String(p[1]); arm = String(p[2])
        best_acf = minimum(main[(t, a)][1] for a in sweep_arms)
        best_cvm = minimum(main[(t, a)][2] for a in sweep_arms)
        a_r = main[(t, arm)][1] / best_acf
        c_r = main[(t, arm)][2] / best_cvm
        @test parse(Float64, p[3]) ≈ a_r atol = 5e-5
        @test parse(Float64, p[4]) ≈ c_r atol = 5e-5
        @test parse(Float64, p[5]) ≈ max(a_r, c_r) atol = 5e-5
        @test parse(Int, p[6]) == ((a_r <= 1.5 && c_r <= 1.5) ? 1 : 0)
    end
end

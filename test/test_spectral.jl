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

include(joinpath(@__DIR__, "..", "runners", "spectral", "spectral_common.jl"))

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

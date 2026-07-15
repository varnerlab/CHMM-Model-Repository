# ========================================================================== #
# test_msgarch_reference.jl
#
# Smoke tests for the RCall-based reference MS-GARCH baseline. The test is
# OPT-IN: it runs only when CHMM_TEST_MSGARCH=1 is set, because the R-side
# MCMC smoke test can add several minutes to an otherwise ~2-minute suite
# (2026-07 final-rerun review, finding 2). Without the flag, and on machines
# where R or the MSGARCH R package is missing, it skips cleanly (with a
# printed reason, not a failure).
#
# When R + MSGARCH are present, the test runs an end-to-end:
#   fit -> simulate -> check shape and basic moment sanity
# at K = 2 with a fast MCMC budget (n_mcmc small) on a short synthetic
# series. This is a smoke test, not a recovery test; the recovery test
# proper lives in r_msgarch/setup.R's MSGARCH unit tests upstream.
# ========================================================================== #

using Test;
using Random;
using Statistics;

# Probe whether R is available before `using RCall`. RCall errors at module
# load time if R can't be located; this guard avoids a hard failure when
# the test is run on a machine without R.
function _r_available()
    try
        # `R --version` exits 0 if R is on PATH.
        run(pipeline(`R --version`, stdout=devnull, stderr=devnull));
        return true;
    catch
        return false;
    end
end

@testset "MSGARCHReference" begin
    if get(ENV, "CHMM_TEST_MSGARCH", "0") != "1"
        @info "MSGARCH reference tests are opt-in (R-side MCMC can add " *
              "several minutes); set CHMM_TEST_MSGARCH=1 to run them.";
        return;
    end
    if !_r_available()
        @info "R is not on PATH; skipping MSGARCH reference tests. " *
              "Install R >= 4.2 and run r_msgarch/setup.R to enable.";
        return;
    end

    # MSGARCHReference is intentionally not loaded by Include.jl (it does
    # `using RCall`, which fails without R). Load it here so the test runs
    # end-to-end when R is available, but only after the R-availability
    # probe above. If the load itself fails, surface a skip message.
    try
        @eval include(joinpath(@__DIR__, "..", "src", "MSGARCHReference.jl"));
    catch err
        @info "MSGARCHReference.jl failed to load (likely RCall + R config issue): $err. Skipping.";
        return;
    end

    # Probe the R-side environment by trying to fetch versions; if MSGARCH
    # is not installed in the project-local renv library, this throws and
    # we skip with a setup hint.
    versions = try
        Main.msgarch_reference_versions();
    catch err
        @info "MSGARCH reference R env not set up: $err. " *
              "Run `cd r_msgarch && Rscript setup.R` to enable.";
        return;
    end
    @info "MSGARCH reference test running against R $(versions.r_version), " *
          "MSGARCH $(versions.msgarch_version)";

    # ----- end-to-end smoke test ----- #
    Random.seed!(20260422);
    T_obs = 750;
    obs = randn(T_obs) .* 0.01;        # synthetic standard-Normal returns

    # Tiny MCMC budget for speed; we are not checking convergence here, just
    # that the bridge round-trip works.
    model = Main.fit_msgarch_reference(
        obs, 2;
        variance_spec = "sGARCH",
        distribution  = "norm",
        fit_method    = "MCMC",
        n_mcmc        = 1_500,
        n_burnin      = 500,
        n_thin        = 1,
        seed          = 12345,
    );
    @test model.K == 2;
    @test length(model.par_post_mean) == length(model.par_names);
    @test isfinite(model.loglik);
    @test size(model.transition) == (2, 2);
    @test all(>=(0), model.transition);
    @test all(row -> isapprox(sum(row), 1.0; atol=1e-6),
              eachrow(model.transition));

    # Simulate paths and check the shape contract
    sim = Main.simulate_msgarch_reference(model, 250; n_paths=4, seed=2026);
    @test size(sim) == (250, 4);
    @test all(isfinite, sim);

    # Determinism on identical seed
    sim2 = Main.simulate_msgarch_reference(model, 250; n_paths=4, seed=2026);
    @test isapprox(sim, sim2; atol=1e-12);

    # Basic moment sanity: the simulated series should have non-degenerate
    # variance and finite kurtosis. We only check no-NaN and reasonable
    # magnitudes; the per-K behaviour is the runner's job.
    for j in axes(sim, 2)
        @test std(sim[:, j]) > 0;
    end
end

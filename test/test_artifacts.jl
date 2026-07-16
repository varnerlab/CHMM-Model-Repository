# Artifact-level regression tests (third-review items 10/14). These validate
# invariants of the vendored paper-facing artifacts: forecast-origin alignment,
# breach-indexing consistency (a lookahead or off-by-one in the VaR filter
# would break the recomputed breach columns), and the paired design of the
# rolling-copula comparison. They read CSVs only; no model fitting.

const _ARTROOT = joinpath(@__DIR__, "..", "results");

function _read_csv_table(path)
    lines = readlines(path);
    hdr = split(lines[1], ",");
    rows = [split(l, ",") for l in lines[2:end]];
    return hdr, rows;
end

@testset "Paper-facing artifact invariants" begin

    @testset "Conditional VaR series: alignment and breach indexing" begin
        path = joinpath(_ARTROOT, "conditional_var_all_families", "conditional_var_series_chmmN_k3.csv");
        @test isfile(path)
        hdr, rows = _read_csv_table(path);
        ci = Dict(String(h) => i for (i, h) in enumerate(hdr));
        # The paper's VaR harness uses 573 forecasts (boundary return included).
        @test length(rows) == 573
        for r in rows
            R = parse(Float64, r[ci["R_oos"]]);
            v05 = parse(Float64, r[ci["var_05"]]);
            v01 = parse(Float64, r[ci["var_01"]]);
            @test v01 <= v05 < 0        # 1% quantile at least as extreme as 5%
            # Stored breach flags must equal the recomputed indicator: any
            # lookahead/off-by-one between forecast and realisation breaks this.
            @test parse(Int, r[ci["breach_05"]]) == Int(R < v05)
            @test parse(Int, r[ci["breach_01"]]) == Int(R < v01)
        end
    end

    @testset "MS-GARCH conditional VaR: same 573-forecast origin set" begin
        for K in (2, 3, 4)
            path = joinpath(_ARTROOT, "msgarch_conditional_var", "var_series_K$(K).csv");
            @test isfile(path)
            hdr, rows = _read_csv_table(path);
            ci = Dict(String(h) => i for (i, h) in enumerate(hdr));
            @test length(rows) == 573
            for r in rows
                R = parse(Float64, r[ci["R_oos"]]);
                v05 = parse(Float64, r[ci["var_05"]]);
                v01 = parse(Float64, r[ci["var_01"]]);
                @test v01 <= v05 < 0
                @test parse(Int, r[ci["breach_05"]]) == Int(R < v05)
                @test parse(Int, r[ci["breach_01"]]) == Int(R < v01)
            end
        end
    end

    @testset "Rolling copula: paired like-for-like design" begin
        path = abspath(joinpath(_ARTROOT, "..", "..", "CHMM-Paper-Repository",
                                "results", "robustness", "rolling_copula_oos.csv"));
        @test isfile(path)
        hdr, rows = _read_csv_table(path);
        ci = Dict(String(h) => i for (i, h) in enumerate(hdr));
        qrows = [r for r in rows if tryparse(Int, r[ci["qi"]]) !== nothing];
        # Every quarter carries BOTH arms' scores (static and rolling) against
        # the same block, and the blocks tile the OoS span with no gap.
        @test length(qrows) >= 9
        total_days = 0; prev_end = nothing;
        for r in qrows
            q_start = parse(Int, r[ci["q_start"]]);
            q_end = parse(Int, r[ci["q_end"]]);
            n_days = parse(Int, r[ci["n_days"]]);
            @test q_end - q_start + 1 == n_days
            prev_end === nothing || @test q_start == prev_end + 1
            prev_end = q_end;
            total_days += n_days;
            @test tryparse(Float64, r[ci["mae_static"]]) !== nothing
            @test tryparse(Float64, r[ci["mae_roll"]]) !== nothing
            d = parse(Float64, r[ci["diff"]]);
            @test isapprox(d, parse(Float64, r[ci["mae_static"]]) -
                              parse(Float64, r[ci["mae_roll"]]); atol=1e-4)
        end
        # Complete coverage of the 572-observation OoS span (final partial
        # block included by the stated convention).
        @test total_days == 572
    end

    @testset "CRPS per-day loss matrix covers all five displayed rows" begin
        path = joinpath(_ARTROOT, "crps_dm_kstar3", "per_day_losses.csv");
        @test isfile(path)
        hdr, rows = _read_csv_table(path);
        for col in ("chmm_n", "chmm_t_pen", "chmm_l", "chmm_ged", "chmm_t_shared_nu")
            @test col in String.(hdr)
        end
        @test length(rows) == 572
    end
end

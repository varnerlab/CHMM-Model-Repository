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

    @testset "Rolling copula: paired 2x2 family/refit design" begin
        path = abspath(joinpath(_ARTROOT, "..", "..", "CHMM-Paper-Repository",
                                "results", "robustness", "rolling_copula_oos.csv"));
        @test isfile(path)
        hdr, rows = _read_csv_table(path);
        ci = Dict(String(h) => i for (i, h) in enumerate(hdr));
        qrows = [r for r in rows if tryparse(Int, r[ci["qi"]]) !== nothing];
        # Every quarter carries ALL FOUR arms' scores (static/rolling x
        # Gaussian/Student-t) against the same block, and the blocks tile the
        # OoS span with no gap.
        @test length(qrows) >= 9
        total_days = 0; prev_end = nothing; n_complete = 0;
        for r in qrows
            q_start = parse(Int, r[ci["q_start"]]);
            q_end = parse(Int, r[ci["q_end"]]);
            n_days = parse(Int, r[ci["n_days"]]);
            @test q_end - q_start + 1 == n_days
            prev_end === nothing || @test q_start == prev_end + 1
            prev_end = q_end;
            total_days += n_days;
            for col in ("mae_static_t", "mae_roll_t", "mae_static_g", "mae_roll_g")
                @test tryparse(Float64, r[ci[col]]) !== nothing
            end
            # The in_inference flag marks exactly the complete 63-day blocks
            # (the partial final block is scored but excluded from paired
            # inference, by the stated convention).
            flag = parse(Int, r[ci["in_inference"]]);
            @test flag == Int(n_days == 63)
            n_complete += flag;
        end
        # Complete coverage of the 572-observation OoS span.
        @test total_days == 572
        @test n_complete == 9
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

    @testset "ML HSMM artifact: EM monotone, returned iterate evaluated, CSV shape" begin
        # (2026-07-16 sixth review, findings 4-5: the ML-HSMM comparator needs
        # live provenance and evidence that the corrected truncated-Pareto EM
        # actually improves the likelihood monotonically.)
        csv_path = joinpath(_ARTROOT, "hsmm_ml", "hsmm_ml_metrics.csv");
        @test isfile(csv_path)
        hdr, rows = _read_csv_table(csv_path);
        @test "final_inc_perobs" in String.(hdr)
        @test length(rows) == 2                       # K = 3 and K = 18
        for K in (3, 18)
            jld_path = joinpath(_ARTROOT, "hsmm_ml", "hsmm_ml_K$(K).jld2");
            @test isfile(jld_path)
            d = JLD2.load(jld_path);
            h = d["ll_history"];
            @test all(isfinite, h)
            @test all(diff(h) .> -1e-6)               # EM monotonicity
            # metrics CSV final_ll matches the last EVALUATED iterate
            row = rows[findfirst(r -> parse(Int, r[1]) == K, rows)];
            @test parse(Float64, row[findfirst(==("final_ll"), String.(hdr))]) ≈ h[end] atol = 1e-3
        end
    end
end

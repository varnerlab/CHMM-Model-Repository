# ========================================================================================= #
# run_ks_block_body_kstar.jl
#
# Block-aware OoS KS recalibration at the body headline (K*=3) and the K*=6 sensitivity
# reference, alongside the existing K=18 panel in tab:ks_block_body. Closes peer-review
# items R1 W5 / R2 W2 / R2 RE4 (the binding language asks for the K*=6 row alongside the
# K=18 row; with the K*=3 rebuild we add both).
#
# Protocol matches run_ks_block_bootstrap_oos.jl: stationary block bootstrap of R_oos
# at mean block length L = 20, B = 1000 replicates, 500 OoS-length simulated paths per
# generator, score each path's KS statistic against the OoS series under both
# asymptotic-iid and block-bootstrap critical values.
#
# Outputs:
#   results/ks_block_bootstrap/KS_Bootstrap_Body_Kstar.txt
#   ../CHMM-paper/results/robustness/ks_block_body_kstar.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, HypothesisTests, Printf
const SEED = 20260422;
Random.seed!(SEED);

const TICKER         = "SPY";
const RISK_FREE      = 0.0;
const DT             = 1/252;
const MAX_ITER       = 60;
const N_PATHS        = 500;
const B_BOOT         = 1000;
const L_BLOCK        = 20;
const LAMBDA         = 20.0;       # CHMM-t penalty rate at body operating point
const K_LIST         = [3, 6];

const OUT_DIR              = joinpath(_ROOT, "results", "ks_block_bootstrap");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80);
println("  Block-aware OoS KS at K*=3 (body headline) and K*=6 (sensitivity).");
println("  Seed $SEED, K_list = $K_LIST, N_paths = $N_PATHS, B_boot = $B_BOOT, L = $L_BLOCK");
println("="^80);

# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_is  = Vector{Float64}(log_growth_matrix(train_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset,   "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_is = length(R_is); n_oos = length(R_oos);
@printf("  T_IS = %d   T_OoS = %d\n", n_is, n_oos);

# --------------------------------------------------------------------------------------- #
function stationary_block_bootstrap(x::Vector{Float64}, T::Int, L::Int)::Vector{Float64}
    n = length(x); out = Vector{Float64}(undef, T);
    p = 1.0 / L; t = 1;
    while t <= T
        i = rand(1:n); block_len = 1;
        while rand() > p; block_len += 1; end
        for _ in 1:block_len
            if t > T; break; end
            out[t] = x[((i - 1) % n) + 1];
            i += 1; t += 1;
        end
    end
    return out;
end

ks_statistic(a, b) = ApproximateTwoSampleKSTest(a, b).δ;

println("\n[block-boot] OoS-anchored 95% KS critical value at L = $L_BLOCK (B = $B_BOOT)...");
Random.seed!(SEED + 1000 + L_BLOCK);
null_stats = Vector{Float64}(undef, B_BOOT);
for b in 1:B_BOOT
    boot = stationary_block_bootstrap(R_oos, n_oos, L_BLOCK);
    null_stats[b] = ks_statistic(R_oos, boot);
end
crit_block = quantile(null_stats, 0.95);
crit_asymp = 1.36 * sqrt(2 / n_oos);
@printf("  L = %d : 95%% block-bootstrap critical value = %.4f\n", L_BLOCK, crit_block);
@printf("  asymptotic 95%% critical value             = %.4f\n", crit_asymp);

# --------------------------------------------------------------------------------------- #
function _fit_family(family::Symbol, K::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    elseif family == :student_t_pen
        return build(MyStudentTHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER,
             ν_shrink_rate=LAMBDA));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    elseif family == :ged
        return build(MyGEDHiddenMarkovModel,
            (observations=R_is, number_of_states=K, max_iter=MAX_ITER));
    else
        error("Unknown family: $family");
    end
end

const FAMILIES = [
    (:gaussian,      "CHMM-N",            11),
    (:student_t_pen, "CHMM-t pen. (λ=20)", 12),
    (:laplace,       "CHMM-L",            13),
    (:ged,           "CHMM-GED",          14),
];

function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    π_stat = max.(π_stat, 1e-12); π_stat ./= sum(π_stat);
    return Categorical(π_stat);
end

function _simulate_paths(model, start_dist, T::Int, n_paths::Int; seed::Int)
    Random.seed!(seed);
    paths = Matrix{Float64}(undef, T, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist); st = model(s0, T);
        for j in 1:T; paths[j, i] = rand(model.emission[st[j]]); end
    end
    return paths;
end

function score_generator(name::String, sim::Matrix{Float64}, R_obs::Vector{Float64})
    n_p = size(sim, 2);
    ks  = Vector{Float64}(undef, n_p);
    pv  = Vector{Float64}(undef, n_p);
    for p in 1:n_p
        test = ApproximateTwoSampleKSTest(R_obs, sim[:, p]);
        ks[p] = test.δ; pv[p] = pvalue(test);
    end
    asymp_pass = mean(pv .>= 0.05);
    block_pass = mean(ks .< crit_block);
    return (
        name=name,
        asymp_pass_pct=round(100 * asymp_pass, digits=1),
        block_pass_pct=round(100 * block_pass, digits=1),
        ks_q50=round(median(ks), digits=4),
        mean_pv=round(mean(pv), digits=3),
    );
end

# --------------------------------------------------------------------------------------- #
println("\n[generators] fitting + scoring at K* = $(K_LIST)...");
panel = NamedTuple[];
for K in K_LIST
    println("\n  K = $K");
    for (family, label, fam_offset) in FAMILIES
        Random.seed!(SEED + 100 * K + fam_offset);
        m = _fit_family(family, K);
        start = _stationary(m, K);
        sim = _simulate_paths(m, start, n_oos, N_PATHS;
                              seed=SEED + 1000 * K + fam_offset);
        full_label = "$label (K* = $K)";
        r = score_generator(full_label, sim, R_oos);
        push!(panel, (K=K, family=String(family), label=full_label,
                      asymp_pct=r.asymp_pass_pct, block_pct=r.block_pass_pct,
                      ks_q50=r.ks_q50, mean_pv=r.mean_pv));
        @printf("    %-30s  asymp = %5.1f%%  block = %5.1f%%  ks_q50 = %.4f  mean_pv = %.3f\n",
                full_label, r.asymp_pass_pct, r.block_pass_pct, r.ks_q50, r.mean_pv);
    end
end

# --------------------------------------------------------------------------------------- #
println("\n[output] writing artefacts...");
out_path = joinpath(OUT_DIR, "KS_Bootstrap_Body_Kstar.txt");
open(out_path, "w") do io
    println(io, "="^110);
    println(io, "Block-aware OoS KS recalibration at the body headline K*=3 and K*=6 sensitivity reference.");
    println(io, "="^110);
    println(io);
    @printf(io, "Setup     : SPY OoS (%d obs); %d OoS-length simulated paths per generator under seed %d.\n",
            n_oos, N_PATHS, SEED);
    @printf(io, "Block boot: stationary block bootstrap of R_oos at mean block length L = %d, B = %d.\n",
            L_BLOCK, B_BOOT);
    @printf(io, "OoS-anchored 95%% block-bootstrap KS critical value : %.4f\n", crit_block);
    @printf(io, "Asymptotic 95%% two-sample KS critical value         : %.4f  (1.36 * sqrt(2 / n_oos))\n",
            crit_asymp);
    println(io);
    @printf(io, "%-32s  %-9s  %-10s  %-9s  %-9s\n",
            "Model", "asymp %", "block L=20%", "KS q50", "mean p-val");
    println(io, "-"^110);
    for r in panel
        @printf(io, "%-32s  %-9.1f  %-10.1f  %-9.4f  %-9.3f\n",
                r.label, r.asymp_pct, r.block_pct, r.ks_q50, r.mean_pv);
    end
    println(io);
    println(io, "Body Table tab:ks_block_body row order (alongside existing K=18 rows):");
    println(io, "  bootstrap, GARCH(1,1) [existing], plus the K*=3 four-emission block (headline),");
    println(io, "  plus the K*=6 four-emission block (sensitivity), plus the existing K=18 four-emission block.");
end

csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "ks_block_body_kstar.csv");
open(csv_path, "w") do io
    println(io, "K,family,label,asymp_pct,block_L20_pct,ks_q50,mean_pv");
    for r in panel
        @printf(io, "%d,%s,\"%s\",%.1f,%.1f,%.4f,%.3f\n",
                r.K, r.family, r.label, r.asymp_pct, r.block_pct, r.ks_q50, r.mean_pv);
    end
end

println("\n" * "="^80);
println("  K*=3 / K*=6 block-aware KS row computation complete.");
@printf("  Human-readable: %s\n", out_path);
@printf("  Paper CSV     : %s\n", csv_path);
println("="^80);

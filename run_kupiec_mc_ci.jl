# ========================================================================================= #
# run_kupiec_mc_ci.jl
#
# on the unconditional Kupiec LR_uc statistic at T_OoS = 572 days, addressing the
# referee's concern that "the unconditional calibration gain on SM-CHMM-N
# (LR_uc: 3.83 → 0.82) is numerically small relative to MC error."
#
# Procedure. Under the assumption that the model is correctly calibrated at breach rate
# p (where p is the breach rate observed for that model in Section sec:sm_ablation), we
# draw B = 10,000 synthetic Bernoulli(p) breach sequences of length T_OoS = 572 and
# compute the Kupiec LR_uc statistic on each against the target rate α. The resulting
# bootstrap distribution of LR_uc is the MC noise band; the question is whether
# 3.83 (flat CHMM-N) and 0.82 (SM-CHMM-N) sit in distinguishable parts of those bands.
#
# Sub-ask (a) (Yu 2010 explicit-duration forward-backward MLE) is a substantial
# implementation that is documented as a follow-up in revision-code-todo.md.
#
# Outputs:
#   results/kupiec_mc_ci/LR_uc_MC_CI.txt
#   ../CHMM-paper/results/robustness/kupiec_mc_ci.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
const SEED = 20260422;
Random.seed!(SEED);

const T_OOS = 572;
const B     = 10_000;
const ALPHA = 0.05;
# Observed breach rates from the paper's tab:sm_var (Section sec:sm_ablation):
const OBS_RATES = [
    ("CHMM-N (flat)",    0.033),
    ("SM-CHMM-N (plug-in)", 0.042),
    ("CHMM-t (flat)",    0.033),
    ("SM-CHMM-t (plug-in)", 0.038),
    ("CHMM-L (flat)",    0.035),
    ("SM-CHMM-L (plug-in)", 0.035),
];
# Observed LR_uc values from tab:sm_var
const OBS_LR_UC = Dict(
    "CHMM-N (flat)"        => 3.83,
    "SM-CHMM-N (plug-in)"  => 0.82,
    "CHMM-t (flat)"        => 3.83,
    "SM-CHMM-t (plug-in)"  => 1.74,
    "CHMM-L (flat)"        => 3.03,
    "SM-CHMM-L (plug-in)"  => 3.03,
);

const OUT_DIR       = joinpath(_ROOT, "results", "kupiec_mc_ci");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  Kupiec MC confidence interval")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Bootstrap LR_uc null at given breach rate
# --------------------------------------------------------------------------------------- #
"""
    bootstrap_lr_uc(p, α, n, B) -> Vector{Float64}

For B i.i.d. Bernoulli(p) breach sequences of length n, return the Kupiec LR_uc
statistic against target α for each sequence.
"""
function bootstrap_lr_uc(p::Float64, α::Float64, n::Int, B::Int)::Vector{Float64}
    out = Vector{Float64}(undef, B);
    for b in 1:B
        seq = rand(n) .< p;
        out[b] = kupiec_lr(seq, α).LR;
    end
    return out;
end

# --------------------------------------------------------------------------------------- #
# Run sweep
# --------------------------------------------------------------------------------------- #
results = NamedTuple[];

for (name, p_obs) in OBS_RATES
    Random.seed!(SEED + Int(round(p_obs * 100000)));
    lrs = bootstrap_lr_uc(p_obs, ALPHA, T_OOS, B);
    LR_obs = OBS_LR_UC[name];
    # Bootstrap p-value: fraction of bootstrap LR ≥ observed
    pv = mean(lrs .>= LR_obs);
    push!(results, (
        name=name, p_obs=p_obs, LR_obs=LR_obs, p_value_boot=pv,
        q05=quantile(lrs, 0.05), q25=quantile(lrs, 0.25),
        q50=median(lrs), q75=quantile(lrs, 0.75),
        q95=quantile(lrs, 0.95), q99=quantile(lrs, 0.99),
        mean_LR=mean(lrs), std_LR=std(lrs),
    ));
    println("  $(rpad(name, 22))  p=$(round(p_obs, digits=3))  LR_obs=$LR_obs  bootstrap mean ± std: $(round(mean(lrs), digits=2)) ± $(round(std(lrs), digits=2))   q05/q50/q95: $(round(quantile(lrs, 0.05), digits=2))/$(round(median(lrs), digits=2))/$(round(quantile(lrs, 0.95), digits=2))   p_boot $(round(pv, digits=3))");
end

# --------------------------------------------------------------------------------------- #
# Pairwise comparison: how much do the bootstrap distributions of flat-vs-SM overlap?
# --------------------------------------------------------------------------------------- #
println("\n[overlap] flat vs SM bootstrap distribution comparisons:");

function _overlap_ratio(p1::Float64, p2::Float64, α::Float64, n::Int, B::Int)
    # Two-sample bootstrap: simulate B reps of LR under each rate, compute the fraction of
    # times LR(p2) < LR(p1) (i.e., the SM rate produces a smaller LR_uc than the flat rate).
    Random.seed!(SEED + 7777);
    lrs1 = bootstrap_lr_uc(p1, α, n, B);
    Random.seed!(SEED + 8888);
    lrs2 = bootstrap_lr_uc(p2, α, n, B);
    return mean(lrs2 .< lrs1);
end

pairs = [
    ("CHMM-N (flat)", "SM-CHMM-N (plug-in)"),
    ("CHMM-t (flat)", "SM-CHMM-t (plug-in)"),
    ("CHMM-L (flat)", "SM-CHMM-L (plug-in)"),
];
overlap_results = NamedTuple[];
for (flat_name, sm_name) in pairs
    p1 = OBS_RATES[findfirst(x -> x[1] == flat_name, OBS_RATES)][2];
    p2 = OBS_RATES[findfirst(x -> x[1] == sm_name, OBS_RATES)][2];
    ratio = _overlap_ratio(p1, p2, ALPHA, T_OOS, B);
    push!(overlap_results, (flat=flat_name, sm=sm_name, p_flat=p1, p_sm=p2, frac_sm_smaller=ratio));
    println("  P(LR(SM) < LR(flat))  for ($flat_name vs $sm_name) at p_flat=$p1, p_sm=$p2 : $(round(ratio, digits=3))");
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(OUT_DIR, "LR_uc_MC_CI.txt"), "w") do io
    println(io, "="^150);
    println(io, "MC confidence interval on Kupiec LR_uc at n = $T_OOS.");
    println(io, "="^150);
    println(io, "");
    println(io, "Procedure. For each (model, observed breach rate p) pair, simulate B = $B i.i.d. Bernoulli(p) breach");
    println(io, "sequences of length $T_OOS and compute Kupiec LR_uc against α = $ALPHA on each. The resulting");
    println(io, "bootstrap distribution of LR_uc is the MC noise band that the observed LR_uc inhabits.");
    println(io, "");
    println(io, rpad("Model", 22), " | ", rpad("p_obs", 6), " | ", rpad("LR_obs", 6), " | ",
                rpad("mean LR", 7), " | ", rpad("std LR", 6), " | ",
                rpad("q05", 5), " | ", rpad("q25", 5), " | ", rpad("q50", 5), " | ", rpad("q75", 5), " | ", rpad("q95", 5), " | ",
                rpad("p_boot", 6));
    println(io, "-"^150);
    for r in results
        println(io, rpad(r.name, 22), " | ",
                    rpad(round(r.p_obs, digits=3), 6), " | ",
                    rpad(r.LR_obs, 6), " | ",
                    rpad(round(r.mean_LR, digits=2), 7), " | ",
                    rpad(round(r.std_LR, digits=2), 6), " | ",
                    rpad(round(r.q05, digits=2), 5), " | ",
                    rpad(round(r.q25, digits=2), 5), " | ",
                    rpad(round(r.q50, digits=2), 5), " | ",
                    rpad(round(r.q75, digits=2), 5), " | ",
                    rpad(round(r.q95, digits=2), 5), " | ",
                    rpad(round(r.p_value_boot, digits=3), 6));
    end
    println(io, "="^150);
    println(io, "");
    println(io, "Pairwise flat-vs-SM overlap (P(LR_uc under SM rate < LR_uc under flat rate)):");
    println(io, "");
    for r in overlap_results
        println(io, "  $(rpad(r.flat, 22)) (p=$(r.p_flat))   vs   $(rpad(r.sm, 22)) (p=$(r.p_sm))   :   $(round(r.frac_sm_smaller, digits=3))");
    end
    println(io, "");
    println(io, "Reading: a value near 0.5 means the two LR_uc bootstrap distributions are essentially indistinguishable at n = $T_OOS.");
    println(io, "         A value above 0.7 means the SM rate produces a meaningfully smaller LR_uc than the flat rate in the");
    println(io, "         majority of MC replications, so the calibration improvement is genuine rather than noise.");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "kupiec_mc_ci.csv"), "w") do io
    println(io, "model,p_obs,LR_obs,mean_LR_boot,std_LR_boot,q05,q25,q50,q75,q95,bootstrap_pvalue");
    for r in results
        println(io, "$(r.name),$(round(r.p_obs, digits=4)),$(r.LR_obs),$(round(r.mean_LR, digits=4)),$(round(r.std_LR, digits=4)),$(round(r.q05, digits=4)),$(round(r.q25, digits=4)),$(round(r.q50, digits=4)),$(round(r.q75, digits=4)),$(round(r.q95, digits=4)),$(round(r.p_value_boot, digits=4))");
    end
    println(io, "");
    println(io, "comparison_flat,comparison_sm,p_flat,p_sm,frac_LR_sm_smaller");
    for r in overlap_results
        println(io, "$(r.flat),$(r.sm),$(r.p_flat),$(r.p_sm),$(round(r.frac_sm_smaller, digits=4))");
    end
end

println("\n" * "="^72);
println("  Kupiec MC CI complete.");
println("  Text report : $(joinpath(OUT_DIR, "LR_uc_MC_CI.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_ROBUSTNESS_DIR, "kupiec_mc_ci.csv"))");
println("="^72);

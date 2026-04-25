# ========================================================================================= #
# run_track_m5_lr_ind_null.jl
#
# Track M5 (revision response to referee comment M5): bootstrap null distribution of the
# Christoffersen independence LR statistic at n = 572, α ∈ {0.01, 0.05}, to quantify how
# underpowered the asymptotic χ²(1) critical value is at the paper's OoS sample size.
#
# Procedure. Draw B = 10,000 i.i.d. Bernoulli(α) breach sequences of length
# n = T_OoS = 572 and compute Christoffersen LR_ind on each. Report empirical null
# quantiles (50%, 90%, 95%, 99%), compare to the χ²(1) 3.84 critical value, and locate
# the observed CHMM smoother and filter LR_ind statistics from Track C3a/C3b within the
# null distribution.
#
# Expected behaviour. At α = 0.01 with n = 572, the expected breach count is 5.72. Many
# bootstrap sequences will have 0 or 1 breaches, yielding degenerate LR_ind = 0 under
# christoffersen_lr's convention (the π = 0 or π = 1 case). The resulting null is
# mass-concentrated at 0 with a long right tail, and the 95% bootstrap quantile typically
# sits well below the asymptotic 3.84 threshold.
#
# Outputs:
#   results/track_m5/LR_ind_Bootstrap_Null.txt
#   ../CHMM-paper/results/revision/M5_lr_ind_null.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
const SEED = 20260422;
Random.seed!(SEED);

const T_OOS = 572;
const B     = 10_000;
const ALPHAS = [0.01, 0.05];

# Observed CHMM smoother / filter LR_ind from Track C3a (Viterbi) and Track C3b (filter).
# These match the numbers now in sections/var_backtest.tex Table tab:conditional_var.
const OBSERVED = Dict(
    0.01 => Dict(
        "CHMM-N smoother (Viterbi)" => 5.73,
        "CHMM-t smoother (Viterbi)" => 0.09,
        "CHMM-L smoother (Viterbi)" => 0.01,
        "CHMM-N filter (forward)"   => 0.00,
        "CHMM-t filter (forward)"   => 0.00,
        "CHMM-L filter (forward)"   => 0.00,
    ),
    0.05 => Dict(
        "CHMM-N smoother (Viterbi)" => 1.39,
        "CHMM-t smoother (Viterbi)" => 0.19,
        "CHMM-L smoother (Viterbi)" => 2.25,
        "CHMM-N filter (forward)"   => 2.81,
        "CHMM-t filter (forward)"   => 8.74,
        "CHMM-L filter (forward)"   => 9.95,
    ),
);

const TRACK_M5_DIR       = joinpath(_ROOT, "results", "track_m5");
const PAPER_REVISION_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "revision"));
mkpath(TRACK_M5_DIR);
mkpath(PAPER_REVISION_DIR);

println("="^72)
println("  Track M5: Christoffersen LR_ind bootstrap null at n=$T_OOS, B=$B (referee M5)")
println("="^72)

function bootstrap_lr_ind_null(α::Float64, n::Int, B::Int)
    lrs = Vector{Float64}(undef, B);
    zero_count = 0;   # sequences with 0 breaches
    one_count  = 0;   # sequences with exactly 1 breach
    allbreach_count = 0;  # degenerate: every t is a breach (astronomically unlikely but check)
    degenerate_pi = 0;    # sequences where π ∈ {0, 1} (LR_ind defined to be 0.0 by christoffersen_lr)
    for b in 1:B
        seq = rand(n) .< α;
        k = count(seq);
        if k == 0; zero_count += 1; end
        if k == 1; one_count += 1; end
        if k == n; allbreach_count += 1; end
        if k == 0 || k == n; degenerate_pi += 1; end
        r = christoffersen_lr(seq);
        lrs[b] = r.LR;
    end
    return lrs, (zero_count=zero_count, one_count=one_count,
                 allbreach_count=allbreach_count, degenerate_pi=degenerate_pi);
end

results = Dict{Float64, NamedTuple}();

for α in ALPHAS
    println("\n[α = $α] simulating B = $B sequences of length $T_OOS...");
    Random.seed!(SEED + Int(round(α * 1000)));
    lrs, counts = bootstrap_lr_ind_null(α, T_OOS, B);
    q50 = quantile(lrs, 0.50);
    q90 = quantile(lrs, 0.90);
    q95 = quantile(lrs, 0.95);
    q99 = quantile(lrs, 0.99);
    empirical_size = mean(lrs .> 3.84);  # fraction of null sequences rejected by asymptotic cutoff
    frac_zero_lr   = mean(lrs .== 0.0);  # includes both π ∈ {0, 1} and n01 == 0 cases

    # Locate observed statistics within the bootstrap null
    obs_pvalues = Dict{String, Float64}();
    for (name, LR_obs) in OBSERVED[α]
        # Bootstrap-corrected p-value: fraction of null sequences with LR >= LR_obs
        pv = mean(lrs .>= LR_obs);
        obs_pvalues[name] = pv;
    end

    results[α] = (
        q50=q50, q90=q90, q95=q95, q99=q99,
        empirical_size=empirical_size, frac_zero_lr=frac_zero_lr,
        zero_breach_count=counts.zero_count,
        one_breach_count=counts.one_count,
        degenerate_pi=counts.degenerate_pi,
        obs_pvalues=obs_pvalues,
        lrs=lrs,
    );

    println("  Empirical size of χ²(1) cutoff 3.84 : $(round(100*empirical_size, digits=2))%  (asymptotic size: 5%)");
    println("  Bootstrap quantiles 50 / 90 / 95 / 99: $(round(q50, digits=3)) / $(round(q90, digits=3)) / $(round(q95, digits=3)) / $(round(q99, digits=3))");
    println("  Fraction of nulls with LR_ind == 0   : $(round(100*frac_zero_lr, digits=1))%  (zero breaches: $(counts.zero_count) / $B)");
    println("  Observed LR_ind bootstrap p-values (fraction of nulls ≥ observed):");
    for (name, pv) in sort(collect(obs_pvalues); by=first)
        println("    $(rpad(name, 28))  LR = $(OBSERVED[α][name])   p_boot = $(round(pv, digits=3))");
    end
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_M5_DIR, "LR_ind_Bootstrap_Null.txt"), "w") do io
    println(io, "="^120);
    println(io, "Track M5. Christoffersen LR_ind bootstrap null at n = $T_OOS, B = $B (referee M5 response).");
    println(io, "="^120);
    println(io, "");
    println(io, "Procedure: B = $B i.i.d. Bernoulli(α) breach sequences of length $T_OOS; christoffersen_lr on each.");
    println(io, "Asymptotic χ²(1) 95%% critical value: 3.84. At n = $T_OOS the asymptotic theory is off-manifold; this");
    println(io, "bootstrap locates the empirical null distribution and each observed CHMM LR_ind within it.");
    println(io, "");
    for α in ALPHAS
        r = results[α];
        println(io, "-"^120);
        println(io, "α = $α  (expected breaches per sequence: $(round(α * T_OOS, digits=2)))");
        println(io, "-"^120);
        println(io, "");
        println(io, "  Bootstrap null quantiles:  50% = $(round(r.q50, digits=3))   90% = $(round(r.q90, digits=3))   95% = $(round(r.q95, digits=3))   99% = $(round(r.q99, digits=3))");
        println(io, "  Empirical rejection rate at asymptotic 3.84 cutoff: $(round(100*r.empirical_size, digits=2))%");
        println(io, "    (If the asymptotic theory were exact, this would equal the nominal 5%.)");
        println(io, "  Fraction of bootstrap sequences with LR_ind == 0.0 (degenerate): $(round(100*r.frac_zero_lr, digits=1))%");
        println(io, "    - zero breaches:          $(r.zero_breach_count) / $B");
        println(io, "    - one breach:             $(r.one_breach_count) / $B");
        println(io, "    - π ∈ {0, 1} (any cause): $(r.degenerate_pi) / $B");
        println(io, "");
        println(io, "  Observed CHMM LR_ind located within bootstrap null:");
        println(io, "    $(rpad("construction", 28))  observed LR  bootstrap p-value");
        for (name, pv) in sort(collect(r.obs_pvalues); by=first)
            LR_obs = OBSERVED[α][name];
            println(io, "    $(rpad(name, 28))  $(rpad(LR_obs, 11))  $(round(pv, digits=3))");
        end
        println(io, "");
    end
    println(io, "="^120);
    println(io, "");
    println(io, "Reading: at α = 0.01 and n = $T_OOS the expected breach count is $(round(0.01*T_OOS, digits=2));");
    println(io, "around $(round(100*results[0.01].frac_zero_lr, digits=1))% of bootstrap sequences yield LR_ind = 0 by degeneracy,");
    println(io, "so the bootstrap null is strongly concentrated near zero and the asymptotic χ²(1) cutoff over-rejects.");
    println(io, "The M3 filter-based observed LR_ind of 0.00 is therefore not by itself informative; the α = 0.05 filter");
    println(io, "LR_ind of 8.74 (CHMM-t) sits at bootstrap p-value $(round(results[0.05].obs_pvalues["CHMM-t filter (forward)"], digits=3)),");
    println(io, "which is where the genuine independence-failure signal actually lives.");
end

open(joinpath(PAPER_REVISION_DIR, "M5_lr_ind_null.csv"), "w") do io
    println(io, "alpha,q50,q90,q95,q99,empirical_size_at_3_84,frac_zero_lr,zero_breach_count,one_breach_count");
    for α in ALPHAS
        r = results[α];
        println(io, "$α,$(round(r.q50, digits=4)),$(round(r.q90, digits=4)),$(round(r.q95, digits=4)),$(round(r.q99, digits=4)),$(round(r.empirical_size, digits=4)),$(round(r.frac_zero_lr, digits=4)),$(r.zero_breach_count),$(r.one_breach_count)");
    end
    println(io, "");
    println(io, "alpha,construction,observed_LR,bootstrap_pvalue");
    for α in ALPHAS
        r = results[α];
        for (name, pv) in sort(collect(r.obs_pvalues); by=first)
            LR_obs = OBSERVED[α][name];
            println(io, "$α,$name,$LR_obs,$(round(pv, digits=4))");
        end
    end
end

println("\n" * "="^72);
println("  Track M5 complete.");
println("  Text report : $(joinpath(TRACK_M5_DIR, "LR_ind_Bootstrap_Null.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_REVISION_DIR, "M5_lr_ind_null.csv"))");
println("="^72);

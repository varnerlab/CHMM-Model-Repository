# ========================================================================================= #
# run_kdisc13_centroid_ablation.jl
#
# Track Minor 6 (revision response): the existing Bin-T NJ row in Table 3 changes both the
# bin count (K_disc = 90 → 13) AND the emission family (centroid → bin-conditional
# Student-t). To isolate the bin-count effect from the emission-family effect the referee
# asks for a parallel K_disc = 13 row with the standard centroid emissions (no Student-t).
#
# This script fits exactly that: a discrete HMM at K_disc = 13 with centroid emissions
# and no jumps, then scores it on the same IS/OoS distributional panel used in
# tab:model_comparison.
#
# Output:
#   results/track_minor6/Kdisc13_Centroid.txt
#   ../CHMM-paper/results/robustness/kdisc13_centroid_ablation.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 500;
const K_DISC    = 13;

const TRACK_DIR        = joinpath(_ROOT, "results", "track_minor6");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(TRACK_DIR); mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  Track Minor 6: K_disc = $K_DISC centroid Discrete HMM (referee Minor 6)")
println("="^72)

println("\n[data] Loading SPY IS + OoS...");
train = MyPortfolioDataSet() |> x -> x["dataset"];
oos = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train["AAPL"]);
ds = Dict{String,DataFrame}();
for (t, data) in train; if nrow(data) == max_days; ds[t] = data; end; end
all_tickers = keys(ds) |> collect |> sort;
all_R = log_growth_matrix(ds, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);
R_oos = Vector{Float64}(log_growth_matrix(oos, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

# --------------------------------------------------------------------------------------- #
# Discrete HMM at K_disc = 13, centroid emissions, no jumps (mirrors run_track_a_metrics.jl
# §"Discrete HMM (NJ + WJ)" but with K_disc = 13 rather than 90)
# --------------------------------------------------------------------------------------- #
println("\n[fit] Discrete NJ at K_disc = $K_DISC, centroid emissions...");
qprobs = range(0.0, 1.0, length=K_DISC + 1) |> collect;
bin_edges = quantile(R_is, qprobs);
function _assign_bin(x, edges)
    Kb = length(edges) - 1;
    for k in 1:Kb; if k == Kb || x < edges[k+1]; return k; end; end
    return Kb;
end;
bin_idx = [_assign_bin(r, bin_edges) for r in R_is];
bin_centers = zeros(K_DISC);
for k in 1:K_DISC
    mk = findall(==(k), bin_idx);
    bin_centers[k] = isempty(mk) ? 0.0 : mean(R_is[mk]);
end
T_counts = zeros(K_DISC, K_DISC);
for t in 2:length(bin_idx); T_counts[bin_idx[t-1], bin_idx[t]] += 1.0; end
T_disc = copy(T_counts);
for i in 1:K_DISC
    rs = sum(T_disc[i, :]);
    T_disc[i, :] = rs > 0 ? T_disc[i, :] ./ rs : fill(1.0/K_DISC, K_DISC);
end
E_disc = zeros(K_DISC, K_DISC);
floor_ = 0.001;
for i in 1:K_DISC
    for j in 1:K_DISC; E_disc[i, j] = (i == j) ? 1.0 : floor_; end
    E_disc[i, :] ./= sum(E_disc[i, :]);
end
π_disc = (T_disc^1000)[1, :];
π_disc ./= sum(π_disc);
sd_disc = Categorical(π_disc);

Random.seed!(SEED + 4);
model_nj = build(MyHiddenMarkovModel, (states=collect(1:K_DISC), T=T_disc, E=E_disc));

# --------------------------------------------------------------------------------------- #
# Simulate
# --------------------------------------------------------------------------------------- #
function _simulate_discrete(model, sd, centers, n_steps, n_paths)
    paths = Matrix{Float64}(undef, n_steps, n_paths);
    for p in 1:n_paths
        s0 = rand(sd);
        states = model(s0, n_steps);
        for t in 1:n_steps; paths[t, p] = centers[states[t]]; end
    end
    return paths;
end

println("\n[sim] simulating $N_PATHS IS + OoS paths...");
Random.seed!(SEED + 100);
sim_is  = _simulate_discrete(model_nj, sd_disc, bin_centers, n_is,  N_PATHS);
Random.seed!(SEED + 101);
sim_oos = _simulate_discrete(model_nj, sd_disc, bin_centers, n_oos, N_PATHS);

# --------------------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------------------- #
function ks_pass_rate(R_ref, sim; α::Float64=0.05)
    n_p = size(sim, 2); n_pass = 0;
    for p in 1:n_p
        pv = pvalue(ApproximateTwoSampleKSTest(R_ref, sim[:, p]));
        if pv >= α; n_pass += 1; end
    end
    return n_pass / n_p;
end
function ad_pass_rate(R_ref, sim; α::Float64=0.05)
    n_p = size(sim, 2); n_pass = 0;
    for p in 1:n_p
        try
            pv = pvalue(KSampleADTest(R_ref, sim[:, p]));
            if pv >= α; n_pass += 1; end
        catch; end
    end
    return n_pass / n_p;
end
function acf_mae(R_obs, sim; max_lag::Int=252)
    obs = autocor(abs.(R_obs), 1:max_lag);
    n_p = size(sim, 2);
    sm = zeros(max_lag);
    for p in 1:n_p; sm .+= autocor(abs.(sim[:, p]), 1:max_lag); end
    sm ./= n_p;
    return mean(abs.(obs .- sm));
end

is_ks  = ks_pass_rate(R_is,  sim_is);
oos_ks = ks_pass_rate(R_oos, sim_oos);
is_ad  = ad_pass_rate(R_is,  sim_is);
oos_ad = ad_pass_rate(R_oos, sim_oos);
sim_kurt = mean([kurtosis(sim_is[:, p]) for p in 1:N_PATHS]);
acf = acf_mae(R_is, sim_is);

println("\n[score] K_disc=$K_DISC centroid NJ:");
println("  IS KS  $(round(100*is_ks, digits=1))%   OoS KS  $(round(100*oos_ks, digits=1))%");
println("  IS AD  $(round(100*is_ad, digits=1))%   OoS AD  $(round(100*oos_ad, digits=1))%");
println("  sim kurt $(round(sim_kurt, digits=3))    ACF-MAE $(round(acf, digits=4))");

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_DIR, "Kdisc13_Centroid.txt"), "w") do io
    println(io, "="^120);
    println(io, "Track Minor 6. K_disc = $K_DISC centroid Discrete HMM (referee Minor 6 response).");
    println(io, "="^120);
    println(io, "");
    println(io, "Setup     : Discrete HMM, $K_DISC quantile bins, centroid emissions, no jumps. SPY IS = $n_is, OoS = $n_oos, N_paths = $N_PATHS.");
    println(io, "Reference : Bin-T NJ in tab:model_comparison uses K_disc = 13 with bin-conditional Student-t emissions; this row uses centroid emissions to isolate the bin-count effect.");
    println(io, "");
    println(io, rpad("metric", 18), "  value");
    println(io, "-"^36);
    println(io, rpad("IS KS pass rate %", 18), "  $(round(100*is_ks, digits=1))");
    println(io, rpad("OoS KS pass rate %", 18), "  $(round(100*oos_ks, digits=1))");
    println(io, rpad("IS AD pass rate %", 18), "  $(round(100*is_ad, digits=1))");
    println(io, rpad("OoS AD pass rate %", 18), "  $(round(100*oos_ad, digits=1))");
    println(io, rpad("sim excess kurt", 18), "  $(round(sim_kurt, digits=3))   (observed 7.686)");
    println(io, rpad("ACF-MAE", 18), "  $(round(acf, digits=4))");
    println(io, "="^120);
    println(io, "");
    println(io, "Reading: comparing this row to Bin-T NJ in tab:model_comparison isolates the emission-family effect");
    println(io, "(centroid vs bin-conditional Student-t at the same bin count K_disc = 13). The kurtosis gap from this row");
    println(io, "(near 0 under centroids) to Bin-T NJ's 26.18 is the contribution of the within-bin Student-t tails.");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "kdisc13_centroid_ablation.csv"), "w") do io
    println(io, "model,K_disc,IS_KS_pct,OoS_KS_pct,IS_AD_pct,OoS_AD_pct,sim_kurt,ACF_MAE");
    println(io, "Discrete NJ centroid,$K_DISC,$(round(100*is_ks, digits=2)),$(round(100*oos_ks, digits=2)),$(round(100*is_ad, digits=2)),$(round(100*oos_ad, digits=2)),$(round(sim_kurt, digits=3)),$(round(acf, digits=5))");
end

println("\n" * "="^72);
println("  Track Minor 6 complete.");
println("  Text: $(joinpath(TRACK_DIR, "Kdisc13_Centroid.txt"))");
println("  CSV : $(joinpath(PAPER_ROBUSTNESS_DIR, "kdisc13_centroid_ablation.csv"))");
println("="^72);

# ========================================================================================= #
# run_chmm_t_shared_nu_crps.jl
#
# Scores OoS (and IS) sample-CRPS for the STORED shared-ν CHMM-t K = 3 simulations in
# results/chmm_t_shared_nu/sims_K3.jld2. The original ablation runner
# (run_chmm_t_shared_nu.jl) reported KS / kurtosis / ACF-MAE but never CRPS, so the paper
# had no stored CRPS value for the shared-ν headline row (second-audit item 3). Uses the
# same unbiased sorted-ensemble estimator as run_kstar3_headline.jl so values are directly
# comparable with results/kstar3_headline/summary.txt.
#
# Output: results/chmm_t_shared_nu/crps.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Statistics, Printf, JLD2, DataFrames

const DT        = 1/252;
const RISK_FREE = 0.0;

const OUT_DIR = joinpath(_ROOT, "results", "chmm_t_shared_nu");
const SIM_PATH = joinpath(OUT_DIR, "sims_K3.jld2");

# Sample CRPS via the unbiased sorted-ensemble identity (same as run_kstar3_headline.jl)
function _sample_crps(y::Real, x::AbstractVector)
    N = length(x);
    s1 = mean(abs.(x .- y));
    xs = sort(x);
    s2_terms = sum(xs[i] * (2i - N - 1) for i in 1:N);
    s2 = s2_terms / (N * (N - 1));
    return s1 - s2;
end

function _crps_path_mean(R::AbstractVector, sim::AbstractMatrix)
    n = length(R);
    mean_crps = 0.0;
    @inbounds for t in 1:n
        x = view(sim, t, :);
        mean_crps += _sample_crps(R[t], x);
    end
    return mean_crps / n;
end

# ----------------------------------------------------------------------------------------- #
println("[1/3] Loading SPY IS / OoS (identical to run_chmm_t_shared_nu.jl)...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = all_R[:, idx_spy];
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
n_oos = length(R_oos);
println("  IS = $n_is  OoS = $n_oos");

# ----------------------------------------------------------------------------------------- #
println("[2/3] Loading stored shared-ν K = 3 simulations...");
d = JLD2.load(SIM_PATH);
sim_is = d["is"];
sim_oos = d["oos"];
# Orient to (T × P) as _crps_path_mean expects
sim_is = size(sim_is, 1) == n_is ? sim_is : permutedims(sim_is);
sim_oos = size(sim_oos, 1) == n_oos ? sim_oos : permutedims(sim_oos);
@assert size(sim_is, 1) == n_is "IS sim length mismatch: $(size(sim_is)) vs n_is=$n_is";
@assert size(sim_oos, 1) == n_oos "OoS sim length mismatch: $(size(sim_oos)) vs n_oos=$n_oos";
println("  sims: IS $(size(sim_is))  OoS $(size(sim_oos))");

# ----------------------------------------------------------------------------------------- #
println("[3/3] Scoring CRPS...");
crps_is = _crps_path_mean(R_is, sim_is);
crps_oos = _crps_path_mean(R_oos, sim_oos);
@printf("  shared-ν CHMM-t (K = 3): CRPS IS = %.4f   CRPS OoS = %.4f\n", crps_is, crps_oos);

out_path = joinpath(OUT_DIR, "crps.txt");
open(out_path, "w") do io
    println(io, "Sample CRPS for the shared-ν CHMM-t (K = 3), scored on the stored");
    println(io, "sims_K3.jld2 simulations with the unbiased sorted-ensemble estimator of");
    println(io, "run_kstar3_headline.jl (comparable with results/kstar3_headline/summary.txt).");
    println(io, "Paths: $(size(sim_oos, 2)); IS T = $n_is; OoS T = $n_oos.");
    println(io);
    @printf(io, "CRPS IS  = %.4f\n", crps_is);
    @printf(io, "CRPS OoS = %.4f\n", crps_oos);
end
println("[done] Wrote $out_path");

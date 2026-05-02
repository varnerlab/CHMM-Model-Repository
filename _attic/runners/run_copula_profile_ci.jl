# =========================================================================== #
# run_copula_profile_ci.jl
#
# B2 (REVISION_PLAN_V3_TO_ACCEPT.md): Profile-LL Wilks 95% CI for ν* in the
# Student-t copula on the six-asset universe. Refines the original 10-point
# grid {2, 3, 4, 5, 6, 8, 10, 15, 20, 30} with unit spacing on ν ∈ [4, 12]
# (which brackets the original ν* = 6) and computes the Wilks 95% CI as the
# contiguous range of ν within −1.92 profile-LL units of the optimum.
#
# Output: results/copula_profile_ci/profile_ll_fine.csv
#         results/copula_profile_ci/profile_ll_summary.txt
# =========================================================================== #

const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."));
using Pkg; Pkg.activate(_PROJECT_ROOT);
include(joinpath(_PROJECT_ROOT, "Include.jl"));

using Random
using Statistics
using LinearAlgebra
using Printf
using DelimitedFiles

const SEED  = 20260420;
const DT    = 1/252;
const RISK_FREE = 0.0;

const OUT_DIR = joinpath(_ROOT, "results", "copula_profile_ci");
mkpath(OUT_DIR);

Random.seed!(SEED);

println("="^70)
println("  B2: Profile-LL Wilks CI for ν* (six-asset Student-t copula)")
println("="^70)

# Data: same six-asset universe as the body cross-asset table
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);

cross_tickers = ["SPY", "NVDA", "JNJ", "JPM", "AAPL", "QQQ"];
cross_idx = [findfirst(==(t), all_tickers) for t in cross_tickers];
R_cross = all_R[:, cross_idx];

U     = _pit_ranks(R_cross);
τ_mat = _kendall_tau_matrix(R_cross);
Σ_cop = sin.((π/2) .* τ_mat);
Σ_cop = _nearest_psd(Σ_cop);

# Coarse grid (original) + fine integer grid in [4, 12]
fine_nu  = collect(4.0:1.0:12.0);
coarse_nu = [2.0, 3.0, 15.0, 20.0, 30.0];
ν_grid = sort(unique(vcat(fine_nu, coarse_nu)));

ll_grid = Float64[];
for ν in ν_grid
    push!(ll_grid, _tcopula_profile_loglik(U, Σ_cop, ν));
end

ν_star_idx = argmax(ll_grid);
ν_star = ν_grid[ν_star_idx];
ll_star = ll_grid[ν_star_idx];

# Wilks 95% CI: contiguous ν within -1.92 profile-LL units of the optimum
threshold = ll_star - 1.92;
in_ci = ll_grid .>= threshold;
# Contiguous block around argmax
function _ci_bounds(idx, in_flag)
    lo = idx; while lo > 1 && in_flag[lo - 1]; lo -= 1; end
    hi = idx; while hi < length(in_flag) && in_flag[hi + 1]; hi += 1; end
    return (lo, hi);
end
lo, hi = _ci_bounds(ν_star_idx, in_ci);
ν_lo = ν_grid[lo]; ν_hi = ν_grid[hi];

# Write CSV
open(joinpath(OUT_DIR, "profile_ll_fine.csv"), "w") do io
    write(io, "nu,profile_loglik,in_wilks_ci\n");
    for (ν, ll) in zip(ν_grid, ll_grid)
        in95 = ll >= threshold ? 1 : 0;
        write(io, @sprintf("%.1f,%.6f,%d\n", ν, ll, in95));
    end
end

# Summary
open(joinpath(OUT_DIR, "profile_ll_summary.txt"), "w") do io
    println(io, "Profile log-likelihood — Student-t copula on six-asset SPY universe");
    println(io, "="^70);
    println(io, "Grid:");
    println(io, rpad("nu", 8), "| log-L           | inside Wilks 95% CI");
    println(io, "-"^60);
    for (ν, ll) in zip(ν_grid, ll_grid)
        flag = ll >= threshold ? "*" : " ";
        println(io, rpad(ν, 8), "| ", lpad(@sprintf("%.4f", ll), 14), "  ", flag);
    end
    println(io);
    println(io, "ν* = $ν_star  (profile log-L = $(round(ll_star, digits=2)))");
    println(io, "Wilks 95% CI threshold (ll_star - 1.92) = $(round(threshold, digits=2))");
    println(io, "ν 95% CI = [$ν_lo, $ν_hi]");
end

@printf("[done] ν* = %.1f, 95%% CI = [%.1f, %.1f]\n", ν_star, ν_lo, ν_hi)
println("[done] Output: $OUT_DIR")

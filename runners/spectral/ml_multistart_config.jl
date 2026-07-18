# ========================================================================================= #
# ml_multistart_config.jl
#
# Single source of truth for the PUBLISHED converged multistart likelihood fits
# (the CHMM-N comparator of the spectral cross-ticker panel and the frontier
# experiment). Both run_spectral_rank_cross_ticker.jl and run_hmm_acf_frontier.jl
# consume these constants and the fitting call below, so the two runners cannot
# drift apart in start count, iteration cap, tolerance, or seed schedule
# (tenth review, engineering item; see CHANGELOG.md).
#
# `idx` is the position of the ticker in the sorted panel_tickers list shared by
# both runners (identical data path: MyPortfolioDataSet, max_days filter, sorted
# keys, log_growth_matrix at Δt = 1/252, risk-free 0).
# ========================================================================================= #

const ML_MULTISTART_SEED     = 20260420;
const ML_MULTISTART_N_STARTS = Dict(3 => 3, 18 => 5);
const ML_MULTISTART_MAX_ITER = 4000;
const ML_MULTISTART_TOL      = 1e-4;

ml_multistart_seed(idx::Int, K::Int) = ML_MULTISTART_SEED + 100 * idx + K;

"""
    fit_published_multistart(R, K, idx) -> (T, μ, σ, π, ll_history, γ, diagnostics)

The exact published converged multistart likelihood fit for ticker index `idx`
at state count `K`: `baum_welch_multistart` under the shared constants above.
Deterministic given (R, K, idx).
"""
function fit_published_multistart(R::Vector{Float64}, K::Int, idx::Int)
    return baum_welch_multistart(R, K; n_starts=ML_MULTISTART_N_STARTS[K],
                                 max_iter=ML_MULTISTART_MAX_ITER,
                                 tol=ML_MULTISTART_TOL,
                                 seed=ml_multistart_seed(idx, K));
end

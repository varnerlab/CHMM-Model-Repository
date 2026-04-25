# ========================================================================================= #
# run_track_c3_filter_var.jl
#
# Track C3b (revision response to referee comment M3):
# Filter-based one-step-ahead regime-conditional VaR.
#
# The Viterbi-based C3a script (run_track_c3_conditional_var.jl) computes the state
# sequence from the FULL OoS path, so the decoded state at time t depends on
# r_{t+1}, ..., r_T; the resulting state-conditional quantile is not a one-step-ahead
# backtest. The M3 fix replaces Viterbi (a smoother) with the forward filter:
#
#     π_t(k) = P(s_t = k | r_{1:t}, θ_IS)
#
# and takes the α-quantile of the one-step-ahead predictive mixture
#
#     F(r) = Σ_k π_t(k) F_k(r)
#
# where F_k is the emission CDF in state k. The resulting VaR_t^filter(α) is measurable
# w.r.t. the information set {r_1, ..., r_t}, so Kupiec and Christoffersen on the
# breach sequence 1{r_t ≤ VaR_t^filter(α)} are a valid one-step-ahead backtest.
#
# Coverage:
#   - CHMM-N, CHMM-t, CHMM-L (flat), both α = 0.01 and α = 0.05.
#   - SM variants are out of scope here (see M6 in revision-code-todo.md).
#
# Outputs:
#   results/track_c3/VaR_filter_LR_tests.txt
#   ../CHMM-paper/results/robustness/filter_var_backtest.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const K_MAIN    = 18;
const MAX_ITER  = 60;

const TRACK_C3_DIR     = joinpath(_ROOT, "results", "track_c3");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(TRACK_C3_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  Track C3b: filter-based one-step-ahead regime-conditional VaR (M3)")
println("  Seed $SEED, K=$K_MAIN")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

# --------------------------------------------------------------------------------------- #
# Fit flat CHMM-N / -t / -L (same seeding as C3a for consistency)
# --------------------------------------------------------------------------------------- #
println("\n[fit] Fitting flat CHMM-N / -t / -L on IS...");
Random.seed!(SEED + 7);
flat_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
Random.seed!(SEED + 8);
flat_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
Random.seed!(SEED + 9);
flat_l = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  flat CHMMs ready.");

# --------------------------------------------------------------------------------------- #
# Forward filter: π_t(k) = P(s_t = k | r_{1:t})
# --------------------------------------------------------------------------------------- #
"""
    forward_filter(observations, model) -> Matrix{Float64}  (N × K)

Computes the normalised forward posterior π_t(k) = P(s_t = k | r_{1:t}, θ) under the
fitted CHMM model, with uniform initial prior (matching the Viterbi convention in
`src/Compute.jl`). Returns an N × K matrix whose rows sum to one.

Uses rescaled log-space forward recursion to prevent underflow over long horizons.
"""
function forward_filter(observations::Vector{Float64},
                        model::Union{MyContinuousHiddenMarkovModel,
                                     MyStudentTHiddenMarkovModel,
                                     MyLaplaceHiddenMarkovModel})::Matrix{Float64}
    N = length(observations);
    K = length(model.states);

    T_mat = zeros(K, K);
    for i in 1:K
        T_mat[i, :] = probs(model.transition[i]);
    end
    log_T = log.(T_mat);

    # log-space rescaled forward: at each t, store log π_t (normalised to sum 1)
    log_pi = zeros(N, K);

    # t = 1: uniform prior × emission likelihood, normalised
    log_pi[1, :] .= log(1.0 / K) .+ [logpdf(model.emission[k], observations[1]) for k in 1:K];
    log_pi[1, :] .-= _logsumexp_vec(log_pi[1, :]);

    # t >= 2: π_t(k) ∝ p(r_t | s_t=k) × Σ_j π_{t-1}(j) T_{j,k}
    for t in 2:N
        for k in 1:K
            log_pi[t, k] = _logsumexp_vec(log_pi[t-1, :] .+ log_T[:, k]) +
                           logpdf(model.emission[k], observations[t]);
        end
        log_pi[t, :] .-= _logsumexp_vec(log_pi[t, :]);
    end

    return exp.(log_pi);
end

# --------------------------------------------------------------------------------------- #
# Mixture quantile via bisection on the mixture CDF
# --------------------------------------------------------------------------------------- #
"""
    mixture_quantile(α, weights, components; tol=1e-8, max_iter=200) -> Float64

Returns the α-quantile of the mixture distribution F(r) = Σ_k w_k F_k(r) via
bisection on the mixture CDF. The initial bracket [min_k VaR_k, max_k VaR_k] is
tight for a convex mixture of monotone CDFs.
"""
function mixture_quantile(α::Float64,
                          weights::AbstractVector{Float64},
                          components::AbstractVector;
                          tol::Float64=1e-8, max_iter::Int=200)::Float64
    K = length(weights);
    @assert K == length(components);

    # Per-component α-quantiles bracket the mixture α-quantile.
    qs = [quantile(components[k], α) for k in 1:K];
    lo = minimum(qs);
    hi = maximum(qs);

    if hi - lo < tol
        return 0.5 * (lo + hi);
    end

    mixture_cdf(r) = sum(weights[k] * cdf(components[k], r) for k in 1:K);

    for _ in 1:max_iter
        mid = 0.5 * (lo + hi);
        if mixture_cdf(mid) < α
            lo = mid;
        else
            hi = mid;
        end
        if hi - lo < tol
            break;
        end
    end

    return 0.5 * (lo + hi);
end

# --------------------------------------------------------------------------------------- #
# Filter-based VaR series
# --------------------------------------------------------------------------------------- #
function filter_var_series(model, observations::Vector{Float64}, α::Float64)::Vector{Float64}
    π_mat = forward_filter(observations, model);   # N × K
    N, K = size(π_mat);
    components = [model.emission[k] for k in 1:K];
    out = Vector{Float64}(undef, N);
    for t in 1:N
        out[t] = mixture_quantile(α, view(π_mat, t, :), components);
    end
    return out;
end

println("\n[filter VaR] Computing one-step-ahead filter-based VaR on R_oos...");

cvar01 = Dict{String, Vector{Float64}}();
cvar05 = Dict{String, Vector{Float64}}();

cvar01["CHMM-N"] = filter_var_series(flat_n, R_oos, 0.01);
cvar05["CHMM-N"] = filter_var_series(flat_n, R_oos, 0.05);
cvar01["CHMM-t"] = filter_var_series(flat_t, R_oos, 0.01);
cvar05["CHMM-t"] = filter_var_series(flat_t, R_oos, 0.05);
cvar01["CHMM-L"] = filter_var_series(flat_l, R_oos, 0.01);
cvar05["CHMM-L"] = filter_var_series(flat_l, R_oos, 0.05);
println("  filter VaR sequences ready.");

# --------------------------------------------------------------------------------------- #
# Kupiec + Christoffersen
# --------------------------------------------------------------------------------------- #
MODELS = ["CHMM-N", "CHMM-t", "CHMM-L"];

results = Dict{String, NamedTuple}();
for m in MODELS
    v01 = cvar01[m]; v05 = cvar05[m];
    br01 = R_oos .<= v01;
    br05 = R_oos .<= v05;
    k01 = kupiec_lr(br01, 0.01); k05 = kupiec_lr(br05, 0.05);
    c01 = christoffersen_lr(br01); c05 = christoffersen_lr(br05);
    cc01 = christoffersen_cc(br01, 0.01); cc05 = christoffersen_cc(br05, 0.05);
    results[m] = (
        med_v01=median(v01), med_v05=median(v05),
        br_rate_01=k01.breach_rate, br_rate_05=k05.breach_rate,
        k01=k01, k05=k05, c01=c01, c05=c05, cc01=cc01, cc05=cc05,
    );
    println("  $m  br01=$(round(100*k01.breach_rate, digits=2))% LRuc01=$(round(k01.LR, digits=2)) LRind01=$(round(c01.LR, digits=2)) | br05=$(round(100*k05.breach_rate, digits=2))% LRuc05=$(round(k05.LR, digits=2)) LRind05=$(round(c05.LR, digits=2))");
end

# --------------------------------------------------------------------------------------- #
# Output: text report (CHMM-Model convention)
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_C3_DIR, "VaR_filter_LR_tests.txt"), "w") do io
    println(io, "="^150);
    println(io, "Track C3b. Filter-based one-step-ahead regime-conditional VaR (M3 revision response). Kupiec + Christoffersen LR tests.");
    println(io, "="^150);
    println(io, "");
    println(io, "Setup  : π_t(k) = P(s_t = k | r_{1:t}, θ_IS) via forward filter under IS-fit flat CHMM.");
    println(io, "         VaR_t(α) = α-quantile of mixture Σ_k π_t(k) F_k, where F_k is the state-k emission CDF.");
    println(io, "         Breach: 1{r_t ≤ VaR_t(α)}. VaR_t is measurable w.r.t. {r_1, ..., r_t}, so this");
    println(io, "         is a one-step-ahead backtest (contrast with C3a Viterbi smoother).");
    println(io, "Target : α ∈ {0.01, 0.05}. LR_uc ~ χ²(1) (crit 3.84). LR_ind ~ χ²(1) (crit 3.84).");
    println(io, "Data   : SPY OoS window ($n_oos daily observations, 2024-01-04 to 2026-04-20).");
    println(io, "");
    println(io, rpad("Model",      8), " | ",
                rpad("medVaR01",   9), " | ", rpad("br%01",      6), " | ",
                rpad("LR_uc01",    7), " | ", rpad("p_uc01",     6), " | ",
                rpad("LR_ind01",   8), " | ", rpad("p_ind01",    7), " | ",
                rpad("medVaR05",   9), " | ", rpad("br%05",      6), " | ",
                rpad("LR_uc05",    7), " | ", rpad("p_uc05",     6), " | ",
                rpad("LR_ind05",   8), " | ", rpad("p_ind05",    7));
    println(io, "-"^150);
    for m in MODELS
        r = results[m];
        println(io, rpad(m, 8), " | ",
                    rpad(round(r.med_v01, digits=3),      9), " | ",
                    rpad(round(100*r.br_rate_01, digits=1), 6), " | ",
                    rpad(round(r.k01.LR, digits=2),       7), " | ",
                    rpad(round(r.k01.pvalue, digits=3),   6), " | ",
                    rpad(round(r.c01.LR, digits=2),       8), " | ",
                    rpad(round(r.c01.pvalue, digits=3),   7), " | ",
                    rpad(round(r.med_v05, digits=3),      9), " | ",
                    rpad(round(100*r.br_rate_05, digits=1), 6), " | ",
                    rpad(round(r.k05.LR, digits=2),       7), " | ",
                    rpad(round(r.k05.pvalue, digits=3),   6), " | ",
                    rpad(round(r.c05.LR, digits=2),       8), " | ",
                    rpad(round(r.c05.pvalue, digits=3),   7));
    end
    println(io, "="^150);
    println(io, "");
    println(io, "Reading: LR_uc tests breach rate = α (unconditional coverage);");
    println(io, "         LR_ind tests independence of consecutive breaches.");
    println(io, "         Both should be below 3.84 (χ²(1) 95% critical value) for a passing one-step-ahead conditional VaR.");
    println(io, "         Compare these LR_ind values against C3a (Viterbi smoother, reported in `VaR_conditional_LR_tests.txt`)");
    println(io, "         and against Track A A8 (unconditional VaR, reported in `VaR_LR_tests.txt`).");
    println(io, "         M5 caveat: at T_OoS = $n_oos and α = 0.01 the expected breach count is ≈ $(round(n_oos * 0.01, digits=1)),");
    println(io, "         so Christoffersen independence is substantially underpowered; small LR_ind values should be");
    println(io, "         read as 'consistent with independence' rather than as resolved rankings.");
end

# --------------------------------------------------------------------------------------- #
# Output: CSV for direct import into the paper (CHMM-paper/results/robustness/)
# --------------------------------------------------------------------------------------- #
open(joinpath(PAPER_ROBUSTNESS_DIR, "filter_var_backtest.csv"), "w") do io
    println(io, "model,alpha,br_pct,LR_uc,p_uc,LR_ind,p_ind,T_OoS,med_VaR");
    for m in MODELS
        r = results[m];
        println(io, "$m,0.01,$(round(100*r.br_rate_01, digits=2)),$(round(r.k01.LR, digits=2)),$(round(r.k01.pvalue, digits=4)),$(round(r.c01.LR, digits=2)),$(round(r.c01.pvalue, digits=4)),$n_oos,$(round(r.med_v01, digits=3))");
        println(io, "$m,0.05,$(round(100*r.br_rate_05, digits=2)),$(round(r.k05.LR, digits=2)),$(round(r.k05.pvalue, digits=4)),$(round(r.c05.LR, digits=2)),$(round(r.c05.pvalue, digits=4)),$n_oos,$(round(r.med_v05, digits=3))");
    end
end

println("\n" * "="^72);
println("  Track C3b complete.");
println("  Text report : $(joinpath(TRACK_C3_DIR, "VaR_filter_LR_tests.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_ROBUSTNESS_DIR, "filter_var_backtest.csv"))");
println("="^72);

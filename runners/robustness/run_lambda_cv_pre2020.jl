# ========================================================================================= #
# run_lambda_cv_pre2020.jl
#
# Held-out cross-validation of the 1/ν_k shrinkage hyperparameter λ on the strictly
# pre-2020 slice. Addresses peer-review item P1.3 (R2.W6, R3.Q2): the λ = 20 choice
# in run_chmm_t_penalised_headline.jl was selected from a sweep on the full IS window
# and the OoS evaluation window may be partially aware of the choice. To remove that
# concern, refit the penalised CHMM-t on the strictly pre-2020 estimation slice
# (2014-01-03 -- 2018-06-29) and select λ* by held-out per-observation log-likelihood
# on the validation slice (2018-07-02 -- 2019-12-31). Both slices are fully pre-COVID
# and pre-2022-rate-hike.
#
# Output: results/nu_shrinkage_sweep/lambda_cv_pre2020.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, HypothesisTests, Distributions, Printf
using Dates

const SEED      = 20260420;
# Body operating point K* = 3 by default. The original headline runner used
# K = 18; per Item 3 of REVIEW_RESPONSE_PLAN.md, the lambda must be re-tuned at
# K = 3 to remove the "tuned at 18, re-used at 3" disclaimer in the discussion.
# Override at the shell to reproduce the K = 18 sweep:
#     LAMBDA_CV_K=18 julia --project=. runners/robustness/run_lambda_cv_pre2020.jl
const K_MAIN    = parse(Int, get(ENV, "LAMBDA_CV_K", "3"));
const MAX_ITER  = 60;
const N_PATHS   = 500;          # validation only; cheaper than headline N_PATHS
const DT        = 1/252;
const RISK_FREE = 0.0;
const ALPHA_KS  = 0.05;
const LAMBDA_GRID = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0];

const OUT_DIR = joinpath(_ROOT, "results", "nu_shrinkage_sweep");
mkpath(OUT_DIR);

println("="^88);
println("  Held-out CV of 1/ν_k shrinkage λ on pre-2020 slice  (peer-review P1.3)");
println("="^88);
println("  λ grid: $LAMBDA_GRID   K = $K_MAIN   seed = $SEED");
println();

# ----------------------------------------------------------------------------------------- #
# Data: strictly pre-2020 slices
# ----------------------------------------------------------------------------------------- #
println("[data] Loading SPY IS + dates ...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
spy_df = train_dataset["SPY"];
n_full = nrow(spy_df);
# Compute G_t in same units as run_full_rebuild.jl (annualised excess log return)
prices = spy_df.volume_weighted_average_price;
returns = (1 / DT) .* (log.(prices[2:end]) .- log.(prices[1:end-1])) .- RISK_FREE;
spy_dates = Date.(spy_df.timestamp[2:end]);  # row j corresponds to G between (j-1, j)

# Estimation slice: 2014-01-03 .. 2018-06-29
est_start  = Date("2014-01-03");
est_end    = Date("2018-06-29");
val_start  = Date("2018-07-02");
val_end    = Date("2019-12-31");

est_idx = findall(d -> est_start <= d <= est_end, spy_dates);
val_idx = findall(d -> val_start <= d <= val_end, spy_dates);
R_est = Vector{Float64}(returns[est_idx]);
R_val = Vector{Float64}(returns[val_idx]);
n_est = length(R_est); n_val = length(R_val);
@printf("  Estimation:  %s .. %s   n = %d\n",
        Dates.format(est_start, "yyyy-mm-dd"), Dates.format(est_end, "yyyy-mm-dd"), n_est);
@printf("  Validation:  %s .. %s   n = %d\n",
        Dates.format(val_start, "yyyy-mm-dd"), Dates.format(val_end, "yyyy-mm-dd"), n_val);
println();

# ----------------------------------------------------------------------------------------- #
# Validation log-likelihood under fitted (T, θ_1:K)
# ----------------------------------------------------------------------------------------- #
function _val_loglik(model::MyStudentTHiddenMarkovModel, R_val::Vector{Float64})
    # Forward log-likelihood under fitted parameters: standard log-space scaling recursion
    K = length(model.states);
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    # initial distribution: stationary
    π̄ = (T_mat^1000)[1, :];
    log_T = log.(max.(T_mat, 1e-300));
    n_o = length(R_val);
    α_log = zeros(K);
    @inbounds for k in 1:K
        α_log[k] = log(π̄[k]) + logpdf(model.emission[k], R_val[1]);
    end
    log_lik = 0.0;
    new_α = similar(α_log);
    for t in 2:n_o
        for k in 1:K
            s = -Inf;
            for j in 1:K
                v = α_log[j] + log_T[j, k];
                s = max(s, v) + log1p(exp(min(s, v) - max(s, v)));
            end
            new_α[k] = s + logpdf(model.emission[k], R_val[t]);
        end
        # rescale to avoid underflow
        m_new = maximum(new_α);
        log_lik += m_new;
        α_log .= new_α .- m_new;
    end
    # Add the last-step total
    m_final = maximum(α_log);
    log_lik += m_final + log(sum(exp.(α_log .- m_final)));
    # Note: the running rescaling has accumulated the logsumexp partial; the value
    # log_lik now equals the total log-likelihood log p(R_val).
    return log_lik;
end

function _stationary(model, K::Int)
    T = zeros(K, K);
    for i in 1:K; T[i, :] = probs(model.transition[i]); end
    π = (T^1000)[1, :];
    return Categorical(π);
end

function _sim_paths(model, sd, n::Int, np::Int)
    sim = Matrix{Float64}(undef, n, np);
    for p in 1:np
        s0 = rand(sd);
        st = model(s0, n);
        for j in 1:n; sim[j, p] = rand(model.emission[st[j]]); end
    end
    return sim;
end

function _val_ks(R_val::Vector{Float64}, model, K::Int; np::Int=N_PATHS)
    sd = _stationary(model, K);
    Random.seed!(SEED + 7);
    sim = _sim_paths(model, sd, length(R_val), np);
    pass = 0;
    for i in 1:np
        if pvalue(ApproximateTwoSampleKSTest(R_val, sim[:, i])) >= ALPHA_KS
            pass += 1;
        end
    end
    return 100 * pass / np;
end

# ----------------------------------------------------------------------------------------- #
# Sweep
# ----------------------------------------------------------------------------------------- #
println("[fit] Sweeping λ on estimation slice and scoring on validation slice ...");
results = NamedTuple[];
val_obs_kurt = kurtosis(R_val);
for λ in LAMBDA_GRID
    Random.seed!(SEED);
    m = build(MyStudentTHiddenMarkovModel,
        (observations=R_est, number_of_states=K_MAIN, max_iter=MAX_ITER,
         ν_shrink_rate=λ));
    val_ll = _val_loglik(m, R_val);
    val_ll_pn = val_ll / n_val;
    val_ks = _val_ks(R_val, m, K_MAIN);
    # Estimation-slice simulated kurtosis under fitted model
    sd = _stationary(m, K_MAIN);
    Random.seed!(SEED + 11);
    sim_est = _sim_paths(m, sd, n_est, 200);
    sim_kurt_est = mean([kurtosis(sim_est[:, p]) for p in 1:200]);
    Random.seed!(SEED + 13);
    sim_val = _sim_paths(m, sd, n_val, 200);
    sim_kurt_val = mean([kurtosis(sim_val[:, p]) for p in 1:200]);
    push!(results, (
        λ = λ, val_ll = val_ll, val_ll_pn = val_ll_pn, val_ks = val_ks,
        sim_kurt_est = sim_kurt_est, sim_kurt_val = sim_kurt_val,
        obs_kurt_est = kurtosis(R_est), obs_kurt_val = val_obs_kurt,
    ));
    @printf("  λ = %6.1f  val LL/obs = %+8.4f  val KS = %5.1f%%  sim kurt est/val = %5.2f / %5.2f\n",
            λ, val_ll_pn, val_ks, sim_kurt_est, sim_kurt_val);
end

best_ll = argmax([r.val_ll_pn for r in results]);
best_ks = argmax([r.val_ks    for r in results]);
println();
@printf("λ* by validation log-likelihood: %.1f  (val LL/obs = %.4f)\n",
        results[best_ll].λ, results[best_ll].val_ll_pn);
@printf("λ* by validation KS pass rate  : %.1f  (val KS = %.1f%%)\n",
        results[best_ks].λ, results[best_ks].val_ks);

# ----------------------------------------------------------------------------------------- #
# Output
# ----------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "lambda_cv_pre2020.txt");
open(out_path, "w") do io
    println(io, "="^96);
    println(io, "Held-out CV of 1/ν_k shrinkage λ on pre-2020 slice  (peer-review P1.3 / R2.W6)");
    println(io, "="^96);
    @printf(io, "Estimation slice : %s .. %s  (n = %d)\n",
            Dates.format(est_start, "yyyy-mm-dd"), Dates.format(est_end, "yyyy-mm-dd"), n_est);
    @printf(io, "Validation slice : %s .. %s  (n = %d)\n",
            Dates.format(val_start, "yyyy-mm-dd"), Dates.format(val_end, "yyyy-mm-dd"), n_val);
    @printf(io, "K = %d, paths = %d, seed = %d, λ grid = %s\n",
            K_MAIN, N_PATHS, SEED, LAMBDA_GRID);
    println(io);
    println(io, "Validation criterion: per-observation held-out log-likelihood (primary) and KS pass rate.");
    println(io);
    @printf(io, "%-8s %-12s %-14s %-9s %-12s %-12s\n",
            "λ", "val LL", "val LL/obs", "val KS%", "sim K (est)", "sim K (val)");
    println(io, "-"^96);
    for r in results
        @printf(io, "%-8.1f %-12.2f %-14.4f %-9.1f %-12.2f %-12.2f\n",
                r.λ, r.val_ll, r.val_ll_pn, r.val_ks, r.sim_kurt_est, r.sim_kurt_val);
    end
    println(io);
    @printf(io, "λ* by validation log-likelihood : %.1f  (val LL/obs = %.4f)\n",
            results[best_ll].λ, results[best_ll].val_ll_pn);
    @printf(io, "λ* by validation KS pass rate   : %.1f  (val KS = %.1f%%)\n",
            results[best_ks].λ, results[best_ks].val_ks);
    println(io);
    println(io, "Reading: if λ* matches the body operating point λ = 20 within the grid spacing,");
    println(io, "the body penalty is held-out-clean. If the held-out optimum is materially different,");
    println(io, "the body should report the held-out-selected λ* alongside.");
end

println();
println("[done] Wrote $out_path");

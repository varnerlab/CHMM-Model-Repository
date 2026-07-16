# =========================================================================== #
# run_var_inference_upgrade.jl
#
# Third-party-review items (c)/(d): inference upgrades for the VaR back-test
# table.
#
#   1. EXACT unconditional coverage. For every row of
#      results/conditional_var_all_families/conditional_var_panel.csv, the new
#      MS-GARCH rows (results/msgarch_conditional_var/msgarch_conditional_var.csv),
#      and the filtered-bootstrap / CAViaR contenders, compute the exact
#      two-sided binomial p-value (minimum-likelihood method) for the breach
#      count under X ~ Bin(T = 573, alpha), alongside the asymptotic Kupiec p.
#
#   2. FINITE-SAMPLE DQ CALIBRATION at alpha = 0.01. The Engle-Manganelli DQ
#      chi^2_6 p-values in the paper are asymptotic; with only ~6 expected
#      breaches in 573 days the chi^2 reference can be badly sized. Parametric
#      bootstrap under each fitted null:
#        - CHMM-N K = 3 (refit on IS with the current src code, seed 20260420),
#        - MS-GARCH K = 4 (parameters from run_msgarch_conditional_var.jl),
#      simulate B synthetic OoS paths of length 573 from the fitted model
#      (started from the filtered state at the IS/OoS boundary), rerun the
#      same one-step-ahead VaR filter + DQ statistic on each path, and report
#      the bootstrap p-value of the observed DQ statistic,
#      p_boot = (1 + #{DQ_b >= DQ_obs}) / (B + 1).
#
#   3. PAIRED QUANTILE-LOSS (pinball) COMPARISON at alpha in {0.01, 0.05}
#      across CHMM-N K=3, MS-GARCH K=4, filtered bootstrap, CAViaR on the
#      same 573 forecast days. CHMM-N and MS-GARCH per-day series are read
#      from their stored CSVs; the filtered-bootstrap and CAViaR series are
#      not stored per-day, so they are recomputed here by replicating
#      runners/baselines/run_filtered_bootstrap_var.jl and run_caviar_var.jl
#      (deterministic given seed 20260420; recomputed breach counts are
#      cross-checked against the stored summary CSVs). Mean loss differentials
#      vs the CHMM-N K=3 benchmark are reported with HAC (Newey-West,
#      Bartlett kernel) t-statistics.
#
# Output: results/var_inference_upgrade/var_inference_upgrade.{txt,csv}
# =========================================================================== #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));

using Random, Statistics, LinearAlgebra, Distributions, Printf

const SEED      = 20260420;
const MAX_ITER  = 60;            # CHMM Baum-Welch iterations (harness convention)
const DT        = 1/252;
const RISK_FREE = 0.0;
const Q_LAGS    = 4;
const B_BOOT    = parse(Int, get(ENV, "VAR_DQ_BOOT_B", "500"));

const OUT_DIR = joinpath(_ROOT, "results", "var_inference_upgrade");
mkpath(OUT_DIR);

const PANEL_CSV   = joinpath(_ROOT, "results", "conditional_var_all_families", "conditional_var_panel.csv");
const MSGARCH_CSV = joinpath(_ROOT, "results", "msgarch_conditional_var", "msgarch_conditional_var.csv");
const MSGARCH_PARAMS_K4 = joinpath(_ROOT, "results", "msgarch_conditional_var", "msgarch_params_K4.txt");
const CHMM_SERIES_CSV   = joinpath(_ROOT, "results", "conditional_var_all_families", "conditional_var_series_chmmN_k3.csv");
const MSGARCH_SERIES_K4 = joinpath(_ROOT, "results", "msgarch_conditional_var", "var_series_K4.csv");
const FB_CSV     = joinpath(_ROOT, "results", "filtered_bootstrap_var", "filtered_bootstrap_var.csv");
const CAVIAR_CSV = joinpath(_ROOT, "results", "caviar_var", "caviar_var.csv");

println("="^80)
println("  VaR inference upgrade: exact binomial UC, bootstrap DQ, pinball loss")
println("  B (DQ bootstrap) = $B_BOOT")
println("="^80)

# -------- data (identical to run_conditional_var_all_families.jl) --------
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
filtered = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; filtered[t] = data; end
end
all_tickers = keys(filtered) |> collect |> sort;
all_R = log_growth_matrix(filtered, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));

# Boundary fix (2026-07 audit): prepend the last-IS-session -> first-held-out-session
# return so the forecast-origin set matches the CHMM VaR panel (n_oos = 573).
R_oos = vcat((1/DT) * log(oos_dataset["SPY"][1, :volume_weighted_average_price] /
                          train_dataset["SPY"][end, :volume_weighted_average_price]) - RISK_FREE,
             R_oos);

n_is = length(R_is); n_oos = length(R_oos);
println("[setup] IS = $n_is, OoS = $n_oos")

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
function mixture_quantile(α::Float64, weights, components;
                          lo::Float64=-50.0, hi::Float64=50.0,
                          tol::Float64=1e-6, max_iter::Int=80)
    cdf_at(x) = sum(w * cdf(c, x) for (w, c) in zip(weights, components));
    a, b = lo, hi;
    if cdf_at(a) - α > 0; a = -200.0; end
    if cdf_at(b) - α < 0; b = 200.0; end
    for _ in 1:max_iter
        m = 0.5 * (a + b);
        fm = cdf_at(m) - α;
        if abs(fm) < tol; return m; end
        if fm < 0; a = m; else; b = m; end
    end
    return 0.5 * (a + b);
end

# Engle-Manganelli DQ statistic. Same regressors as run_engle_manganelli_dq.jl
# (intercept + q lagged centred hits + contemporaneous VaR_t); the OLS step uses
# a QR solve (X \ Y) so rank-deficient bootstrap replicates (e.g. zero-breach
# paths, where the lagged-hit columns are constant) remain well-defined:
# DQ = beta' X'X beta / [alpha(1-alpha)] = ||X beta||^2 / [alpha(1-alpha)].
function dq_stat(breaches::AbstractVector{Bool}, var_thr::AbstractVector{Float64},
                 α::Float64; q::Int = Q_LAGS)
    n = length(breaches);
    hit = Float64.(breaches) .- α;
    n_eff = n - q;
    p = q + 2;
    X = zeros(n_eff, p);
    Y = zeros(n_eff);
    for t in (q+1):n
        row = t - q;
        X[row, 1] = 1.0;
        for i in 1:q
            X[row, 1 + i] = hit[t - i];
        end
        X[row, p] = var_thr[t];
        Y[row] = hit[t];
    end
    beta_hat = X \ Y;
    fitted = X * beta_hat;
    DQ = dot(fitted, fitted) / (α * (1.0 - α));
    p_value = 1.0 - cdf(Chisq(p), DQ);
    return (DQ = DQ, p_value = p_value, dof = p);
end

# Exact two-sided binomial p-value (minimum-likelihood method): sum P(X = k)
# over all k whose point probability does not exceed that of the observed count.
function exact_binom_p(x::Int, n::Int, p0::Float64)
    d = Binomial(n, p0);
    px = pdf(d, x);
    s = 0.0;
    for k in 0:n
        pk = pdf(d, k);
        if pk <= px * (1.0 + 1e-12); s += pk; end
    end
    return min(1.0, s);
end

# Newey-West (Bartlett) HAC t-statistic for the mean of a loss-differential series.
function nw_tstat(d::AbstractVector{Float64})
    T = length(d);
    L = floor(Int, 4.0 * (T / 100.0)^(2.0 / 9.0));
    μd = mean(d);
    u = d .- μd;
    s = mean(u .^ 2);
    for l in 1:L
        γl = sum(u[(l+1):end] .* u[1:(end-l)]) / T;
        s += 2.0 * (1.0 - l / (L + 1.0)) * γl;
    end
    se = sqrt(max(s, 1e-300) / T);
    t = μd / se;
    p = 2.0 * (1.0 - cdf(Normal(), abs(t)));
    return (mean = μd, t = t, p = p, lags = L);
end

pinball(r::Float64, q::Float64, α::Float64) = (α - (r < q ? 1.0 : 0.0)) * (r - q);

# --------------------------------------------------------------------------- #
# Section 1: exact binomial unconditional coverage
# --------------------------------------------------------------------------- #
println("\n[1] Exact binomial unconditional-coverage p-values (T = $n_oos)");

exact_rows = NamedTuple[];
function push_exact!(model::String, K::Int, α::Float64, x::Int, p_uc_asym::Float64)
    p_ex = exact_binom_p(x, n_oos, α);
    push!(exact_rows, (model=model, K=K, α=α, breaches=x,
                       p_exact=p_ex, p_uc_asym=p_uc_asym));
    @printf("  %-22s K=%-3d α=%.2f  br=%2d  p_exact=%.4f  p_uc(asym)=%.4f\n",
        model, K, α, x, p_ex, p_uc_asym);
end

panel_df = CSV.read(PANEL_CSV, DataFrame);
for r in eachrow(panel_df)
    push_exact!(String(r.family), Int(r.K), Float64(r.alpha), Int(r.breaches), Float64(r.p_uc));
end
msg_df = CSV.read(MSGARCH_CSV, DataFrame);
for r in eachrow(msg_df)
    push_exact!("MS-GARCH", Int(r.K), Float64(r.alpha), Int(r.breaches), Float64(r.p_uc));
end
fb_df = CSV.read(FB_CSV, DataFrame);
for r in eachrow(fb_df)
    push_exact!("Filtered bootstrap", 0, Float64(r.alpha), Int(r.breaches), Float64(r.p_uc));
end
cav_df = CSV.read(CAVIAR_CSV, DataFrame);
for r in eachrow(cav_df)
    push_exact!("CAViaR-SAV", 0, Float64(r.alpha), Int(r.breaches), Float64(r.p_uc));
end

# --------------------------------------------------------------------------- #
# Section 2a: CHMM-N K=3 — observed VaR series + parametric DQ bootstrap (1%)
# --------------------------------------------------------------------------- #
println("\n[2a] CHMM-N K=3: refit on IS + DQ bootstrap at α = 0.01 (B = $B_BOOT)");

function filter_predictive(y::AbstractVector, T::AbstractMatrix, components, π_init)
    Kc = length(components); n = length(y);
    pred = zeros(n + 1, Kc); pred[1, :] = π_init;
    for t in 1:n
        logpost = log.(pred[t, :]) .+ [logpdf(components[k], y[t]) for k in 1:Kc];
        Z = _logsumexp_vec(logpost);
        post = isfinite(Z) ? exp.(logpost .- Z) : copy(pred[t, :]);
        pred[t + 1, :] = vec(post' * T);
    end
    return pred;
end

Random.seed!(SEED);
chmm = build(MyContinuousHiddenMarkovModel,
             (observations=R_is, number_of_states=3, max_iter=MAX_ITER));
K_chmm = length(chmm.states);
T_chmm = zeros(K_chmm, K_chmm);
for i in 1:K_chmm; T_chmm[i, :] = probs(chmm.transition[i]); end
π̄_chmm = (T_chmm^2000)[1, :]; π̄_chmm ./= sum(π̄_chmm);
comps_chmm = [chmm.emission[k] for k in 1:K_chmm];

# Predictive state weights at the IS/OoS boundary (information through the IS end)
pred_is = filter_predictive(R_is, T_chmm, comps_chmm, π̄_chmm);
w0_chmm = pred_is[end, :];

# One-step-ahead 1% VaR series over a segment continuing from the boundary
function chmm_var01_segment(R_seg::AbstractVector)
    pred = filter_predictive(R_seg, T_chmm, comps_chmm, w0_chmm);
    v = zeros(length(R_seg));
    for j in 1:length(R_seg)
        v[j] = mixture_quantile(0.01, pred[j, :], comps_chmm);
    end
    return v;
end

var01_chmm_obs = chmm_var01_segment(R_oos);
br_chmm_obs = R_oos .< var01_chmm_obs;
dq_chmm_obs = dq_stat(br_chmm_obs, var01_chmm_obs, 0.01);
@printf("  observed: br=%d  DQ=%.3f  p_asym=%.4f  (panel cross-check: 9 breaches)\n",
    sum(br_chmm_obs), dq_chmm_obs.DQ, dq_chmm_obs.p_value);

function simulate_chmm_segment(rng::AbstractRNG, n::Int)
    out = zeros(n);
    s = rand(rng, Categorical(collect(w0_chmm)));
    for t in 1:n
        out[t] = rand(rng, comps_chmm[s]);
        s = rand(rng, Categorical(collect(T_chmm[s, :])));
    end
    return out;
end

rng_chmm = MersenneTwister(SEED + 1);
dq_boot_chmm = zeros(B_BOOT);
t0 = time();
for b in 1:B_BOOT
    R_sim = simulate_chmm_segment(rng_chmm, n_oos);
    v = chmm_var01_segment(R_sim);
    br = R_sim .< v;
    dq_boot_chmm[b] = dq_stat(br, v, 0.01).DQ;
    if b % 100 == 0; @printf("    b = %d / %d  (%.1f s)\n", b, B_BOOT, time() - t0); end
end
p_boot_chmm = (1 + count(>=(dq_chmm_obs.DQ), dq_boot_chmm)) / (B_BOOT + 1);
@printf("  bootstrap: p_boot = %.4f  (null DQ quantiles: 50%%=%.2f 90%%=%.2f 95%%=%.2f 99%%=%.2f)\n",
    p_boot_chmm, quantile(dq_boot_chmm, 0.5), quantile(dq_boot_chmm, 0.9),
    quantile(dq_boot_chmm, 0.95), quantile(dq_boot_chmm, 0.99));

# --------------------------------------------------------------------------- #
# Section 2b: MS-GARCH K=4 — observed VaR series + parametric DQ bootstrap (1%)
# --------------------------------------------------------------------------- #
println("\n[2b] MS-GARCH K=4: DQ bootstrap at α = 0.01 (B = $B_BOOT)");

function load_msgarch_params(path::String)
    d = Dict{String,Vector{Float64}}();
    K = 0;
    for ln in readlines(path)
        parts = split(strip(ln));
        isempty(parts) && continue;
        key = String(parts[1]);
        vals = parse.(Float64, parts[2:end]);
        if key == "K"; K = Int(vals[1]); else; d[key] = vals; end
    end
    T = zeros(K, K);
    for i in 1:K; T[i, :] = d["T$(i)"]; end
    return (K=K, ω=d["omega"], α=d["alpha"], β=d["beta"], μ=d["mu"][1], T=T);
end

msg4 = load_msgarch_params(MSGARCH_PARAMS_K4);

# Filter over IS to obtain the boundary state: predictive regime weights and
# per-regime conditional variances for the first OoS day.
_, σ2_is_msg, γ_is_msg = _hamilton_filter_msgarch(R_is, msg4.ω, msg4.α, msg4.β, msg4.μ, msg4.T);
w0_msg = msg4.T' * γ_is_msg[end, :];
w0_msg = max.(w0_msg, 0.0); w0_msg ./= sum(w0_msg);
σ2_0_msg = [max(msg4.ω[k] + msg4.α[k] * (R_is[end] - msg4.μ)^2 + msg4.β[k] * σ2_is_msg[end, k], 1e-12)
            for k in 1:msg4.K];

# One-step-ahead 1% VaR filter over a segment continuing from the boundary
function msgarch_var01_segment(R_seg::AbstractVector)
    K = msg4.K;
    π_pred = copy(w0_msg);
    σ2 = copy(σ2_0_msg);
    v = zeros(length(R_seg));
    for t in 1:length(R_seg)
        comps = [Normal(msg4.μ, sqrt(σ2[k])) for k in 1:K];
        v[t] = mixture_quantile(0.01, π_pred, comps);
        # posterior update + one-step prediction
        logpost = log.(max.(π_pred, 1e-300)) .+ [logpdf(comps[k], R_seg[t]) for k in 1:K];
        Z = _logsumexp_vec(logpost);
        post = isfinite(Z) ? exp.(logpost .- Z) : copy(π_pred);
        π_pred = msg4.T' * post;
        π_pred = max.(π_pred, 0.0); π_pred ./= sum(π_pred);
        r = R_seg[t] - msg4.μ;
        for k in 1:K
            σ2[k] = max(msg4.ω[k] + msg4.α[k] * r^2 + msg4.β[k] * σ2[k], 1e-12);
        end
    end
    return v;
end

var01_msg_obs = msgarch_var01_segment(R_oos);
br_msg_obs = R_oos .< var01_msg_obs;
dq_msg_obs = dq_stat(br_msg_obs, var01_msg_obs, 0.01);
@printf("  observed: br=%d  DQ=%.3f  p_asym=%.4f\n",
    sum(br_msg_obs), dq_msg_obs.DQ, dq_msg_obs.p_value);

function simulate_msgarch_segment(rng::AbstractRNG, n::Int)
    K = msg4.K;
    out = zeros(n);
    s = rand(rng, Categorical(collect(w0_msg)));
    σ2 = copy(σ2_0_msg);
    for t in 1:n
        out[t] = msg4.μ + sqrt(σ2[s]) * randn(rng);
        r = out[t] - msg4.μ;
        for k in 1:K
            σ2[k] = max(msg4.ω[k] + msg4.α[k] * r^2 + msg4.β[k] * σ2[k], 1e-12);
        end
        s = rand(rng, Categorical(collect(msg4.T[s, :])));
    end
    return out;
end

rng_msg = MersenneTwister(SEED + 2);
dq_boot_msg = zeros(B_BOOT);
t0 = time();
for b in 1:B_BOOT
    R_sim = simulate_msgarch_segment(rng_msg, n_oos);
    v = msgarch_var01_segment(R_sim);
    br = R_sim .< v;
    dq_boot_msg[b] = dq_stat(br, v, 0.01).DQ;
    if b % 100 == 0; @printf("    b = %d / %d  (%.1f s)\n", b, B_BOOT, time() - t0); end
end
p_boot_msg = (1 + count(>=(dq_msg_obs.DQ), dq_boot_msg)) / (B_BOOT + 1);
@printf("  bootstrap: p_boot = %.4f  (null DQ quantiles: 50%%=%.2f 90%%=%.2f 95%%=%.2f 99%%=%.2f)\n",
    p_boot_msg, quantile(dq_boot_msg, 0.5), quantile(dq_boot_msg, 0.9),
    quantile(dq_boot_msg, 0.95), quantile(dq_boot_msg, 0.99));

# --------------------------------------------------------------------------- #
# Section 3: paired pinball-loss comparison
# --------------------------------------------------------------------------- #
println("\n[3] Paired quantile-loss (pinball) comparison, α in {0.01, 0.05}");

# CHMM-N K=3 and MS-GARCH K=4: stored per-day series
chmm_series = CSV.read(CHMM_SERIES_CSV, DataFrame);
msg_series  = CSV.read(MSGARCH_SERIES_K4, DataFrame);
@assert nrow(chmm_series) == n_oos "CHMM series length mismatch";
@assert nrow(msg_series)  == n_oos "MS-GARCH series length mismatch";
@assert maximum(abs.(chmm_series.R_oos .- msg_series.R_oos)) < 1e-4 "R_oos mismatch across stored series";

# Filtered bootstrap: recompute per-day series (replicates run_filtered_bootstrap_var.jl)
println("  [recompute] filtered bootstrap (GARCH(1,1)-t + empirical residual quantile)...");
Random.seed!(SEED);
garcht = fit_garcht11(R_is);
sigma_is_fb = sqrt.(garcht.σ2_hist);
z_is_fb = (R_is .- garcht.μ) ./ sigma_is_fb;
function _roll_sigma(garcht_fit, R_is_v, R_oos_v)
    n = length(R_oos_v);
    out = zeros(n);
    s2 = garcht_fit.σ2_hist[end];
    r_lag = R_is_v[end];
    for t in 1:n
        s2 = max(garcht_fit.ω + garcht_fit.α * (r_lag - garcht_fit.μ)^2 + garcht_fit.β * s2, 1e-12);
        out[t] = sqrt(s2);
        r_lag = R_oos_v[t];
    end
    return out;
end
sigma_oos_fb = _roll_sigma(garcht, R_is, R_oos);
function emp_quantile(x::AbstractVector, alpha::Float64)
    sorted = sort(x);
    n = length(sorted);
    pos = max(1, min(n, ceil(Int, alpha * n)));
    return sorted[pos];
end
fb_var = Dict(α => garcht.μ .+ sigma_oos_fb .* emp_quantile(z_is_fb, α) for α in (0.01, 0.05));
for α in (0.01, 0.05)
    n_br = sum(R_oos .< fb_var[α]);
    stored = fb_df[isapprox.(fb_df.alpha, α), :breaches][1];
    @printf("    α=%.2f  recomputed breaches = %d  (stored: %d)\n", α, n_br, stored);
end

# CAViaR SAV: recompute per-day series (replicates run_caviar_var.jl)
println("  [recompute] CAViaR-SAV (Engle-Manganelli 2004)...");
function caviar_sav_path(beta::Vector{Float64}, R::AbstractVector, var_init::Float64)
    b1, b2, b3 = beta[1], beta[2], beta[3];
    T = length(R);
    var_t = zeros(T);
    var_t[1] = var_init;
    for t in 2:T
        var_t[t] = b1 + b2 * var_t[t-1] + b3 * abs(R[t-1]);
    end
    return var_t;
end
function tick_loss(beta::Vector{Float64}, R::AbstractVector, alpha::Float64, var_init::Float64)
    var_t = caviar_sav_path(beta, R, var_init);
    T = length(R);
    s = 0.0;
    for t in 1:T
        diff = R[t] - var_t[t];
        s += (alpha - (R[t] < var_t[t] ? 1.0 : 0.0)) * diff;
    end
    return s / T;
end
function _fit_caviar_sav(R::AbstractVector, alpha::Float64)
    var_init = quantile(R[1:min(250, length(R))], alpha);
    best_loss = Inf; best_beta = [var_init * (1 - 0.9), 0.9, 0.05];
    for b1_try in [-0.20, -0.10, -0.05, -0.01, 0.0]
        for b2_try in [0.85, 0.90, 0.95]
            for b3_try in [-0.30, -0.15, -0.05, 0.05, 0.15]
                p = [b1_try, b2_try, b3_try];
                l = tick_loss(p, R, alpha, var_init);
                if isfinite(l) && l < best_loss
                    best_loss = l; best_beta = copy(p);
                end
            end
        end
    end
    best, _ = _nelder_mead(p -> tick_loss(p, R, alpha, var_init), best_beta; max_iter=2000);
    return (beta = best, var_init = var_init);
end
cav_var = Dict{Float64, Vector{Float64}}();
for α in (0.01, 0.05)
    Random.seed!(SEED);
    fitc = _fit_caviar_sav(R_is, α);
    var_full = caviar_sav_path(fitc.beta, vcat(R_is, R_oos), fitc.var_init);
    cav_var[α] = var_full[(n_is+1):end];
    n_br = sum(R_oos .< cav_var[α]);
    stored = cav_df[isapprox.(cav_df.alpha, α), :breaches][1];
    @printf("    α=%.2f  recomputed breaches = %d  (stored: %d)\n", α, n_br, stored);
end

# Assemble per-day quantile forecasts per model
series = Dict(
    "CHMM-N K=3"         => Dict(0.01 => Vector{Float64}(chmm_series.var_01),
                                 0.05 => Vector{Float64}(chmm_series.var_05)),
    "MS-GARCH K=4"       => Dict(0.01 => Vector{Float64}(msg_series.var_01),
                                 0.05 => Vector{Float64}(msg_series.var_05)),
    "Filtered bootstrap" => fb_var,
    "CAViaR-SAV"         => cav_var,
);
const BENCH = "CHMM-N K=3";
MODELS = ["CHMM-N K=3", "MS-GARCH K=4", "Filtered bootstrap", "CAViaR-SAV"];

pinball_rows = NamedTuple[];
for α in (0.01, 0.05)
    L = Dict(m => [pinball(R_oos[t], series[m][α][t], α) for t in 1:n_oos] for m in MODELS);
    println(@sprintf("\n  α = %.2f  (mean pinball loss; diff vs %s, NW t)", α, BENCH));
    for m in MODELS
        ml = mean(L[m]);
        if m == BENCH
            @printf("    %-20s  mean loss = %.5f  (benchmark)\n", m, ml);
            push!(pinball_rows, (model=m, α=α, mean_loss=ml,
                                 dloss=0.0, t_nw=NaN, p_nw=NaN, lags=0));
        else
            d = L[m] .- L[BENCH];
            nw = nw_tstat(d);
            @printf("    %-20s  mean loss = %.5f  Δ vs bench = %+.5f  t_NW = %+.2f  p = %.4f (lags=%d)\n",
                m, ml, nw.mean, nw.t, nw.p, nw.lags);
            push!(pinball_rows, (model=m, α=α, mean_loss=ml,
                                 dloss=nw.mean, t_nw=nw.t, p_nw=nw.p, lags=nw.lags));
        end
    end
end

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
csv_path = joinpath(OUT_DIR, "var_inference_upgrade.csv");
open(csv_path, "w") do io
    println(io, "section,model,K,alpha,breaches,p_exact_binom,p_uc_asym,DQ_obs,DQ_p_asym,DQ_p_boot,B,mean_pinball,dloss_vs_chmm,t_NW,p_NW,nw_lags");
    for r in exact_rows
        @printf(io, "exact_binomial,%s,%d,%.2f,%d,%.5f,%.5f,,,,,,,,,\n",
            r.model, r.K, r.α, r.breaches, r.p_exact, r.p_uc_asym);
    end
    @printf(io, "dq_bootstrap,CHMM-N,3,0.01,%d,,,%.4f,%.5f,%.5f,%d,,,,,\n",
        sum(br_chmm_obs), dq_chmm_obs.DQ, dq_chmm_obs.p_value, p_boot_chmm, B_BOOT);
    @printf(io, "dq_bootstrap,MS-GARCH,4,0.01,%d,,,%.4f,%.5f,%.5f,%d,,,,,\n",
        sum(br_msg_obs), dq_msg_obs.DQ, dq_msg_obs.p_value, p_boot_msg, B_BOOT);
    for r in pinball_rows
        @printf(io, "pinball,%s,,%.2f,,,,,,,,%.6f,%.6f,%.4f,%.5f,%d\n",
            r.model, r.α, r.mean_loss, r.dloss,
            isnan(r.t_nw) ? 0.0 : r.t_nw, isnan(r.p_nw) ? 1.0 : r.p_nw, r.lags);
    end
end

txt_path = joinpath(OUT_DIR, "var_inference_upgrade.txt");
open(txt_path, "w") do io
    println(io, "="^110);
    println(io, "VaR inference upgrade (third-party review items c/d)");
    println(io, "SPY OoS, T = $n_oos aligned one-step-ahead forecasts, seed = $SEED");
    println(io, "="^110);

    println(io, "\n[1] Exact two-sided binomial unconditional-coverage p-values, X ~ Bin($n_oos, α)");
    println(io, "    (minimum-likelihood two-sided p; asymptotic Kupiec chi^2_1 p alongside)");
    @printf(io, "    %-22s %-4s %-5s %-9s %-9s %-9s\n", "model", "K", "α", "breaches", "p_exact", "p_uc_asym");
    println(io, "    " * "-"^70);
    for r in exact_rows
        @printf(io, "    %-22s %-4d %-5.2f %-9d %-9.4f %-9.4f\n",
            r.model, r.K, r.α, r.breaches, r.p_exact, r.p_uc_asym);
    end

    println(io, "\n[2] Finite-sample DQ calibration at α = 0.01 (parametric bootstrap, B = $B_BOOT)");
    println(io, "    Simulate OoS paths of length $n_oos from the fitted null (started from the");
    println(io, "    filtered IS/OoS-boundary state), rerun the identical one-step-ahead VaR filter");
    println(io, "    + DQ statistic (q = $Q_LAGS lags + intercept + VaR_t) on each path;");
    println(io, "    p_boot = (1 + #{DQ_b >= DQ_obs}) / (B + 1).");
    @printf(io, "    %-14s %-9s %-9s %-9s %-9s %-30s\n",
        "model", "breaches", "DQ_obs", "p_asym", "p_boot", "null DQ q50/q90/q95/q99");
    println(io, "    " * "-"^90);
    @printf(io, "    %-14s %-9d %-9.3f %-9.4f %-9.4f %5.2f / %5.2f / %5.2f / %5.2f\n",
        "CHMM-N K=3", sum(br_chmm_obs), dq_chmm_obs.DQ, dq_chmm_obs.p_value, p_boot_chmm,
        quantile(dq_boot_chmm, 0.5), quantile(dq_boot_chmm, 0.9),
        quantile(dq_boot_chmm, 0.95), quantile(dq_boot_chmm, 0.99));
    @printf(io, "    %-14s %-9d %-9.3f %-9.4f %-9.4f %5.2f / %5.2f / %5.2f / %5.2f\n",
        "MS-GARCH K=4", sum(br_msg_obs), dq_msg_obs.DQ, dq_msg_obs.p_value, p_boot_msg,
        quantile(dq_boot_msg, 0.5), quantile(dq_boot_msg, 0.9),
        quantile(dq_boot_msg, 0.95), quantile(dq_boot_msg, 0.99));
    println(io, "    (asymptotic chi^2_6 quantiles for reference: q50 = 5.35, q90 = 10.64, q95 = 12.59, q99 = 16.81)");

    println(io, "\n[3] Paired quantile-loss (pinball) comparison on the same $n_oos forecast days");
    println(io, "    L_t(α) = (α - 1{r_t < q_t}) (r_t - q_t); Δ = mean L(model) - mean L(CHMM-N K=3);");
    println(io, "    HAC (Newey-West, Bartlett) t-statistic on the daily loss differential.");
    println(io, "    CHMM-N / MS-GARCH series from stored per-day CSVs; filtered-bootstrap and CAViaR");
    println(io, "    per-day series recomputed by replicating their runners (breach counts cross-checked");
    println(io, "    against the stored summary CSVs).");
    for α in (0.01, 0.05)
        @printf(io, "\n    α = %.2f\n", α);
        @printf(io, "    %-20s %-12s %-12s %-8s %-8s %-6s\n",
            "model", "mean loss", "Δ vs CHMM", "t_NW", "p_NW", "lags");
        println(io, "    " * "-"^70);
        for r in pinball_rows
            if r.α == α
                if r.model == BENCH
                    @printf(io, "    %-20s %-12.5f %-12s %-8s %-8s %-6s\n",
                        r.model, r.mean_loss, "(bench)", "-", "-", "-");
                else
                    @printf(io, "    %-20s %-12.5f %-+12.5f %-8.2f %-8.4f %-6d\n",
                        r.model, r.mean_loss, r.dloss, r.t_nw, r.p_nw, r.lags);
                end
            end
        end
    end
    println(io);
    println(io, "Reading: positive Δ means the model has HIGHER (worse) average quantile loss than");
    println(io, "the CHMM-N K=3 benchmark at that α; |t_NW| > 1.96 indicates a significant difference.");
end

println("\n[done] Output: $OUT_DIR");

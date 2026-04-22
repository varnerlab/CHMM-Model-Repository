# ========================================================================================= #
# run_vix_diagnostics.jl
#
# Block D diagnostics on the fitted VIX CHMMs.
# Observable: log(VIX) levels.
#
#   - Stationarity of log(VIX) paths     (augmented Dickey-Fuller, pass-rate)
#   - AR(1) half-life of log(VIX) levels (match observed, literature 20-40 d)
#   - ACF of log(VIX) levels (slow-decay benchmark; Ghosh 2022,
#     Chen et al. on VIX long memory, fractional d ∈ [0.3, 0.5])
#   - KS on log(VIX) level marginals     (pass-rate at α = 0.05)
#   - Skewness / excess kurtosis of implied Δlog(VIX) increments
#   - Hill right-tail index (upper 5%)   (heavy-tail proxy on |Δlog VIX|)
#
# Plus two baseline rows (AR(1) on log levels and Gaussian i.i.d. on Δlog VIX)
# and Figure D1 four-panel (path overlay / increment histogram / ACF(log VIX) /
# QQ-plot on increments).
#
# Outputs:
#   results/VIX/Table-D1-BlockD.txt
#   results/VIX/Table-D1-BlockD.tex       (LaTeX tabular rows, for paper inclusion)
#   results/VIX/Fig-D1-BlockD-Panel.{svg,pdf}
#   (also copied to CHMM-paper/paper/sections/figs/ if that directory exists)
#
# Usage: julia --project=. run_vix_diagnostics.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random;
Random.seed!(20260421);

const K              = 18;
const RESULTS_DIR    = joinpath(_ROOT, "results", "VIX");
const PAPER_FIGS_DIR = joinpath(dirname(_ROOT), "CHMM-paper", "paper", "sections", "figs");
const ACF_LAG        = 252;
const ADF_ALPHA      = 0.05;
const KS_ALPHA       = 0.05;
const HILL_TAIL_FRAC = 0.05;      # top 5%
const N_SIM_EVAL     = 200;       # paths to evaluate for rate metrics (≤ N_PATHS)

# --- Load observed series + simulations ---
obs = load(joinpath(RESULTS_DIR, "observed.jld2"));
log_vix_train = obs["log_vix_train"];       # log(VIX) in-sample
log_vix_oos   = obs["log_vix_oos"];
dlog_train    = obs["dlog_train"];          # Δlog(VIX) IS, retained for moments of increments
y_train       = obs["y_train"];             # log(VIX) levels (IS)
y_oos         = obs["y_oos"];               # log(VIX) levels (OoS)
n_train = length(y_train);

sims = Dict{String,NamedTuple}();
for (tag, label) in (("N", "CHMM-N"), ("t", "CHMM-t"), ("L", "CHMM-L"))
    d = load(joinpath(RESULTS_DIR, "K$K", tag, "simulations.jld2"));
    sims[label] = (sim_is=d["sim_is"], sim_oos=d["sim_oos"]);
end
ar1 = load(joinpath(RESULTS_DIR, "baselines", "ar1.jld2"));
iid = load(joinpath(RESULTS_DIR, "baselines", "gaussian_iid.jld2"));
sims["AR(1) levels"]     = (sim_is=ar1["sim_is"], sim_oos=ar1["sim_oos"]);
sims["Gaussian i.i.d."]  = (sim_is=iid["sim_is"], sim_oos=iid["sim_oos"]);

println("="^72);
println("  Block D diagnostics");
println("  IS obs: $n_train   OoS obs: $(length(y_oos))");
println("="^72);

# ========================================================================================= #
# Diagnostic primitives
# ========================================================================================= #

# AR(1) fit on a vector; returns (c, φ, σ_ε, half_life)
function _ar1_fit(y::AbstractVector{Float64})
    X = hcat(ones(length(y) - 1), y[1:end-1]);
    β = X \ y[2:end];
    c, φ = β[1], β[2];
    ε = y[2:end] .- X * β;
    σε = std(ε);
    hl = φ > 0 && φ < 1 ? log(2) / (-log(φ)) : NaN;
    return c, φ, σε, hl;
end

# Augmented Dickey-Fuller (AR(1) form) with no lags and intercept only.
# Null H0: unit root. We implement the OLS Δy_t = α + (φ - 1) y_{t-1} + ε_t and
# compare the t-statistic on (φ - 1) to MacKinnon 5% critical value for the
# single-series ADF with intercept (no trend): τ_0.05 = -2.86.
# Reference: MacKinnon (1996), "Numerical Distribution Functions for Unit
# Root and Cointegration Tests", JAE 11(6), 601-618.
const ADF_CRIT_5PCT_INTERCEPT = -2.86;

function _adf_intercept(y::AbstractVector{Float64})
    T  = length(y);
    dy = y[2:T] .- y[1:T-1];
    X  = hcat(ones(T-1), y[1:T-1]);
    β  = X \ dy;
    resid = dy .- X * β;
    dof = T - 1 - size(X, 2);
    σ2 = sum(resid.^2) / dof;
    XtX_inv = inv(X' * X);
    se_γ = sqrt(σ2 * XtX_inv[2, 2]);
    τ = β[2] / se_γ;                  # t-stat on γ = φ - 1
    return τ;
end

_adf_rejects(y) = _adf_intercept(y) < ADF_CRIT_5PCT_INTERCEPT;

# Hill (1975) right-tail estimator on positive order statistics of |x|.
# Returns α̂ = 1/ξ where ξ = (1/k) Σ log(x_(n-i+1) / x_(n-k))   for i = 1..k.
function _hill_right(x::AbstractVector{Float64}; frac::Float64=HILL_TAIL_FRAC)
    xs = sort(x);
    n = length(xs);
    k = max(5, floor(Int, frac * n));
    u = xs[n - k];
    if u <= 0 || !isfinite(u); return NaN; end
    logs = log.(xs[n - k + 1 : n] ./ u);
    ξ = mean(logs);
    return ξ > 0 ? 1.0 / ξ : NaN;
end

# ACF distance (MAE) over lags 1..L between two series
function _acf_mae(x::AbstractVector{Float64}, y::AbstractVector{Float64}; L::Int=ACF_LAG)
    Lu = min(L, length(x)-1, length(y)-1);
    return mean(abs.(autocor(x, 1:Lu) .- autocor(y, 1:Lu)));
end

# ========================================================================================= #
# Per-family evaluation
# ========================================================================================= #
function _eval_row(label::String, sim_is::AbstractMatrix{Float64},
        y_train_levels::AbstractVector{Float64},
        dlog_train::AbstractVector{Float64})
    np = size(sim_is, 2);
    n_eval = min(N_SIM_EVAL, np);

    # Implied Δlog(VIX) increments from simulated log-level paths, pooled
    sim_incr_pool = vec(diff(sim_is[:, 1:n_eval]; dims=1));
    μs = mean(sim_incr_pool); σs = std(sim_incr_pool);
    skew_s = mean(((sim_incr_pool .- μs) ./ σs).^3);
    kurt_s = mean(((sim_incr_pool .- μs) ./ σs).^4) - 3.0;
    hill_s = _hill_right(abs.(sim_incr_pool));

    # Path-level diagnostics treat each column of sim_is as a log(VIX) level path
    ks_pass = 0;
    adf_pass = 0;
    hl_vals = Float64[];
    acf_mae_vals = Float64[];
    for i in 1:n_eval
        path = sim_is[:, i];

        # KS on log(VIX) level marginals
        pv = pvalue(ApproximateTwoSampleKSTest(y_train_levels, path));
        if pv > KS_ALPHA; ks_pass += 1; end

        # ADF on the simulated level path directly
        if _adf_rejects(path); adf_pass += 1; end

        _, φ, _, hl = _ar1_fit(path);
        push!(hl_vals, hl);
        push!(acf_mae_vals, _acf_mae(y_train_levels, path; L=ACF_LAG));
    end

    ks_pct   = round(100 * ks_pass / n_eval, digits=1);
    adf_pct  = round(100 * adf_pass / n_eval, digits=1);
    hl_med   = round(median(filter(isfinite, hl_vals)), digits=1);
    acf_mae  = round(median(acf_mae_vals), digits=4);

    return (label=label, ks=ks_pct, adf=adf_pct, hl=hl_med,
            acf_mae=acf_mae, skew=round(skew_s, digits=2),
            kurt=round(kurt_s, digits=2), hill=round(hill_s, digits=2));
end

println("\n[eval] computing per-family diagnostics...");
rows = NamedTuple[];

# Observed baseline for reference (put first). Moments of Δlog(VIX) increments.
μd = mean(dlog_train); σd = std(dlog_train);
skew_o = mean(((dlog_train .- μd) ./ σd).^3);
kurt_o = mean(((dlog_train .- μd) ./ σd).^4) - 3.0;
hill_o = _hill_right(abs.(dlog_train));
adf_o  = _adf_rejects(log_vix_train);
_, _, _, hl_o = _ar1_fit(log_vix_train);

push!(rows, (label="Observed (log VIX)",
             ks=NaN, adf=adf_o ? 100.0 : 0.0, hl=round(hl_o, digits=1),
             acf_mae=0.0, skew=round(skew_o, digits=2),
             kurt=round(kurt_o, digits=2), hill=round(hill_o, digits=2)));

for fam in ("CHMM-N", "CHMM-t", "CHMM-L")
    println("  $fam");
    s = sims[fam];
    push!(rows, _eval_row(fam, s.sim_is, y_train, dlog_train));
end

println("  AR(1) on log(VIX) levels");
push!(rows, _eval_row("AR(1) on log VIX",
                     sims["AR(1) levels"].sim_is, y_train, dlog_train));

println("  Gaussian i.i.d. on log(VIX)");
push!(rows, _eval_row("Gaussian i.i.d.",
                     sims["Gaussian i.i.d."].sim_is, y_train, dlog_train));

# ========================================================================================= #
# Emit Table D1 (text + LaTeX)
# ========================================================================================= #
_fmt(v; d=1) = (v isa Number && isfinite(v)) ? string(round(v, digits=d)) : "—";

open(joinpath(RESULTS_DIR, "Table-D1-BlockD.txt"), "w") do io
    println(io, "Table D1. Block D volatility-index generalization (VIX, K=$K, $N_SIM_EVAL paths)");
    println(io, "="^130);
    println(io, rpad("Model", 20) *
                lpad("KS %", 7) *
                lpad("ADF %", 7) *
                lpad("HL (d)", 9) *
                lpad("ACF-MAE", 9) *
                lpad("Skew", 8) *
                lpad("Kurt", 8) *
                lpad("Hill α̂", 9));
    println(io, "-"^130);
    for r in rows
        println(io, rpad(r.label, 20) *
                    lpad(_fmt(r.ks;   d=1), 7) *
                    lpad(_fmt(r.adf;  d=1), 7) *
                    lpad(_fmt(r.hl;   d=1), 9) *
                    lpad(_fmt(r.acf_mae; d=4), 9) *
                    lpad(_fmt(r.skew; d=2), 8) *
                    lpad(_fmt(r.kurt; d=2), 8) *
                    lpad(_fmt(r.hill; d=2), 9));
    end
    println(io, "="^130);
    println(io, "KS %:    share of simulated paths passing 2-sample KS on log(VIX) levels at α=0.05");
    println(io, "ADF %:   share of simulated log-level paths rejecting unit root at α=0.05 (MacKinnon 1996)");
    println(io, "HL (d):  median AR(1) half-life of log(VIX) = log(2)/-log(φ), trading days");
    println(io, "ACF-MAE: median MAE of ACF(log VIX), lags 1..$ACF_LAG, vs observed");
    println(io, "Skew/Kurt: sample moments of Δlog(VIX); Hill α̂: right-tail index on |Δlog(VIX)|, top $(round(100*HILL_TAIL_FRAC))%");
end

# LaTeX row dump (booktabs row format)
open(joinpath(RESULTS_DIR, "Table-D1-BlockD.tex"), "w") do io
    for r in rows
        println(io, "$(r.label) & " *
                    "$(_fmt(r.ks; d=1)) & " *
                    "$(_fmt(r.adf; d=1)) & " *
                    "$(_fmt(r.hl; d=1)) & " *
                    "$(_fmt(r.acf_mae; d=4)) & " *
                    "$(_fmt(r.skew; d=2)) & " *
                    "$(_fmt(r.kurt; d=2)) & " *
                    "$(_fmt(r.hill; d=2)) \\\\");
    end
end

println("\nTable D1 -> $(joinpath(RESULTS_DIR, "Table-D1-BlockD.txt"))");
println("Table D1 LaTeX -> $(joinpath(RESULTS_DIR, "Table-D1-BlockD.tex"))");

# ========================================================================================= #
# Figure D1: four-panel (path overlay / increment hist / ACF(log VIX) / QQ increments)
# ========================================================================================= #
println("\n[fig] building Figure D1 four-panel...");

# (a) Path overlay: observed log(VIX) vs CHMM-t median / envelope
cht = sims["CHMM-t"];
n_eval = min(N_SIM_EVAL, size(cht.sim_is, 2));
lv = cht.sim_is[:, 1:n_eval];
med_path = vec(median(lv, dims=2));
q10 = [quantile(lv[t, :], 0.10) for t in 1:size(lv, 1)];
q90 = [quantile(lv[t, :], 0.90) for t in 1:size(lv, 1)];

p_a = plot(1:length(y_train), y_train,
    lw=1.5, color=:red, alpha=0.85, label="Observed log(VIX)",
    title="(a) log(VIX) path: observed vs CHMM-t",
    titlefontsize=8, xlabel="Trading day (IS index)", ylabel="log(VIX)");
plot!(p_a, 1:size(lv, 1), med_path,
    lw=2, color=:navy, label="CHMM-t median of $n_eval sims");
plot!(p_a, 1:size(lv, 1), q10, fillrange=q90,
    alpha=0.15, color=:navy, label="CHMM-t 10-90 pct");

# (b) log(VIX) level marginal density: observed + three CHMM families
p_b = plot(title="(b) log(VIX) level marginal density",
    titlefontsize=8, xlabel="log(VIX)", ylabel="Probability density");
histogram!(p_b, y_train, normalize=:pdf, bins=60, alpha=0.35,
    color=:lightgray, label="Observed (T=$n_train)");
density!(p_b, vec(sims["CHMM-N"].sim_is[:, 1:n_eval]),
    lw=2, color=:steelblue,    label="CHMM-N pool");
density!(p_b, vec(sims["CHMM-t"].sim_is[:, 1:n_eval]),
    lw=2, color=:navy,         label="CHMM-t pool");
density!(p_b, vec(sims["CHMM-L"].sim_is[:, 1:n_eval]),
    lw=2, color=:firebrick,    label="CHMM-L pool");
xl = quantile(y_train, 0.001); xh = quantile(y_train, 0.999);
xlims!(p_b, xl - 0.1 * (xh - xl), xh + 0.1 * (xh - xl));

# (c) ACF(log VIX) observed vs simulated-median
acf_obs = autocor(y_train, 1:ACF_LAG);
acf_chmm = zeros(ACF_LAG, 3);
for (j, fam) in enumerate(("CHMM-N", "CHMM-t", "CHMM-L"))
    sim_is_ = sims[fam].sim_is;
    acf_arch = Array{Float64,2}(undef, ACF_LAG, n_eval);
    for i in 1:n_eval
        acf_arch[:, i] = autocor(sim_is_[:, i], 1:ACF_LAG);
    end
    acf_chmm[:, j] = vec(median(acf_arch, dims=2));
end
acf_ar1 = zeros(ACF_LAG);
let sim_is_ = sims["AR(1) levels"].sim_is
    acf_arch = Array{Float64,2}(undef, ACF_LAG, n_eval);
    for i in 1:n_eval
        acf_arch[:, i] = autocor(sim_is_[:, i], 1:ACF_LAG);
    end
    acf_ar1 = vec(median(acf_arch, dims=2));
end

p_c = plot(1:ACF_LAG, acf_obs,
    lw=2.5, color=:red, ls=:dash, label="Observed log(VIX)",
    title="(c) ACF(log VIX), lag 1..$ACF_LAG",
    titlefontsize=8, xlabel="Lag (trading days)", ylabel="ACF of log(VIX)");
plot!(p_c, 1:ACF_LAG, acf_chmm[:, 1], lw=1.5, color=:steelblue, label="CHMM-N median");
plot!(p_c, 1:ACF_LAG, acf_chmm[:, 2], lw=1.5, color=:navy,       label="CHMM-t median");
plot!(p_c, 1:ACF_LAG, acf_chmm[:, 3], lw=1.5, color=:firebrick,  label="CHMM-L median");
plot!(p_c, 1:ACF_LAG, acf_ar1,        lw=1.2, color=:darkgreen, ls=:dot, label="AR(1) median");

# (d) QQ plot of log(VIX) levels: observed vs CHMM-t pooled
probs_qq = range(0.001, 0.999, length=200);
q_obs = quantile(y_train, probs_qq);
q_chmt = quantile(vec(sims["CHMM-t"].sim_is[:, 1:n_eval]), probs_qq);
p_d = plot(q_obs, q_obs, lw=2, color=:black, ls=:dash, label="Identity",
    title="(d) Q-Q plot of log(VIX) levels",
    titlefontsize=8, xlabel="Observed quantiles", ylabel="CHMM-t quantiles (pooled)");
scatter!(p_d, q_obs, q_chmt, ms=3, alpha=0.6, color=:navy, label="CHMM-t");

fig = plot(p_a, p_b, p_c, p_d, layout=(2,2), size=(1200, 800),
    plot_title="Fig D1 (Block D). VIX CHMM generalization | K=$K | $n_eval sim paths",
    plot_titlefontsize=11);

svg_path = joinpath(RESULTS_DIR, "Fig-D1-BlockD-Panel.svg");
pdf_path = joinpath(RESULTS_DIR, "Fig-D1-BlockD-Panel.pdf");
savefig(fig, svg_path);
savefig(fig, pdf_path);
println("Figure D1 -> $svg_path");

# --- Copy figure to the paper's figs directory if present ---
if isdir(PAPER_FIGS_DIR)
    cp(pdf_path, joinpath(PAPER_FIGS_DIR, basename(pdf_path)); force=true);
    cp(svg_path, joinpath(PAPER_FIGS_DIR, basename(svg_path)); force=true);
    println("[copy] figure -> $PAPER_FIGS_DIR");
end

println();
println("="^72);
println("  BLOCK D DIAGNOSTICS COMPLETE");
println("="^72);

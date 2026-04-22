# ========================================================================================= #
# run_track_c1_smchmm.jl
#
# Track C1 runner: fit SM-CHMM-N / SM-CHMM-t / SM-CHMM-L on SPY at K=18, append
# simulation archives to the Track A cache, and recompute the extended metrics panel + the
# downstream utility panel (A1..A10) with the three new rows.
#
# Inputs:
#   data/... (via Files.jl)
#   results/track_a/sim_archive_cache.jld2 (from run_track_a_metrics.jl)
#
# Outputs:
#   results/track_c1/sim_archive_sm.jld2                 SM-CHMM simulation archives
#   results/track_c1/Table-4-Extended-Metrics-C1.txt     Extended panel incl. SM rows
#   results/track_c1/leverage_effect_c1.txt
#   results/track_c1/aggregational_kurtosis_c1.txt
#   results/track_c1/sim_pvalues_c1.txt
#   results/track_c1/tstr_vol_forecaster_c1.txt
#   results/track_c1/vol_target_strategy_c1.txt
#   results/track_c1/VaR_LR_tests_c1.txt
#   results/track_c1/sojourn_summary.txt                 per-state sojourn family + mean
#   results/track_c1/Track-C1-summary.txt
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER       = "SPY";
const RISK_FREE    = 0.0;
const DT           = 1/252;
const N_PATHS      = 1000;
const L_LAGS       = 252;
const K_MAIN       = 18;
const MAX_ITER     = 60;
const WINDOW_LEN   = 20;
const N_WINDOWS    = 500;
const SIG_DEPTH    = 3;
const MAX_LAG_LEV  = 20;
const HORIZONS_AG  = [1, 5, 10, 21];

const TRACK_A_DIR  = joinpath(_ROOT, "results", "track_a");
const TRACK_C_DIR  = joinpath(_ROOT, "results", "track_c1");
mkpath(TRACK_C_DIR);

println("="^72)
println("  Track C1 runner: semi-Markov CHMM (SM-CHMM-N / -t / -L) at K=$K_MAIN")
println("  Seed: $SEED, N_PATHS=$N_PATHS")
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
R_is = all_R[:, idx_spy];
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE);
n_oos = length(R_oos);
println("  IS: $n_is obs; OoS: $n_oos obs");

# --------------------------------------------------------------------------------------- #
# Fit SM-CHMMs
# --------------------------------------------------------------------------------------- #
sm_cache_path = joinpath(TRACK_C_DIR, "sm_models.jld2");
sim_cache_path = joinpath(TRACK_C_DIR, "sim_archive_sm.jld2");

println("\n[fit] Fitting SM-CHMM-N / -t / -L at K=$K_MAIN (plug-in estimator)...");

models = Dict{String, MySemiMarkovContinuousHMM}();
for (tag, fam) in (("SM-CHMM-N", :gaussian), ("SM-CHMM-t", :student_t), ("SM-CHMM-L", :laplace))
    print("  $tag...");
    Random.seed!(SEED + hash(tag) % 10^6);
    m = fit_sm_chmm(Vector{Float64}(R_is), K_MAIN, fam; max_iter=MAX_ITER);
    models[tag] = m;
    println(" done. sojourns: $(Dict(k => m.sojourn_family[k] for k in m.states))");
end

save(sm_cache_path, Dict("SM-CHMM-N" => models["SM-CHMM-N"],
                         "SM-CHMM-t" => models["SM-CHMM-t"],
                         "SM-CHMM-L" => models["SM-CHMM-L"]));

# --------------------------------------------------------------------------------------- #
# Simulate archives
# --------------------------------------------------------------------------------------- #
println("\n[sim] Simulating $N_PATHS paths per SM-CHMM variant...");

function _sim_archive(m::MySemiMarkovContinuousHMM, n_steps::Int, n_paths::Int, seed::Integer)
    Random.seed!(Int(seed));
    return simulate_sm_chmm(m, n_steps; n_paths=n_paths);
end

sm_archive = Dict{String,NamedTuple}();
for tag in ("SM-CHMM-N", "SM-CHMM-t", "SM-CHMM-L")
    print("  $tag IS ($n_is x $N_PATHS)...");
    s_is  = _sim_archive(models[tag], n_is,  N_PATHS, SEED + hash(tag * "is")  % 10^6);
    print(" OoS ($n_oos x $N_PATHS)...");
    s_oos = _sim_archive(models[tag], n_oos, N_PATHS, SEED + hash(tag * "oos") % 10^6);
    sm_archive[tag] = (is=s_is, oos=s_oos);
    println(" done.");
end

save(sim_cache_path, "archive", sm_archive);
println("  wrote $sim_cache_path");

# --------------------------------------------------------------------------------------- #
# Merge with Track A archive and recompute metrics (A1..A10)
# --------------------------------------------------------------------------------------- #
println("\n[merge] Loading Track A archive and appending SM rows...");
base_archive = load(joinpath(TRACK_A_DIR, "sim_archive_cache.jld2"))["archive"];
archive = merge(base_archive, sm_archive);

MODEL_ORDER_FULL = [
    "Bootstrap", "Gaussian", "Laplace",
    "DiscreteNJ", "DiscreteWJ", "GARCH",
    "CHMM-N", "CHMM-t", "CHMM-L",
    "SM-CHMM-N", "SM-CHMM-t", "SM-CHMM-L",
];

println("\n[A1,A2,A3] Computing MMD / sig-MMD / discriminator AUC (W=$WINDOW_LEN, n_windows=$N_WINDOWS)");

Random.seed!(SEED + 100);
obs_windows_is = let
    W_all = windowize(R_is, WINDOW_LEN; stride=max(1, (n_is - WINDOW_LEN) ÷ N_WINDOWS));
    idx = randperm(size(W_all, 2))[1:min(N_WINDOWS, size(W_all, 2))];
    W_all[:, idx];
end;
obs_windows_oos = let
    W_all = windowize(R_oos, WINDOW_LEN; stride=max(1, (n_oos - WINDOW_LEN) ÷ N_WINDOWS));
    idx = randperm(size(W_all, 2))[1:min(N_WINDOWS, size(W_all, 2))];
    W_all[:, idx];
end;
n_obs_win_is = size(obs_windows_is, 2);
n_obs_win_oos = size(obs_windows_oos, 2);

mmd_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER_FULL
    print("  $m...");
    rng = MersenneTwister(SEED + hash(m) % 10^6);
    s_is, s_oos = archive[m].is, archive[m].oos;
    syn_windows_is = sample_windows_from_archive(s_is, WINDOW_LEN, n_obs_win_is; rng=rng);
    syn_windows_oos = sample_windows_from_archive(s_oos, WINDOW_LEN, n_obs_win_oos; rng=rng);
    mmd_is = mmd2_rbf(obs_windows_is, syn_windows_is; rng=rng);
    mmd_oos = mmd2_rbf(obs_windows_oos, syn_windows_oos; rng=rng);
    sig_is = sig_mmd2(obs_windows_is, syn_windows_is; depth=SIG_DEPTH, rng=rng);
    sig_oos = sig_mmd2(obs_windows_oos, syn_windows_oos; depth=SIG_DEPTH, rng=rng);
    auc_is = discriminator_auc(R_is, s_is; window=WINDOW_LEN, n_windows=N_WINDOWS, rng=rng);
    auc_oos = discriminator_auc(R_oos, s_oos; window=WINDOW_LEN, n_windows=N_WINDOWS, rng=rng);
    mmd_results[m] = (
        mmd_is=mmd_is, mmd_oos=mmd_oos,
        sig_is=sig_is, sig_oos=sig_oos,
        auc_is=auc_is.auc, auc_is_std=auc_is.auc_std,
        auc_oos=auc_oos.auc, auc_oos_std=auc_oos.auc_std,
    );
    println(" MMD=$(round(mmd_is, digits=5)) sig=$(round(sig_is, digits=5)) AUC=$(round(auc_is.auc, digits=3))");
end

# --------------------------------------------------------------------------------------- #
# A6 / A7 / A9 for the three SM rows (to add to the panel)
# --------------------------------------------------------------------------------------- #
println("\n[A6] Leverage effect on SM rows...");
lev_obs_is = leverage_effect(R_is; max_lag=MAX_LAG_LEV);
lev_obs_oos = leverage_effect(R_oos; max_lag=MAX_LAG_LEV);

lev_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER_FULL
    s_is, s_oos = archive[m].is, archive[m].oos;
    avg_is = Float64[]; asym_is = Float64[];
    for i in 1:size(s_is, 2)
        r = leverage_effect(s_is[:, i]; max_lag=MAX_LAG_LEV);
        push!(avg_is, r.avg_neg); push!(asym_is, r.asymmetry);
    end
    avg_oos = Float64[]; asym_oos = Float64[];
    for i in 1:size(s_oos, 2)
        r = leverage_effect(s_oos[:, i]; max_lag=MAX_LAG_LEV);
        push!(avg_oos, r.avg_neg); push!(asym_oos, r.asymmetry);
    end
    lev_results[m] = (
        med_is=median(avg_is), med_asym_is=median(asym_is),
        med_oos=median(avg_oos), med_asym_oos=median(asym_oos),
        p_avg_is=sim_pvalue(lev_obs_is.avg_neg, avg_is),
        p_avg_oos=sim_pvalue(lev_obs_oos.avg_neg, avg_oos),
    );
end

println("[A7] Aggregational kurtosis on SM rows...");
agg_obs_is  = aggregational_kurtosis(R_is;  horizons=HORIZONS_AG);
agg_obs_oos = aggregational_kurtosis(R_oos; horizons=HORIZONS_AG);

agg_results = Dict{String,Dict}();
for m in MODEL_ORDER_FULL
    s_is, s_oos = archive[m].is, archive[m].oos;
    d = Dict{Symbol,Any}();
    for (tag, arch) in (:is => s_is, :oos => s_oos)
        per_h = Dict{Int, Vector{Float64}}();
        for h in HORIZONS_AG; per_h[h] = Float64[]; end
        for i in 1:size(arch, 2)
            ak = aggregational_kurtosis(arch[:, i]; horizons=HORIZONS_AG);
            for h in HORIZONS_AG; push!(per_h[h], ak[h]); end
        end
        d[tag] = per_h;
    end
    agg_results[m] = d;
end

println("[A9] p-values on SM rows...");
function _kurt(x)
    μ = mean(x); σ = std(x);
    return σ > 0 ? sum(((x .- μ) ./ σ) .^ 4) / length(x) - 3.0 : 0.0;
end
function _acf_mae_abs(obs, sim; L=L_LAGS)
    a_o = autocor(abs.(obs), 1:min(L, length(obs)-1));
    a_s = autocor(abs.(sim), 1:min(L, length(sim)-1));
    return mean(abs.(a_o .- a_s));
end

obs_kurt_is = _kurt(R_is); obs_kurt_oos = _kurt(R_oos);
obs_ak_is_5  = agg_obs_is[5];  obs_ak_is_21  = agg_obs_is[21];
obs_ak_oos_5 = agg_obs_oos[5]; obs_ak_oos_21 = agg_obs_oos[21];

pv_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER_FULL
    s_is, s_oos = archive[m].is, archive[m].oos;
    kurts_is  = [_kurt(s_is[:, i])  for i in 1:size(s_is, 2)];
    kurts_oos = [_kurt(s_oos[:, i]) for i in 1:size(s_oos, 2)];
    acf_mae_is  = [_acf_mae_abs(R_is,  s_is[:, i])  for i in 1:size(s_is, 2)];
    acf_mae_oos = [_acf_mae_abs(R_oos, s_oos[:, i]) for i in 1:size(s_oos, 2)];
    lev_entry = lev_results[m];
    ak5_is  = agg_results[m][:is][5];
    ak21_is = agg_results[m][:is][21];
    ak5_oos  = agg_results[m][:oos][5];
    ak21_oos = agg_results[m][:oos][21];
    pv = (
        p_kurt_is  = sim_pvalue(obs_kurt_is,  kurts_is),
        p_kurt_oos = sim_pvalue(obs_kurt_oos, kurts_oos),
        p_acf_is   = sim_pvalue(0.0, acf_mae_is),
        p_acf_oos  = sim_pvalue(0.0, acf_mae_oos),
        p_lev_is   = lev_entry.p_avg_is,
        p_lev_oos  = lev_entry.p_avg_oos,
        p_ak5_is   = sim_pvalue(obs_ak_is_5,  ak5_is),
        p_ak21_is  = sim_pvalue(obs_ak_is_21, ak21_is),
        p_ak5_oos  = sim_pvalue(obs_ak_oos_5, ak5_oos),
        p_ak21_oos = sim_pvalue(obs_ak_oos_21, ak21_oos),
    );
    pv_joint_is  = mean([pv.p_kurt_is,  pv.p_lev_is,  pv.p_ak5_is,  pv.p_ak21_is]);
    pv_joint_oos = mean([pv.p_kurt_oos, pv.p_lev_oos, pv.p_ak5_oos, pv.p_ak21_oos]);
    pv_results[m] = merge(pv, (pv_joint_is=pv_joint_is, pv_joint_oos=pv_joint_oos));
    println("  $m pv̄ IS=$(round(pv_joint_is, digits=3)) pv̄ OoS=$(round(pv_joint_oos, digits=3))");
end

# --------------------------------------------------------------------------------------- #
# A4 / A5 TSTR + Strategy (reuse helpers from run_track_a_utility.jl for SM rows)
# --------------------------------------------------------------------------------------- #
println("\n[A4] TSTR vol forecaster (SM rows only; real-trained benchmark reused)...");

function _build_har_features(r::AbstractVector)
    rv = r .^ 2;
    T = length(rv);
    idx = 23:T;
    X = Array{Float64,2}(undef, length(idx), 4);
    y = Vector{Float64}(undef, length(idx));
    for (i, t) in enumerate(idx)
        X[i, 1] = 1.0;
        X[i, 2] = rv[t-1];
        X[i, 3] = mean(rv[t-5:t-1]);
        X[i, 4] = mean(rv[t-22:t-1]);
        y[i]    = rv[t];
    end
    return X, y;
end
_har_fit(X, y) = X \ y;
function _har_predict(β, r::AbstractVector)
    X, y = _build_har_features(r);
    yhat = max.(X * β, 1e-10);
    return yhat, y;
end
_rmse(yhat, y) = sqrt(mean((yhat .- y) .^ 2));
function _qlike(yhat, y)
    yhat = max.(yhat, 1e-10); y = max.(y, 1e-10);
    return mean(y ./ yhat .- log.(y ./ yhat) .- 1.0);
end

X_is, y_is = _build_har_features(R_is);
β_real = _har_fit(X_is, y_is);
yhat_real, y_oos_targets = _har_predict(β_real, R_oos);
real_rmse  = _rmse(yhat_real, y_oos_targets);
real_qlike = _qlike(yhat_real, y_oos_targets);
println("  Real-trained HAR  OoS RMSE=$(round(real_rmse, digits=5)) QLIKE=$(round(real_qlike, digits=5))");

n_paths_use = 50;
tstr_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER_FULL
    s_is = archive[m].is;
    rs = vec(s_is[:, 1:min(n_paths_use, size(s_is, 2))]);
    Xs, ys = _build_har_features(rs);
    β_s = _har_fit(Xs, ys);
    yhat_s, _ = _har_predict(β_s, R_oos);
    tstr_results[m] = (
        rmse=_rmse(yhat_s, y_oos_targets),
        qlike=_qlike(yhat_s, y_oos_targets),
        β=β_s,
    );
end

println("\n[A5] Vol-target strategy on SM rows...");

const TARGET_ANNUAL_VOL = 0.15;
const W_MIN = 0.1; const W_MAX = 3.0;

function _run_strategy(yhat::AbstractVector, R::AbstractVector)
    σ̂ = sqrt.(max.(yhat, 1e-10));
    w = clamp.(TARGET_ANNUAL_VOL ./ σ̂, W_MIN, W_MAX);
    daily_pnl = w .* R .* DT;
    ann_ret = mean(daily_pnl) * 252;
    ann_vol = std(daily_pnl) * sqrt(252);
    sharpe = ann_vol > 0 ? ann_ret / ann_vol : 0.0;
    eq = cumsum(daily_pnl);
    peak = accumulate(max, eq);
    mdd = maximum(peak .- eq);
    turnover = sum(abs.(diff(w))) * 252 / length(w);
    return (sharpe=sharpe, ann_ret=ann_ret, ann_vol=ann_vol, mdd=mdd, turnover=turnover, mean_w=mean(w));
end

R_oos_target = R_oos[23:end];
strat_real = _run_strategy(yhat_real, R_oos_target);
strat_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER_FULL
    yhat_s, _ = _har_predict(tstr_results[m].β, R_oos);
    strat_results[m] = _run_strategy(yhat_s, R_oos_target);
end

# --------------------------------------------------------------------------------------- #
# A8 VaR LR tests (on SM rows)
# --------------------------------------------------------------------------------------- #
println("\n[A8] Kupiec + Christoffersen VaR LR tests on SM rows...");
var_lr_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER_FULL
    v01 = quantile(vec(archive[m].is), 0.01);
    v05 = quantile(vec(archive[m].is), 0.05);
    br_01 = R_oos .<= v01;
    br_05 = R_oos .<= v05;
    k01 = kupiec_lr(br_01, 0.01); k05 = kupiec_lr(br_05, 0.05);
    c01 = christoffersen_lr(br_01); c05 = christoffersen_lr(br_05);
    cc01 = christoffersen_cc(br_01, 0.01); cc05 = christoffersen_cc(br_05, 0.05);
    var_lr_results[m] = (v01=v01, v05=v05, k01=k01, k05=k05, c01=c01, c05=c05, cc01=cc01, cc05=cc05);
end

# --------------------------------------------------------------------------------------- #
# Write outputs
# --------------------------------------------------------------------------------------- #
println("\n[output] Writing Track C1 tables...");

# Sojourn summary
open(joinpath(TRACK_C_DIR, "sojourn_summary.txt"), "w") do io
    println(io, "="^100);
    println(io, "Track C1 semi-Markov CHMM sojourn summary (SPY, K=$K_MAIN, seed=$SEED)");
    println(io, "="^100);
    for tag in ("SM-CHMM-N", "SM-CHMM-t", "SM-CHMM-L")
        m = models[tag];
        println(io, "");
        println(io, "$tag");
        println(io, "-"^40);
        println(io, "state | μ         | φ       | σ/b       | ν       | sojourn_family | E[D]");
        for k in m.states
            fam = m.sojourn_family[k];
            mean_d = mean(m.sojourn[k]);
            println(io, "$(rpad(k, 5)) | $(rpad(round(m.emission_mu[k], digits=4), 9)) | $(rpad(round(m.emission_phi[k], digits=3), 7)) | $(rpad(round(m.emission_sigma[k], digits=4), 9)) | $(rpad(round(m.emission_nu[k], digits=2), 7)) | $(rpad(fam, 14)) | $(round(mean_d, digits=2))");
        end
    end
end

# Table 4 extended with SM rows
open(joinpath(TRACK_C_DIR, "Table-4-Extended-Metrics-C1.txt"), "w") do io
    println(io, "="^130);
    println(io, "TABLE 4 (Track A + C1). MMD, signature-MMD, discriminator AUC; SM-CHMM rows appended.");
    println(io, "="^130);
    println(io, "");
    println(io, "Setup   : SPY daily log excess growth; IS n=$n_is, OoS n=$n_oos; seed=$SEED; N_paths=$N_PATHS.");
    println(io, "Windows : length W=$WINDOW_LEN days; observed windows IS $n_obs_win_is, OoS $n_obs_win_oos.");
    println(io, "MMD     : RBF kernel, unbiased estimator, bandwidth by median heuristic on pooled sample.");
    println(io, "sig-MMD : depth-$SIG_DEPTH truncated signature of time-augmented 2D path.");
    println(io, "Disc AUC: logistic regression on 10 hand-crafted window features, 5-fold CV.");
    println(io, "Reading : lower MMD / sig-MMD is better; disc AUC near 0.50 means indistinguishable from real.");
    println(io, "-"^130);
    println(io, rpad("Model", 13), " | ",
                rpad("MMD IS",   11), " | ", rpad("MMD OoS",  11), " | ",
                rpad("sig IS",   11), " | ", rpad("sig OoS",  11), " | ",
                rpad("AUC IS",   14), " | ", rpad("AUC OoS",  14));
    println(io, "-"^130);
    for m in MODEL_ORDER_FULL
        r = mmd_results[m];
        auc_is_s  = "$(round(r.auc_is,  digits=3)) ± $(round(r.auc_is_std,  digits=3))";
        auc_oos_s = "$(round(r.auc_oos, digits=3)) ± $(round(r.auc_oos_std, digits=3))";
        println(io, rpad(m, 13), " | ",
                    rpad(round(r.mmd_is,  digits=5), 11), " | ",
                    rpad(round(r.mmd_oos, digits=5), 11), " | ",
                    rpad(round(r.sig_is,  digits=5), 11), " | ",
                    rpad(round(r.sig_oos, digits=5), 11), " | ",
                    rpad(auc_is_s, 14), " | ", rpad(auc_oos_s, 14));
    end
    println(io, "="^130);
end

open(joinpath(TRACK_C_DIR, "leverage_effect_c1.txt"), "w") do io
    println(io, "Track C1 leverage effect (SM-CHMM rows appended)");
    println(io, "Observed IS  avg_neg=$(round(lev_obs_is.avg_neg, digits=4))  asym=$(round(lev_obs_is.asymmetry, digits=5))");
    println(io, "Observed OoS avg_neg=$(round(lev_obs_oos.avg_neg, digits=4))  asym=$(round(lev_obs_oos.asymmetry, digits=5))");
    println(io, "");
    println(io, rpad("Model", 13), " | avg_neg IS  | asym IS   | pv avg IS | avg_neg OoS | asym OoS  | pv avg OoS");
    for m in MODEL_ORDER_FULL
        r = lev_results[m];
        println(io, rpad(m, 13), " | ",
                    rpad(round(r.med_is, digits=4), 10), " | ",
                    rpad(round(r.med_asym_is, digits=4), 9), " | ",
                    rpad(round(r.p_avg_is, digits=3), 9), " | ",
                    rpad(round(r.med_oos, digits=4), 11), " | ",
                    rpad(round(r.med_asym_oos, digits=4), 9), " | ",
                    rpad(round(r.p_avg_oos, digits=3), 9));
    end
end

open(joinpath(TRACK_C_DIR, "aggregational_kurtosis_c1.txt"), "w") do io
    println(io, "Track C1 aggregational kurtosis at horizons $HORIZONS_AG (SM-CHMM rows appended)");
    println(io, "Observed IS  : ", join(["h=$(h) k=$(round(agg_obs_is[h], digits=2))" for h in HORIZONS_AG], "   "));
    println(io, "Observed OoS : ", join(["h=$(h) k=$(round(agg_obs_oos[h], digits=2))" for h in HORIZONS_AG], "   "));
    println(io, "");
    println(io, "IS median sim kurtosis over $N_PATHS paths");
    println(io, rpad("Model", 13), " | ", join([rpad("h=$h", 8) for h in HORIZONS_AG], " | "));
    for m in MODEL_ORDER_FULL
        per_h = agg_results[m][:is];
        row = [rpad(m, 13)];
        for h in HORIZONS_AG; push!(row, rpad(round(median(per_h[h]), digits=2), 8)); end
        println(io, join(row, " | "));
    end
    println(io, "");
    println(io, "OoS median sim kurtosis");
    println(io, rpad("Model", 13), " | ", join([rpad("h=$h", 8) for h in HORIZONS_AG], " | "));
    for m in MODEL_ORDER_FULL
        per_h = agg_results[m][:oos];
        row = [rpad(m, 13)];
        for h in HORIZONS_AG; push!(row, rpad(round(median(per_h[h]), digits=2), 8)); end
        println(io, join(row, " | "));
    end
end

open(joinpath(TRACK_C_DIR, "sim_pvalues_c1.txt"), "w") do io
    println(io, "Track C1 p-values (SM-CHMM rows appended)");
    println(io, rpad("Model", 13), " | p_kurt  | p_acf   | p_lev   | p_ak5   | p_ak21  | pv̄ IS   | pv̄ OoS");
    for m in MODEL_ORDER_FULL
        pv = pv_results[m];
        println(io, rpad(m, 13), " | ",
                    rpad(round(pv.p_kurt_is, digits=3), 7), " | ",
                    rpad(round(pv.p_acf_is, digits=3), 7), " | ",
                    rpad(round(pv.p_lev_is, digits=3), 7), " | ",
                    rpad(round(pv.p_ak5_is, digits=3), 7), " | ",
                    rpad(round(pv.p_ak21_is, digits=3), 7), " | ",
                    rpad(round(pv.pv_joint_is, digits=3), 7), " | ",
                    rpad(round(pv.pv_joint_oos, digits=3), 7));
    end
end

open(joinpath(TRACK_C_DIR, "tstr_vol_forecaster_c1.txt"), "w") do io
    println(io, "Track C1 TSTR HAR (SM-CHMM rows appended)");
    println(io, "Real-trained: RMSE=$(round(real_rmse, digits=5)) QLIKE=$(round(real_qlike, digits=5))");
    println(io, rpad("Model", 13), " | RMSE       | RMSE/real  | QLIKE      | QLIKE/real");
    for m in MODEL_ORDER_FULL
        r = tstr_results[m];
        println(io, rpad(m, 13), " | ",
                    rpad(round(r.rmse, digits=5), 10), " | ",
                    rpad(round(r.rmse/real_rmse, digits=3), 10), " | ",
                    rpad(round(r.qlike, digits=5), 10), " | ",
                    rpad(round(r.qlike/real_qlike, digits=3), 10));
    end
end

open(joinpath(TRACK_C_DIR, "vol_target_strategy_c1.txt"), "w") do io
    println(io, "Track C1 vol-target strategy (SM-CHMM rows appended)");
    println(io, rpad("Train", 13), " | Sharpe | AnnRet  | AnnVol  | MDD     | Turnover | Mean w");
    println(io, rpad("Real IS", 13), " | ",
                rpad(round(strat_real.sharpe, digits=2), 6), " | ",
                rpad(round(strat_real.ann_ret, digits=3), 7), " | ",
                rpad(round(strat_real.ann_vol, digits=3), 7), " | ",
                rpad(round(strat_real.mdd, digits=4), 7), " | ",
                rpad(round(strat_real.turnover, digits=2), 8), " | ",
                rpad(round(strat_real.mean_w, digits=2), 6));
    for m in MODEL_ORDER_FULL
        r = strat_results[m];
        println(io, rpad(m, 13), " | ",
                    rpad(round(r.sharpe, digits=2), 6), " | ",
                    rpad(round(r.ann_ret, digits=3), 7), " | ",
                    rpad(round(r.ann_vol, digits=3), 7), " | ",
                    rpad(round(r.mdd, digits=4), 7), " | ",
                    rpad(round(r.turnover, digits=2), 8), " | ",
                    rpad(round(r.mean_w, digits=2), 6));
    end
end

open(joinpath(TRACK_C_DIR, "VaR_LR_tests_c1.txt"), "w") do io
    println(io, "Track C1 VaR LR tests (SM-CHMM rows appended). α ∈ {0.01, 0.05}, χ²(1) crit 3.84.");
    println(io, rpad("Model", 13), " | VaR01   | br%01 | LR_uc01 | p_uc01 | LR_ind01 | LR_cc01 | VaR05   | br%05 | LR_uc05 | p_uc05 | LR_ind05 | LR_cc05");
    for m in MODEL_ORDER_FULL
        r = var_lr_results[m];
        println(io, rpad(m, 13), " | ",
                    rpad(round(r.v01, digits=3), 7), " | ",
                    rpad(round(100*r.k01.breach_rate, digits=1), 5), " | ",
                    rpad(round(r.k01.LR, digits=2), 7), " | ",
                    rpad(round(r.k01.pvalue, digits=3), 6), " | ",
                    rpad(round(r.c01.LR, digits=2), 8), " | ",
                    rpad(round(r.cc01.LR, digits=2), 7), " | ",
                    rpad(round(r.v05, digits=3), 7), " | ",
                    rpad(round(100*r.k05.breach_rate, digits=1), 5), " | ",
                    rpad(round(r.k05.LR, digits=2), 7), " | ",
                    rpad(round(r.k05.pvalue, digits=3), 6), " | ",
                    rpad(round(r.c05.LR, digits=2), 8), " | ",
                    rpad(round(r.cc05.LR, digits=2), 7));
    end
end

# Summary digest
open(joinpath(TRACK_C_DIR, "Track-C1-summary.txt"), "w") do io
    println(io, "Track C1 summary (semi-Markov port from CHMM-Vol-Model)");
    println(io, "="^80);
    println(io, "");
    println(io, "Setup : SPY K=$K_MAIN, plug-in estimator (Viterbi + per-state AR(1) + sojourn-family pick).");
    println(io, "");
    println(io, "Sojourn families chosen per state:");
    for tag in ("SM-CHMM-N", "SM-CHMM-t", "SM-CHMM-L")
        m = models[tag];
        pareto_count = count(==(:pareto), values(m.sojourn_family));
        nb_count = count(==(:nb), values(m.sojourn_family));
        geo_count = count(==(:geometric), values(m.sojourn_family));
        println(io, "  $tag  pareto=$pareto_count  nb=$nb_count  geometric=$geo_count");
    end
    println(io, "");
    println(io, "Headline SM vs flat (IS):");
    for tag in ("CHMM-N", "CHMM-t", "CHMM-L")
        sm_tag = "SM-$tag";
        mmd_flat = mmd_results[tag].mmd_is;
        mmd_sm = mmd_results[sm_tag].mmd_is;
        auc_flat = mmd_results[tag].auc_is;
        auc_sm = mmd_results[sm_tag].auc_is;
        pv_flat = pv_results[tag].pv_joint_oos;
        pv_sm = pv_results[sm_tag].pv_joint_oos;
        var01_flat = var_lr_results[tag].k01.LR;
        var01_sm = var_lr_results[sm_tag].k01.LR;
        var05_flat = var_lr_results[tag].k05.LR;
        var05_sm = var_lr_results[sm_tag].k05.LR;
        println(io, "  $tag -> $sm_tag");
        println(io, "     MMD IS:       $(round(mmd_flat, digits=5))  -> $(round(mmd_sm, digits=5))");
        println(io, "     disc AUC IS:  $(round(auc_flat, digits=3))  -> $(round(auc_sm, digits=3))");
        println(io, "     pv̄ OoS:       $(round(pv_flat, digits=3))  -> $(round(pv_sm, digits=3))");
        println(io, "     LR_uc 1% VaR: $(round(var01_flat, digits=2))  -> $(round(var01_sm, digits=2))");
        println(io, "     LR_uc 5% VaR: $(round(var05_flat, digits=2))  -> $(round(var05_sm, digits=2))");
    end
end

println("\n" * "="^72);
println("  Track C1 complete.");
println("  Results: $TRACK_C_DIR");
println("="^72);

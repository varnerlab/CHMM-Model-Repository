# ========================================================================================= #
# run_track_c3_external_covariates.jl
#
# Track C3 (external-covariate pass): time-varying transition matrix driven by
# lagged realized volatility plus lagged VIX level. This extends the first-pass
# lagged-RV script without changing the underlying flat CHMM emission families.
#
#   P(s_t = j | s_{t-1} = i, x_t) = softmax(a_ij + b_ij' x_t),
#
# where x_t contains standardized lagged RV(20) and standardized lagged log(VIX).
#
# Output:
#   results/track_c3/VaR_external_covariate_transition_LR_tests.txt
#   results/track_c3/Track-C3-external-summary.txt
# ========================================================================================= #

const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."));
using Pkg; Pkg.activate(_PROJECT_ROOT);
include(joinpath(_PROJECT_ROOT, "Include.jl"));

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1 / 252;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const RV_WINDOW = 20;
const MAX_SOFTMAX_ITER = 500;
const SOFTMAX_LR = 0.05;
const SOFTMAX_RIDGE = 1e-3;
const VIX_HISTORY_CSV = joinpath(dirname(_ROOT), "CHMM-Vol-Model", "data", "VIX_History.csv");

const TRACK_C3_DIR = joinpath(_ROOT, "results", "track_c3");
mkpath(TRACK_C3_DIR);

println("="^72)
println("  Track C3: time-varying transitions via lagged RV + VIX")
println("  Seed $SEED, K=$K_MAIN, RV window=$RV_WINDOW")
println("="^72)

println("\n[data] Loading SPY IS + OoS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days
        dataset[t] = data;
    end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is = Vector{Float64}(all_R[:, idx_spy]);
spy_train_df = dataset[TICKER];
dates_is = Date.(spy_train_df.timestamp[2:end]);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
spy_oos_df = oos_dataset[TICKER];
dates_oos = Date.(spy_oos_df.timestamp[2:end]);

n_is = length(R_is);
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

function lagged_realized_vol(r::AbstractVector; window::Int=20)
    n = length(r);
    out = Vector{Float64}(undef, n);
    base = sqrt(mean(r .^ 2));
    out[1] = base;
    for t in 2:n
        lo = max(1, t - window);
        out[t] = sqrt(mean(r[lo:(t - 1)] .^ 2));
    end
    return out;
end

function zscore_with_train(train::AbstractVector, test::AbstractVector)
    μ = mean(train);
    σ = std(train);
    if !(isfinite(σ) && σ > 0)
        return train .* 0.0, test .* 0.0;
    end
    return (train .- μ) ./ σ, (test .- μ) ./ σ;
end

function _softmax(logits::AbstractVector)
    m = maximum(logits);
    z = exp.(logits .- m);
    return z ./ sum(z);
end

function fit_transition_logits(decoded::Vector{Int}, X::Matrix{Float64}, K::Int;
                               max_iter::Int=MAX_SOFTMAX_ITER,
                               lr::Float64=SOFTMAX_LR,
                               ridge::Float64=SOFTMAX_RIDGE)
    p = size(X, 2);
    rows = Vector{NamedTuple}(undef, K);
    for i in 1:K
        idx = findall(decoded[1:(end - 1)] .== i);
        y = decoded[idx .+ 1];
        Xi = X[idx .+ 1, :];

        counts = fill(1.0, K);
        for yy in y
            counts[yy] += 1.0;
        end
        p_fallback = counts ./ sum(counts);

        if length(y) < 3K
            rows[i] = (alpha=zeros(K - 1), beta=zeros(K - 1, p), fallback=p_fallback, n_obs=length(y));
            continue;
        end

        α = [log(p_fallback[j] / p_fallback[K]) for j in 1:(K - 1)];
        β = zeros(K - 1, p);

        scale = max(length(y), 1);
        for _ in 1:max_iter
            gα = zeros(K - 1);
            gβ = zeros(K - 1, p);
            for t in eachindex(y)
                logits = vcat(α .+ β * vec(Xi[t, :]), 0.0);
                probs = _softmax(logits);
                xt = vec(Xi[t, :]);
                for j in 1:(K - 1)
                    target = y[t] == j ? 1.0 : 0.0;
                    err = target - probs[j];
                    gα[j] += err;
                    gβ[j, :] .+= err .* xt;
                end
            end
            gα .-= ridge .* α;
            gβ .-= ridge .* β;
            α .+= (lr / scale) .* gα;
            β .+= (lr / scale) .* gβ;
        end
        rows[i] = (alpha=α, beta=β, fallback=p_fallback, n_obs=length(y));
    end
    return rows;
end

function transition_row_tv(rows::Vector{<:NamedTuple}, prev_state::Int, x_t::Vector{Float64}, K::Int)
    row = rows[prev_state];
    if row.n_obs < 3K
        return row.fallback;
    end
    logits = vcat(row.alpha .+ row.beta * x_t, 0.0);
    return _softmax(logits);
end

function emission_grid(model; α_lo::Float64=1e-4, α_hi::Float64=0.9999, n_grid::Int=2500)
    K = length(model.states);
    q_lo = minimum([quantile(model.emission[k], α_lo) for k in 1:K]);
    q_hi = maximum([quantile(model.emission[k], α_hi) for k in 1:K]);
    grid = collect(range(q_lo, q_hi; length=n_grid));
    cdf_mat = zeros(K, n_grid);
    for k in 1:K
        for g in 1:n_grid
            cdf_mat[k, g] = cdf(model.emission[k], grid[g]);
        end
    end
    return grid, cdf_mat;
end

function mixture_var(grid::Vector{Float64}, cdf_mat::Matrix{Float64}, weights::Vector{Float64}, α::Float64)
    mix_cdf = vec(weights' * cdf_mat);
    idx = findfirst(>=(α), mix_cdf);
    return isnothing(idx) ? grid[end] : grid[idx];
end

function var_series_tv(model, rows, decoded_oos::Vector{Int}, X_oos::Matrix{Float64}, α::Float64)
    K = length(model.states);
    grid, cdf_mat = emission_grid(model);
    T = length(decoded_oos);
    out = Vector{Float64}(undef, T);
    for t in 1:T
        prev_state = t == 1 ? decoded_oos[1] : decoded_oos[t - 1];
        w = transition_row_tv(rows, prev_state, vec(X_oos[t, :]), K);
        out[t] = mixture_var(grid, cdf_mat, w, α);
    end
    return out;
end

function var_series_flat(model, decoded_oos::Vector{Int}, α::Float64)
    return [quantile(model.emission[s], α) for s in decoded_oos];
end

function load_vix_level_by_date(csv_path::String)
    raw = CSV.read(csv_path, DataFrame);
    rename!(raw, Symbol.(lowercase.(string.(names(raw)))));
    raw.date = Date.(raw.date, dateformat"mm/dd/yyyy");
    select!(raw, [:date, :close]);
    raw = raw[.!ismissing.(raw.close), :];
    raw.close = Float64.(raw.close);
    raw.log_vix = log.(raw.close);
    return Dict(Date(d) => v for (d, v) in zip(raw.date, raw.log_vix));
end

function lagged_vix_covariate(dates::Vector{Date}, vix_by_date::Dict{Date, Float64})
    values = Vector{Float64}(undef, length(dates));
    first_date = minimum(collect(keys(vix_by_date)));
    first_value = vix_by_date[first_date];
    for (i, d) in enumerate(dates)
        d_prev = d - Day(1);
        while !(d_prev in keys(vix_by_date)) && d_prev >= first_date
            d_prev -= Day(1);
        end
        values[i] = get(vix_by_date, d_prev, first_value);
    end
    return values;
end

println("\n[data] Loading VIX history from sibling CHMM-Vol-Model...");
isfile(VIX_HISTORY_CSV) || error("Missing VIX history CSV at $VIX_HISTORY_CSV");
vix_by_date = load_vix_level_by_date(VIX_HISTORY_CSV);

rv_is_raw = lagged_realized_vol(R_is; window=RV_WINDOW);
rv_oos_raw = lagged_realized_vol(R_oos; window=RV_WINDOW);
vix_is_raw = lagged_vix_covariate(dates_is, vix_by_date);
vix_oos_raw = lagged_vix_covariate(dates_oos, vix_by_date);

rv_is, rv_oos = zscore_with_train(rv_is_raw, rv_oos_raw);
vix_is, vix_oos = zscore_with_train(vix_is_raw, vix_oos_raw);
X_is = hcat(rv_is, vix_is);
X_oos = hcat(rv_oos, vix_oos);

println("  VIX coverage: $(minimum(dates_is)) to $(maximum(dates_oos))");

println("\n[fit] Fitting flat CHMM-N / -t / -L...");
Random.seed!(SEED + 31);
flat_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
Random.seed!(SEED + 32);
flat_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
Random.seed!(SEED + 33);
flat_l = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));

println("\n[decode] Viterbi on IS + OoS...");
dec_is_n = viterbi(R_is, flat_n); dec_oos_n = viterbi(R_oos, flat_n);
dec_is_t = viterbi(R_is, flat_t); dec_oos_t = viterbi(R_oos, flat_t);
dec_is_l = viterbi(R_is, flat_l); dec_oos_l = viterbi(R_oos, flat_l);

println("\n[fit] Time-varying transition logits on lagged RV + VIX...");
rows_n = fit_transition_logits(dec_is_n, X_is, K_MAIN);
rows_t = fit_transition_logits(dec_is_t, X_is, K_MAIN);
rows_l = fit_transition_logits(dec_is_l, X_is, K_MAIN);

println("\n[eval] Computing flat-vs-external-covariate VaR series...");
MODELS = [
    ("CHMM-N flat", flat_n, dec_oos_n, nothing),
    ("CHMM-N XTV",  flat_n, dec_oos_n, rows_n),
    ("CHMM-t flat", flat_t, dec_oos_t, nothing),
    ("CHMM-t XTV",  flat_t, dec_oos_t, rows_t),
    ("CHMM-L flat", flat_l, dec_oos_l, nothing),
    ("CHMM-L XTV",  flat_l, dec_oos_l, rows_l),
];

results = Dict{String, NamedTuple}();
for (name, model, decoded, rows) in MODELS
    v01 = isnothing(rows) ? var_series_flat(model, decoded, 0.01) : var_series_tv(model, rows, decoded, X_oos, 0.01);
    v05 = isnothing(rows) ? var_series_flat(model, decoded, 0.05) : var_series_tv(model, rows, decoded, X_oos, 0.05);
    br01 = R_oos .<= v01; br05 = R_oos .<= v05;
    k01 = kupiec_lr(br01, 0.01); k05 = kupiec_lr(br05, 0.05);
    c01 = christoffersen_lr(br01); c05 = christoffersen_lr(br05);
    cc01 = christoffersen_cc(br01, 0.01); cc05 = christoffersen_cc(br05, 0.05);
    results[name] = (
        med_v01=median(v01), med_v05=median(v05),
        br_rate_01=k01.breach_rate, br_rate_05=k05.breach_rate,
        k01=k01, k05=k05, c01=c01, c05=c05, cc01=cc01, cc05=cc05,
    );
    println("  $name  br01=$(round(100 * k01.breach_rate, digits=2))% LRuc01=$(round(k01.LR, digits=2)) LRind01=$(round(c01.LR, digits=2)) | br05=$(round(100 * k05.breach_rate, digits=2))% LRuc05=$(round(k05.LR, digits=2)) LRind05=$(round(c05.LR, digits=2))");
end

open(joinpath(TRACK_C3_DIR, "VaR_external_covariate_transition_LR_tests.txt"), "w") do io
    println(io, "="^154);
    println(io, "Track C3 external-covariate transitions. Lagged RV + lagged VIX logistic transition VaR. Kupiec + Christoffersen LR tests.");
    println(io, "="^154);
    println(io, "");
    println(io, "Setup  : P(s_t = j | s_{t-1} = i, x_t) = softmax(a_ij + b_ij' x_t).");
    println(io, "         x_t = [ standardized lagged RV($RV_WINDOW), standardized lagged log(VIX) ].");
    println(io, "         VIX source = sibling CHMM-Vol-Model/data/VIX_History.csv, aligned by previous available trading day.");
    println(io, "Baseline: flat row uses the C3a per-state emission quantile at decoded state k_t.");
    println(io, "Target : α ∈ {0.01, 0.05}. LR_uc ~ χ²(1) (crit 3.84). LR_ind ~ χ²(1) (crit 3.84).");
    println(io, "");
    println(io, rpad("Model", 12), " | ",
                rpad("medVaR01", 9), " | ", rpad("br%01", 6), " | ",
                rpad("LR_uc01", 7), " | ", rpad("LR_ind01", 8), " | ",
                rpad("medVaR05", 9), " | ", rpad("br%05", 6), " | ",
                rpad("LR_uc05", 7), " | ", rpad("LR_ind05", 8));
    println(io, "-"^154);
    for (name, _, _, _) in MODELS
        r = results[name];
        println(io, rpad(name, 12), " | ",
                    rpad(round(r.med_v01, digits=3), 9), " | ",
                    rpad(round(100 * r.br_rate_01, digits=1), 6), " | ",
                    rpad(round(r.k01.LR, digits=2), 7), " | ",
                    rpad(round(r.c01.LR, digits=2), 8), " | ",
                    rpad(round(r.med_v05, digits=3), 9), " | ",
                    rpad(round(100 * r.br_rate_05, digits=1), 6), " | ",
                    rpad(round(r.k05.LR, digits=2), 7), " | ",
                    rpad(round(r.c05.LR, digits=2), 8));
    end
    println(io, "="^154);
end

function fmt_metric(x; digits::Int=2)
    return string(round(x, digits=digits));
end

open(joinpath(TRACK_C3_DIR, "Track-C3-external-summary.txt"), "w") do io
    println(io, "Track C3 external-covariate summary: time-varying transitions via lagged RV + VIX");
    println(io, "===========================================================================");
    println(io, "Covariates: standardized lagged RV($RV_WINDOW) and lagged log(VIX).");
    println(io, "VIX input : sibling CHMM-Vol-Model/data/VIX_History.csv.");
    println(io, "");
    for family in ("N", "t", "L")
        flat = results["CHMM-$family flat"];
        xtv = results["CHMM-$family XTV"];
        println(io, "CHMM-$family:");
        println(io, "  flat  : LR_uc01=$(fmt_metric(flat.k01.LR)) LR_ind01=$(fmt_metric(flat.c01.LR)) | LR_uc05=$(fmt_metric(flat.k05.LR)) LR_ind05=$(fmt_metric(flat.c05.LR))");
        println(io, "  XTV   : LR_uc01=$(fmt_metric(xtv.k01.LR)) LR_ind01=$(fmt_metric(xtv.c01.LR)) | LR_uc05=$(fmt_metric(xtv.k05.LR)) LR_ind05=$(fmt_metric(xtv.c05.LR))");
    end
    println(io, "");
    println(io, "Interpretation: use this file to compare whether external VIX conditioning");
    println(io, "improves on the flat C3a row or the lagged-RV-only TVT first pass.");
end

println("\n[done] Wrote:");
println("  ", joinpath(TRACK_C3_DIR, "VaR_external_covariate_transition_LR_tests.txt"));
println("  ", joinpath(TRACK_C3_DIR, "Track-C3-external-summary.txt"));

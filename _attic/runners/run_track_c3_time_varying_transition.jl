# ========================================================================================= #
# run_track_c3_time_varying_transition.jl
#
# Track C3 (first pass): time-varying transition matrix driven by lagged realized
# volatility. Rather than using the flat CHMM transition row T[i, :], fit
#
#   P(s_t = j | s_{t-1} = i, x_t) = softmax(a_{ij} + b_{ij} x_t),
#
# where x_t is a standardized lagged 20-day realized-volatility covariate.
#
# This is a light-touch post-fit upgrade on top of the existing flat CHMMs:
#   1. fit flat CHMM-N / -t / -L on SPY in-sample returns,
#   2. Viterbi-decode the in-sample state path,
#   3. fit per-origin-state multinomial logits for next-state probabilities,
#   4. on OoS, compute VaR_t from the one-step predictive mixture emission.
#
# Output:
#   results/track_c3/VaR_time_varying_transition_LR_tests.txt
#   results/track_c3/Track-C3-summary.txt
# ========================================================================================= #

const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."));
using Pkg; Pkg.activate(_PROJECT_ROOT);
include(joinpath(_PROJECT_ROOT, "Include.jl"));

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const RV_WINDOW = 20;
const MAX_SOFTMAX_ITER = 400;
const SOFTMAX_LR = 0.05;
const SOFTMAX_RIDGE = 1e-3;

const TRACK_C3_DIR = joinpath(_ROOT, "results", "track_c3");
mkpath(TRACK_C3_DIR);

println("="^72)
println("  Track C3: time-varying transitions via lagged realized volatility")
println("  Seed $SEED, K=$K_MAIN, RV window=$RV_WINDOW")
println("="^72)

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

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
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
        out[t] = sqrt(mean(r[lo:(t-1)] .^ 2));
    end
    μ = mean(out);
    σ = std(out);
    return σ > 0 ? (out .- μ) ./ σ : out .* 0.0;
end

function _softmax(logits::AbstractVector)
    m = maximum(logits);
    z = exp.(logits .- m);
    return z ./ sum(z);
end

function fit_transition_logits(decoded::Vector{Int}, x::Vector{Float64}, K::Int;
                               max_iter::Int=MAX_SOFTMAX_ITER,
                               lr::Float64=SOFTMAX_LR,
                               ridge::Float64=SOFTMAX_RIDGE)
    rows = Vector{NamedTuple}(undef, K);
    for i in 1:K
        idx = findall(decoded[1:end-1] .== i);
        y = decoded[idx .+ 1];
        xi = x[idx .+ 1];

        # Empirical row fallback.
        counts = fill(1.0, K);
        for yy in y
            counts[yy] += 1.0;
        end
        p_fallback = counts ./ sum(counts);

        # If the state is too rare, keep the flat row.
        if length(y) < 3K
            rows[i] = (alpha=zeros(K-1), beta=zeros(K-1), fallback=p_fallback, n_obs=length(y));
            continue;
        end

        α = [log(p_fallback[j] / p_fallback[K]) for j in 1:(K-1)];
        β = zeros(K-1);

        scale = max(length(y), 1);
        for _ in 1:max_iter
            gα = zeros(K-1);
            gβ = zeros(K-1);
            for (t, yy) in enumerate(y)
                logits = vcat(α .+ β .* xi[t], 0.0);
                p = _softmax(logits);
                for j in 1:(K-1)
                    target = yy == j ? 1.0 : 0.0;
                    gα[j] += target - p[j];
                    gβ[j] += (target - p[j]) * xi[t];
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

function transition_row_tv(rows::Vector{<:NamedTuple}, prev_state::Int, x_t::Float64, K::Int)
    row = rows[prev_state];
    if row.n_obs < 3K
        return row.fallback;
    end
    logits = vcat(row.alpha .+ row.beta .* x_t, 0.0);
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

function var_series_tv(model, rows, decoded_oos::Vector{Int}, x_oos::Vector{Float64}, α::Float64)
    K = length(model.states);
    grid, cdf_mat = emission_grid(model);
    T = length(decoded_oos);
    out = Vector{Float64}(undef, T);
    for t in 1:T
        prev_state = t == 1 ? decoded_oos[1] : decoded_oos[t-1];
        w = transition_row_tv(rows, prev_state, x_oos[t], K);
        out[t] = mixture_var(grid, cdf_mat, w, α);
    end
    return out;
end

function var_series_flat(model, decoded_oos::Vector{Int}, α::Float64)
    return [quantile(model.emission[s], α) for s in decoded_oos];
end

println("\n[fit] Fitting flat CHMM-N / -t / -L...");
Random.seed!(SEED + 21);
flat_n = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
Random.seed!(SEED + 22);
flat_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
Random.seed!(SEED + 23);
flat_l = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));

println("\n[decode] Viterbi on IS + OoS...");
dec_is_n = viterbi(R_is, flat_n); dec_oos_n = viterbi(R_oos, flat_n);
dec_is_t = viterbi(R_is, flat_t); dec_oos_t = viterbi(R_oos, flat_t);
dec_is_l = viterbi(R_is, flat_l); dec_oos_l = viterbi(R_oos, flat_l);

x_is = lagged_realized_vol(R_is; window=RV_WINDOW);
x_oos = lagged_realized_vol(R_oos; window=RV_WINDOW);

println("\n[fit] Time-varying transition logits on lagged realized vol...");
rows_n = fit_transition_logits(dec_is_n, x_is, K_MAIN);
rows_t = fit_transition_logits(dec_is_t, x_is, K_MAIN);
rows_l = fit_transition_logits(dec_is_l, x_is, K_MAIN);

println("\n[eval] Computing flat-vs-TV VaR series...");
MODELS = [
    ("CHMM-N flat", flat_n, dec_oos_n, nothing),
    ("CHMM-N TVT",  flat_n, dec_oos_n, rows_n),
    ("CHMM-t flat", flat_t, dec_oos_t, nothing),
    ("CHMM-t TVT",  flat_t, dec_oos_t, rows_t),
    ("CHMM-L flat", flat_l, dec_oos_l, nothing),
    ("CHMM-L TVT",  flat_l, dec_oos_l, rows_l),
];

results = Dict{String, NamedTuple}();
for (name, model, decoded, rows) in MODELS
    v01 = isnothing(rows) ? var_series_flat(model, decoded, 0.01) : var_series_tv(model, rows, decoded, x_oos, 0.01);
    v05 = isnothing(rows) ? var_series_flat(model, decoded, 0.05) : var_series_tv(model, rows, decoded, x_oos, 0.05);
    br01 = R_oos .<= v01; br05 = R_oos .<= v05;
    k01 = kupiec_lr(br01, 0.01); k05 = kupiec_lr(br05, 0.05);
    c01 = christoffersen_lr(br01); c05 = christoffersen_lr(br05);
    cc01 = christoffersen_cc(br01, 0.01); cc05 = christoffersen_cc(br05, 0.05);
    results[name] = (
        med_v01=median(v01), med_v05=median(v05),
        br_rate_01=k01.breach_rate, br_rate_05=k05.breach_rate,
        k01=k01, k05=k05, c01=c01, c05=c05, cc01=cc01, cc05=cc05,
    );
    println("  $name  br01=$(round(100*k01.breach_rate, digits=2))% LRuc01=$(round(k01.LR, digits=2)) LRind01=$(round(c01.LR, digits=2)) | br05=$(round(100*k05.breach_rate, digits=2))% LRuc05=$(round(k05.LR, digits=2)) LRind05=$(round(c05.LR, digits=2))");
end

open(joinpath(TRACK_C3_DIR, "VaR_time_varying_transition_LR_tests.txt"), "w") do io
    println(io, "="^150);
    println(io, "Track C3. Time-varying transition VaR via lagged realized volatility. Kupiec + Christoffersen LR tests.");
    println(io, "="^150);
    println(io, "");
    println(io, "Setup  : P(s_t = j | s_{t-1} = i, x_t) = softmax(a_ij + b_ij x_t), x_t = lagged standardized RV($RV_WINDOW).");
    println(io, "         VaR_t(α) is the α-quantile of the one-step predictive emission mixture under that row.");
    println(io, "Baseline: flat row uses the C3a per-state emission quantile at decoded state k_t.");
    println(io, "Target : α ∈ {0.01, 0.05}. LR_uc ~ χ²(1) (crit 3.84). LR_ind ~ χ²(1) (crit 3.84).");
    println(io, "");
    println(io, rpad("Model",      12), " | ",
                rpad("medVaR01",   9), " | ", rpad("br%01",      6), " | ",
                rpad("LR_uc01",    7), " | ", rpad("LR_ind01",   8), " | ",
                rpad("medVaR05",   9), " | ", rpad("br%05",      6), " | ",
                rpad("LR_uc05",    7), " | ", rpad("LR_ind05",   8));
    println(io, "-"^150);
    for (name, _, _, _) in MODELS
        r = results[name];
        println(io, rpad(name, 12), " | ",
                    rpad(round(r.med_v01, digits=3),      9), " | ",
                    rpad(round(100*r.br_rate_01, digits=1), 6), " | ",
                    rpad(round(r.k01.LR, digits=2),       7), " | ",
                    rpad(round(r.c01.LR, digits=2),       8), " | ",
                    rpad(round(r.med_v05, digits=3),      9), " | ",
                    rpad(round(100*r.br_rate_05, digits=1), 6), " | ",
                    rpad(round(r.k05.LR, digits=2),       7), " | ",
                    rpad(round(r.c05.LR, digits=2),       8));
    end
    println(io, "="^150);
end

open(joinpath(TRACK_C3_DIR, "Track-C3-summary.txt"), "w") do io
    println(io, "Track C3 summary: time-varying transitions via lagged realized volatility")
    println(io, "="^80);
    println(io, "");
    println(io, "Compare each TVT row against the matching flat C3a row.");
    println(io, "");
    for family in ("CHMM-N", "CHMM-t", "CHMM-L")
        rf = results["$family flat"];
        rt = results["$family TVT"];
        println(io, "$family:");
        println(io, "  1 % VaR  flat LR_uc=$(round(rf.k01.LR, digits=2)) LR_ind=$(round(rf.c01.LR, digits=2))  |  TVT LR_uc=$(round(rt.k01.LR, digits=2)) LR_ind=$(round(rt.c01.LR, digits=2))");
        println(io, "  5 % VaR  flat LR_uc=$(round(rf.k05.LR, digits=2)) LR_ind=$(round(rf.c05.LR, digits=2))  |  TVT LR_uc=$(round(rt.k05.LR, digits=2)) LR_ind=$(round(rt.c05.LR, digits=2))");
        println(io, "");
    end
    println(io, "Interpretation: if TVT improves LR_ind without materially worsening LR_uc,");
    println(io, "it supports the C3 thesis that transition dynamics, not emission shape alone,");
    println(io, "carry the breach-clustering signal.");
end

println("\n" * "="^72);
println("  Track C3 complete.");
println("  Results: $TRACK_C3_DIR");
println("="^72);

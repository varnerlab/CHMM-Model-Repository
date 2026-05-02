# ========================================================================================= #
# run_track_c3_conditional_var.jl
#
# Track C3a (lightweight first pass): regime-conditional VaR via Viterbi decode.
# Directly attacks the Christoffersen-independence failure surfaced by A8 / C1:
# instead of using a fixed α-quantile of the generator's pooled archive, at each OoS time t
# decode the current regime and take the α-quantile of that state's conditional emission.
#
# Expected behavior: when the OoS path enters a high-vol regime, conditional VaR widens,
# producing fewer breaches during crisis windows and more during calm windows. Under
# correct conditional VaR, breaches are i.i.d. Bernoulli(α) and Christoffersen independence
# passes.
#
# Coverage:
#   - CHMM-N, CHMM-t, CHMM-L (flat): use viterbi + per-state emission quantile
#   - SM-CHMM-N, SM-CHMM-t, SM-CHMM-L: use the matching flat CHMM's Viterbi on R_oos, then
#     take the SM's state-k conditional mean (μ + φ (y_{t-1} - μ)) + σ * residual-quantile(α)
#
# Output:
#   results/track_c3/VaR_conditional_LR_tests.txt
#   results/track_c3/Track-C3a-summary.txt
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
const N_PATHS   = 1000;
const K_MAIN    = 18;
const MAX_ITER  = 60;

const TRACK_C3_DIR = joinpath(_ROOT, "results", "track_c3");
mkpath(TRACK_C3_DIR);

println("="^72)
println("  Track C3a: regime-conditional VaR via Viterbi decode")
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
# Fit flat CHMM-N / -t / -L and SM-CHMM variants (reuse C1 cache if present)
# --------------------------------------------------------------------------------------- #
sm_cache = joinpath(_ROOT, "results", "track_c1", "sm_models.jld2");

println("\n[fit] Refitting flat CHMM-N / -t / -L and reloading SM-CHMMs...");
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

sm_n = nothing; sm_t = nothing; sm_l = nothing;
if isfile(sm_cache)
    println("  Loading SM models from cache...");
    sm_models = load(sm_cache);
    sm_n = sm_models["SM-CHMM-N"];
    sm_t = sm_models["SM-CHMM-t"];
    sm_l = sm_models["SM-CHMM-L"];
else
    println("  SM cache not found; refitting...");
    Random.seed!(SEED + 100); sm_n = fit_sm_chmm(R_is, K_MAIN, :gaussian;  max_iter=MAX_ITER);
    Random.seed!(SEED + 101); sm_t = fit_sm_chmm(R_is, K_MAIN, :student_t; max_iter=MAX_ITER);
    Random.seed!(SEED + 102); sm_l = fit_sm_chmm(R_is, K_MAIN, :laplace;   max_iter=MAX_ITER);
end

# Viterbi decode R_oos once per flat family (SM variants reuse these decodings)
println("\n[decode] Viterbi on R_oos per family...");
dec_n = viterbi(R_oos, flat_n);
dec_t = viterbi(R_oos, flat_t);
dec_l = viterbi(R_oos, flat_l);

# --------------------------------------------------------------------------------------- #
# Conditional-VaR computation
# --------------------------------------------------------------------------------------- #
println("\n[conditional VaR] Computing VaR_t from per-state emission quantile...");

# Standardised quantile for each residual family
function _residual_quantile(family::Symbol, α::Float64, ν::Float64)
    if family == :gaussian
        return quantile(Normal(0.0, 1.0), α);
    elseif family == :student_t
        return quantile(TDist(ν), α);
    elseif family == :laplace
        return quantile(Laplace(0.0, 1.0), α);
    else
        error("unknown family $family");
    end
end

function _var_series_flat(model, decoded::Vector{Int64}, α::Float64)
    return [quantile(model.emission[s], α) for s in decoded];
end

function _var_series_sm(m::MySemiMarkovContinuousHMM, decoded::Vector{Int64},
                        obs::AbstractVector, α::Float64)
    fam = m.emission_family;
    T = length(decoded);
    out = Vector{Float64}(undef, T);
    for t in 1:T
        s = decoded[t];
        # Current-state μ, φ, σ or b, ν
        μ = m.emission_mu[s]; φ = m.emission_phi[s]; σ = m.emission_sigma[s]; ν = m.emission_nu[s];
        y_prev = t == 1 ? μ : obs[t-1];
        m_cond = μ + φ * (y_prev - μ);
        q = _residual_quantile(fam, α, ν);
        out[t] = m_cond + σ * q;
    end
    return out;
end

# Build a dict of conditional-VaR sequences per model
cvar01 = Dict{String, Vector{Float64}}();
cvar05 = Dict{String, Vector{Float64}}();

cvar01["CHMM-N"]   = _var_series_flat(flat_n, dec_n, 0.01);
cvar05["CHMM-N"]   = _var_series_flat(flat_n, dec_n, 0.05);
cvar01["CHMM-t"]   = _var_series_flat(flat_t, dec_t, 0.01);
cvar05["CHMM-t"]   = _var_series_flat(flat_t, dec_t, 0.05);
cvar01["CHMM-L"]   = _var_series_flat(flat_l, dec_l, 0.01);
cvar05["CHMM-L"]   = _var_series_flat(flat_l, dec_l, 0.05);

# For SM variants we also try two naive approximations, clearly labelled:
#   SM-* (AR-res):  μ_k + φ_k (y_{t-1} - μ_k) + σ_k * residual-quantile(α)
#   SM-* (stat-σ):  μ_k + σ_k/sqrt(1-φ_k^2) * residual-quantile(α)   stationary within-state σ
# These use the flat-CHMM Viterbi decoding, which means the state label does not match the
# SM model. We report them as diagnostics only, not as publishable conditional VaR.
function _var_series_sm_stationary(m::MySemiMarkovContinuousHMM, decoded::Vector{Int64}, α::Float64)
    fam = m.emission_family;
    T = length(decoded);
    out = Vector{Float64}(undef, T);
    for t in 1:T
        s = decoded[t];
        μ = m.emission_mu[s]; φ = m.emission_phi[s]; σ = m.emission_sigma[s]; ν = m.emission_nu[s];
        σ_stat = σ / sqrt(max(1 - φ^2, 1e-6));
        q = _residual_quantile(fam, α, ν);
        out[t] = μ + σ_stat * q;
    end
    return out;
end

cvar01["SM-CHMM-N (AR-res)"] = _var_series_sm(sm_n, dec_n, R_oos, 0.01);
cvar05["SM-CHMM-N (AR-res)"] = _var_series_sm(sm_n, dec_n, R_oos, 0.05);
cvar01["SM-CHMM-t (AR-res)"] = _var_series_sm(sm_t, dec_t, R_oos, 0.01);
cvar05["SM-CHMM-t (AR-res)"] = _var_series_sm(sm_t, dec_t, R_oos, 0.05);
cvar01["SM-CHMM-L (AR-res)"] = _var_series_sm(sm_l, dec_l, R_oos, 0.01);
cvar05["SM-CHMM-L (AR-res)"] = _var_series_sm(sm_l, dec_l, R_oos, 0.05);
cvar01["SM-CHMM-N (stat-σ)"] = _var_series_sm_stationary(sm_n, dec_n, 0.01);
cvar05["SM-CHMM-N (stat-σ)"] = _var_series_sm_stationary(sm_n, dec_n, 0.05);
cvar01["SM-CHMM-t (stat-σ)"] = _var_series_sm_stationary(sm_t, dec_t, 0.01);
cvar05["SM-CHMM-t (stat-σ)"] = _var_series_sm_stationary(sm_t, dec_t, 0.05);
cvar01["SM-CHMM-L (stat-σ)"] = _var_series_sm_stationary(sm_l, dec_l, 0.01);
cvar05["SM-CHMM-L (stat-σ)"] = _var_series_sm_stationary(sm_l, dec_l, 0.05);

# Kupiec + Christoffersen on each
MODELS = ["CHMM-N", "CHMM-t", "CHMM-L",
          "SM-CHMM-N (AR-res)", "SM-CHMM-t (AR-res)", "SM-CHMM-L (AR-res)",
          "SM-CHMM-N (stat-σ)", "SM-CHMM-t (stat-σ)", "SM-CHMM-L (stat-σ)"];

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
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_C3_DIR, "VaR_conditional_LR_tests.txt"), "w") do io
    println(io, "="^150);
    println(io, "Track C3a. Regime-conditional VaR via Viterbi decode. Kupiec + Christoffersen LR tests.");
    println(io, "="^150);
    println(io, "");
    println(io, "Setup  : VaR_t(α) = (per-state emission α-quantile) at Viterbi-decoded state for R_oos[t].");
    println(io, "         For SM-CHMM, VaR_t includes the AR(1) conditional mean μ_k + φ_k (y_{t-1} - μ_k).");
    println(io, "Target : α ∈ {0.01, 0.05}. LR_uc ~ χ²(1) (crit 3.84). LR_ind ~ χ²(1) (crit 3.84).");
    println(io, "Data   : SPY OoS window ($n_oos daily observations, 2024-01-04 to 2026-04-20).");
    println(io, "");
    println(io, rpad("Model",      12), " | ",
                rpad("medVaR01",   9), " | ", rpad("br%01",      6), " | ",
                rpad("LR_uc01",    7), " | ", rpad("p_uc01",     6), " | ",
                rpad("LR_ind01",   8), " | ", rpad("p_ind01",    7), " | ",
                rpad("medVaR05",   9), " | ", rpad("br%05",      6), " | ",
                rpad("LR_uc05",    7), " | ", rpad("p_uc05",     6), " | ",
                rpad("LR_ind05",   8), " | ", rpad("p_ind05",    7));
    println(io, "-"^150);
    for m in MODELS
        r = results[m];
        println(io, rpad(m, 12), " | ",
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
    println(io, "         Both should be below 3.84 for a passing conditional VaR.");
    println(io, "         Compare these LR_ind values against Track A A8 (unconditional VaR): LR_ind 6-20 there.");
end

# Summary digest
open(joinpath(TRACK_C3_DIR, "Track-C3a-summary.txt"), "w") do io
    println(io, "Track C3a (regime-conditional VaR) summary")
    println(io, "="^80);
    println(io, "");
    println(io, "Problem (from A8, unconditional VaR):");
    println(io, "   LR_ind was 6-20 across every model: consecutive breaches cluster in the 2024-2026");
    println(io, "   OoS window because an α-quantile of the pooled archive is fixed while observed vol");
    println(io, "   varies by regime.");
    println(io, "");
    println(io, "Fix: at each t, Viterbi-decode the current state k_t on R_oos and set VaR_t(α) to the");
    println(io, "     α-quantile of state k_t's emission. For SM-CHMM, VaR_t also adds the AR(1)");
    println(io, "     conditional-mean term μ_k + φ_k (y_{t-1} - μ_k).");
    println(io, "");
    println(io, "Results (1 % VaR):");
    for m in MODELS
        r = results[m];
        pass_uc = r.k01.LR < 3.84 ? "PASS" : "FAIL";
        pass_ind = r.c01.LR < 3.84 ? "PASS" : "FAIL";
        println(io, "  $(rpad(m, 12))  br $(round(100*r.k01.breach_rate, digits=2))%  LR_uc $(round(r.k01.LR, digits=2)) ($pass_uc)  LR_ind $(round(r.c01.LR, digits=2)) ($pass_ind)");
    end
    println(io, "");
    println(io, "Results (5 % VaR):");
    for m in MODELS
        r = results[m];
        pass_uc = r.k05.LR < 3.84 ? "PASS" : "FAIL";
        pass_ind = r.c05.LR < 3.84 ? "PASS" : "FAIL";
        println(io, "  $(rpad(m, 12))  br $(round(100*r.k05.breach_rate, digits=2))%  LR_uc $(round(r.k05.LR, digits=2)) ($pass_uc)  LR_ind $(round(r.c05.LR, digits=2)) ($pass_ind)");
    end
end

println("\n" * "="^72);
println("  Track C3a complete.");
println("  Results: $TRACK_C3_DIR");
println("="^72);

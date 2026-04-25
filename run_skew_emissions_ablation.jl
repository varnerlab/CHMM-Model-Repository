# ========================================================================================= #
# run_skew_emissions_ablation.jl
#
# Track M9 (revision response to referee comment M9): emission-family ablation for
# within-state skewness. SPY IS skewness is -0.75; the paper's three symmetric emission
# families (Gaussian, Student-t, Laplace) recover that skewness only through asymmetric
# state occupancy of symmetric emissions. This track reports two ablations:
#
#   (a) K = 1 skew-t vs symmetric t on the full SPY IS: a direct single-emission check
#       that confirms the Fernandez-Steel γ parameter picks up the observed skew.
#
#   (b) K = 18 plug-in skew-CHMM-t / skew-CHMM-L: use the symmetric CHMM-t / CHMM-L fit
#       of the main paper (fixed transitions, per-state μ_k, σ_k, ν_k or b_k) and then
#       fit per-state γ_k by weighted MLE over the EM posterior γ_t(k). Simulate the
#       resulting skew-CHMM-{t, L}, report simulated IS skewness, kurtosis, KS pass rate,
#       and compare against the symmetric variants.
#
# The K = 18 ablation is a "plug-in" rather than full joint EM; a principled full skew-t
# EM is a larger code project that would redefine the M-step for (μ_k, σ_k, ν_k, γ_k)
# simultaneously and is deferred to a follow-up session. The plug-in captures most of
# the question the referee is asking: "does adding within-state asymmetry materially
# improve the match to observed SPY skewness?"
#
# Outputs:
#   results/track_m9/Skew_Emissions.txt
#   ../CHMM-paper/results/robustness/skew_emissions_ablation.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
using Statistics
using HypothesisTests
const SEED = 20260422;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const N_PATHS   = 500;
const K_MAIN    = 18;
const MAX_ITER  = 60;

const TRACK_M9_DIR       = joinpath(_ROOT, "results", "track_m9");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(TRACK_M9_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  Track M9: skew-t / skew-Laplace emission ablation (referee M9)")
println("  Seed $SEED, K=$K_MAIN, N_paths=$N_PATHS")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS...");
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
spy_is = train_dataset["SPY"];
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
obs_skew = skewness(R_is);
obs_kurt = kurtosis(R_is);
println("  IS $n_is, observed skewness $(round(obs_skew, digits=3)), excess kurtosis $(round(obs_kurt, digits=3))");

# --------------------------------------------------------------------------------------- #
# (a) K = 1 symmetric t vs skew-t on full SPY IS
# --------------------------------------------------------------------------------------- #
println("\n" * "="^72);
println("  (a) K = 1: symmetric t vs skew-t (Fernandez-Steel)");
println("="^72);

# Symmetric Student-t MLE via simple grid + Nelder-Mead over (μ, σ, ν).
# Small wrapper around the existing GARCH-t fit idea but with α=β=0 (no AR/volatility term)
# so it reduces to an i.i.d. t MLE.
function _nll_sym_t(p::Vector{Float64}, obs::Vector{Float64})::Float64
    μ, σ, ν = p[1], p[2], p[3];
    if σ <= 0 || ν <= 2.1 || ν > 200; return 1e10; end
    acc = 0.0; N = length(obs);
    c = lgamma((ν + 1) / 2) - lgamma(ν / 2) - 0.5 * log(π * (ν - 2));
    for t in 1:N
        z = (obs[t] - μ) / σ;
        acc += c - log(σ) - 0.5 * (ν + 1) * log1p(z^2 / (ν - 2));
    end
    return -acc;
end

function _fit_sym_t(obs::Vector{Float64})
    μ_init = mean(obs); σ_init = std(obs); ν_init = 6.0;
    best, _ = _nelder_mead(p -> _nll_sym_t(p, obs), [μ_init, σ_init, ν_init]; max_iter=3000);
    return (μ=best[1], σ=best[2], ν=best[3]);
end

# Skew-t MLE via two-step: fit symmetric t to get (μ, σ, ν), then fit γ by 1D MLE.
# This matches the K = 18 plug-in approach below (single-state version).
Random.seed!(SEED + 1);
sym = _fit_sym_t(R_is);
# Fit γ with unit weights
w_unit = ones(n_is);
γ_hat = fit_gamma_mle_skewt(R_is, w_unit, sym.μ, sym.σ, sym.ν);
println("  Symmetric t MLE : μ=$(round(sym.μ, digits=4)), σ=$(round(sym.σ, digits=4)), ν=$(round(sym.ν, digits=3))");
println("  Skew-t γ MLE    : $(round(γ_hat, digits=4))   (γ < 1 = left-skew; SPY IS obs skew $(round(obs_skew, digits=3)))");

Random.seed!(SEED + 10);
sim_sym = Vector{Float64}(undef, n_is);
scale_to_unit_var = sqrt((sym.ν - 2) / sym.ν);
for i in 1:n_is
    sim_sym[i] = sym.μ + rand(TDist(sym.ν)) * sym.σ * scale_to_unit_var;
end

Random.seed!(SEED + 11);
sim_skew = [sample_skewt(sym.μ, sym.σ, sym.ν, γ_hat) for _ in 1:n_is];

sym_skew = skewness(sim_sym); sym_kurt = kurtosis(sim_sym);
skw_skew = skewness(sim_skew); skw_kurt = kurtosis(sim_skew);
println("  Simulated symmetric t : skewness $(round(sym_skew, digits=3)), excess kurtosis $(round(sym_kurt, digits=3))");
println("  Simulated skew-t      : skewness $(round(skw_skew, digits=3)), excess kurtosis $(round(skw_kurt, digits=3))");

k1_results = (
    sym_μ=sym.μ, sym_σ=sym.σ, sym_ν=sym.ν,
    skew_γ=γ_hat,
    sim_sym_skew=sym_skew, sim_sym_kurt=sym_kurt,
    sim_skew_skew=skw_skew, sim_skew_kurt=skw_kurt,
);

# --------------------------------------------------------------------------------------- #
# (b) K = 18 plug-in skew-CHMM: fit symmetric CHMM-t / CHMM-L, extract posteriors γ_t(k),
#     fit per-state γ_k by weighted MLE, simulate, score
# --------------------------------------------------------------------------------------- #
println("\n" * "="^72);
println("  (b) K = $K_MAIN plug-in: skew-CHMM-t and skew-CHMM-L");
println("="^72);

# Posterior extraction: run the standard log-space forward-backward given a fitted CHMM,
# returning the γ_t(k) matrix. Uses the same "uniform initial prior" convention as the
# Viterbi helper in src/Compute.jl.
function posterior_gamma(observations::Vector{Float64},
                         model::Union{MyContinuousHiddenMarkovModel,
                                      MyStudentTHiddenMarkovModel,
                                      MyLaplaceHiddenMarkovModel})::Matrix{Float64}
    N = length(observations);
    K = length(model.states);
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    log_T = log.(T_mat);
    log_B = zeros(N, K);
    for t in 1:N, k in 1:K
        log_B[t, k] = logpdf(model.emission[k], observations[t]);
    end
    # Forward
    log_α = zeros(N, K);
    log_α[1, :] .= log(1.0 / K) .+ log_B[1, :];
    for t in 2:N
        for k in 1:K
            log_α[t, k] = _logsumexp_vec(log_α[t-1, :] .+ log_T[:, k]) + log_B[t, k];
        end
    end
    # Backward
    log_β = zeros(N, K);
    for t in (N-1):-1:1
        for i in 1:K
            log_β[t, i] = _logsumexp_vec(log_T[i, :] .+ log_B[t+1, :] .+ log_β[t+1, :]);
        end
    end
    γ = zeros(N, K);
    for t in 1:N
        lse = _logsumexp_vec(log_α[t, :] .+ log_β[t, :]);
        γ[t, :] .= exp.(log_α[t, :] .+ log_β[t, :] .- lse);
    end
    return γ;
end

# Simulate a skew-CHMM by reusing the transition matrix from the symmetric model and
# drawing from per-state skew emissions. Initialisation: stationary distribution.
function simulate_skew_chmm_t(flat_t, μ::Vector{Float64}, σ::Vector{Float64},
                              ν::Vector{Float64}, γ::Vector{Float64}, n_steps::Int)
    K = length(flat_t.states);
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(flat_t.transition[i]); end
    # Stationary start
    π_stat = ones(K) ./ K;
    for _ in 1:50; π_stat = T_mat' * π_stat; π_stat ./= sum(π_stat); end
    s = rand(Categorical(π_stat));
    out = Vector{Float64}(undef, n_steps);
    for t in 1:n_steps
        out[t] = sample_skewt(μ[s], σ[s], ν[s], γ[s]);
        s = rand(Categorical(T_mat[s, :]));
    end
    return out;
end

function simulate_skew_chmm_l(flat_l, μ::Vector{Float64}, b::Vector{Float64},
                              γ::Vector{Float64}, n_steps::Int)
    K = length(flat_l.states);
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(flat_l.transition[i]); end
    π_stat = ones(K) ./ K;
    for _ in 1:50; π_stat = T_mat' * π_stat; π_stat ./= sum(π_stat); end
    s = rand(Categorical(π_stat));
    out = Vector{Float64}(undef, n_steps);
    for t in 1:n_steps
        out[t] = sample_skewl(μ[s], b[s], γ[s]);
        s = rand(Categorical(T_mat[s, :]));
    end
    return out;
end

function _simulate_paths(sim_fn, T::Int, n_paths::Int)::Matrix{Float64}
    paths = Matrix{Float64}(undef, T, n_paths);
    for p in 1:n_paths; paths[:, p] = sim_fn(); end
    return paths;
end

function ks_pass_rate(R_ref::Vector{Float64}, sim::Matrix{Float64}; α::Float64=0.05)
    n_sim = size(sim, 2); n_pass = 0;
    for p in 1:n_sim
        pv = pvalue(ApproximateTwoSampleKSTest(R_ref, sim[:, p]));
        if pv >= α; n_pass += 1; end
    end
    return n_pass / n_sim;
end

# ----- CHMM-t symmetric fit + plug-in skew ----- #
println("\n  [fit] symmetric CHMM-t at K=$K_MAIN...");
Random.seed!(SEED + 8);
flat_t = build(MyStudentTHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
γ_post_t = posterior_gamma(R_is, flat_t);
μ_t = [flat_t.emission[k].μ for k in 1:K_MAIN];
σ_t = [flat_t.emission[k].σ for k in 1:K_MAIN];
ν_t = [params(flat_t.emission[k].ρ)[1] for k in 1:K_MAIN];
γ_k_t = Vector{Float64}(undef, K_MAIN);
for k in 1:K_MAIN
    γ_k_t[k] = fit_gamma_mle_skewt(R_is, γ_post_t[:, k], μ_t[k], σ_t[k], ν_t[k]);
end
println("  plug-in skew-t γ_k  : min $(round(minimum(γ_k_t), digits=3))  median $(round(median(γ_k_t), digits=3))  max $(round(maximum(γ_k_t), digits=3))");
n_left_skew = count(γ_k_t .< 0.95);
n_right_skew = count(γ_k_t .> 1.05);
println("  states with γ_k < 0.95 (left-skew): $n_left_skew / $K_MAIN;  γ_k > 1.05 (right-skew): $n_right_skew / $K_MAIN");

println("\n  [fit] symmetric CHMM-L at K=$K_MAIN...");
Random.seed!(SEED + 9);
flat_l = build(MyLaplaceHiddenMarkovModel,
    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
γ_post_l = posterior_gamma(R_is, flat_l);
μ_l = [flat_l.emission[k].μ for k in 1:K_MAIN];
b_l = [flat_l.emission[k].θ for k in 1:K_MAIN];
γ_k_l = Vector{Float64}(undef, K_MAIN);
for k in 1:K_MAIN
    γ_k_l[k] = fit_gamma_mle_skewl(R_is, γ_post_l[:, k], μ_l[k], b_l[k]);
end
println("  plug-in skew-L γ_k  : min $(round(minimum(γ_k_l), digits=3))  median $(round(median(γ_k_l), digits=3))  max $(round(maximum(γ_k_l), digits=3))");
n_left_skew_l = count(γ_k_l .< 0.95);
n_right_skew_l = count(γ_k_l .> 1.05);
println("  states with γ_k < 0.95 (left-skew): $n_left_skew_l / $K_MAIN;  γ_k > 1.05 (right-skew): $n_right_skew_l / $K_MAIN");

# ----- Simulate and score ----- #
println("\n  [sim+score] K=$K_MAIN simulations...");

# Symmetric CHMM-t reference
Random.seed!(SEED + 100);
sim_sym_t = _simulate_paths(() -> simulate_returns(flat_t, n_is), n_is, N_PATHS);
sym_t_skew = mean([skewness(sim_sym_t[:, p]) for p in 1:N_PATHS]);
sym_t_kurt = mean([kurtosis(sim_sym_t[:, p]) for p in 1:N_PATHS]);
sym_t_ks = ks_pass_rate(R_is, sim_sym_t);

Random.seed!(SEED + 101);
sim_skew_t = _simulate_paths(() -> simulate_skew_chmm_t(flat_t, μ_t, σ_t, ν_t, γ_k_t, n_is), n_is, N_PATHS);
skw_t_skew = mean([skewness(sim_skew_t[:, p]) for p in 1:N_PATHS]);
skw_t_kurt = mean([kurtosis(sim_skew_t[:, p]) for p in 1:N_PATHS]);
skw_t_ks = ks_pass_rate(R_is, sim_skew_t);

Random.seed!(SEED + 102);
sim_sym_l = _simulate_paths(() -> simulate_returns(flat_l, n_is), n_is, N_PATHS);
sym_l_skew = mean([skewness(sim_sym_l[:, p]) for p in 1:N_PATHS]);
sym_l_kurt = mean([kurtosis(sim_sym_l[:, p]) for p in 1:N_PATHS]);
sym_l_ks = ks_pass_rate(R_is, sim_sym_l);

Random.seed!(SEED + 103);
sim_skew_l = _simulate_paths(() -> simulate_skew_chmm_l(flat_l, μ_l, b_l, γ_k_l, n_is), n_is, N_PATHS);
skw_l_skew = mean([skewness(sim_skew_l[:, p]) for p in 1:N_PATHS]);
skw_l_kurt = mean([kurtosis(sim_skew_l[:, p]) for p in 1:N_PATHS]);
skw_l_ks = ks_pass_rate(R_is, sim_skew_l);

panel = [
    ("CHMM-t (sym, K=18)",    sym_t_skew, sym_t_kurt, sym_t_ks),
    ("CHMM-t (skew, K=18)",   skw_t_skew, skw_t_kurt, skw_t_ks),
    ("CHMM-L (sym, K=18)",    sym_l_skew, sym_l_kurt, sym_l_ks),
    ("CHMM-L (skew, K=18)",   skw_l_skew, skw_l_kurt, skw_l_ks),
];

println("\n  Results (observed IS skewness $(round(obs_skew, digits=3)), excess kurtosis $(round(obs_kurt, digits=3))):");
for (name, sk, kt, ks) in panel
    println("    $(rpad(name, 24))  sim_skew $(round(sk, digits=3))  sim_kurt $(round(kt, digits=3))  IS_KS $(round(100*ks, digits=1))%");
end

# --------------------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_M9_DIR, "Skew_Emissions.txt"), "w") do io
    println(io, "="^140);
    println(io, "Track M9. Skew-t / skew-Laplace emission ablation (referee M9 response).");
    println(io, "="^140);
    println(io, "");
    println(io, "Fernandez-Steel skewing: f(x; μ, σ, γ) = (2/(σ(γ+1/γ))) * f_0((x-μ)/σ * γ^{-sign(x-μ)}),");
    println(io, "  γ < 1 = left-skew, γ > 1 = right-skew, γ = 1 recovers symmetric.");
    println(io, "Observed SPY IS: skewness $(round(obs_skew, digits=3))  excess kurtosis $(round(obs_kurt, digits=3))  ($n_is obs).");
    println(io, "");
    println(io, "(a) K = 1 symmetric t vs skew-t:");
    println(io, "  Symmetric t MLE : μ = $(round(k1_results.sym_μ, digits=4))   σ = $(round(k1_results.sym_σ, digits=4))   ν = $(round(k1_results.sym_ν, digits=3))");
    println(io, "  Skew-t γ MLE    : γ = $(round(k1_results.skew_γ, digits=4))  (γ < 1 indicates left-skew, consistent with observed SPY skewness $(round(obs_skew, digits=3)))");
    println(io, "  Simulated symmetric t  : skewness = $(round(k1_results.sim_sym_skew, digits=3))   kurtosis = $(round(k1_results.sim_sym_kurt, digits=3))");
    println(io, "  Simulated skew-t       : skewness = $(round(k1_results.sim_skew_skew, digits=3))   kurtosis = $(round(k1_results.sim_skew_kurt, digits=3))");
    println(io, "");
    println(io, "(b) K = $K_MAIN plug-in skew-CHMM-{t, L}:");
    println(io, "  Skew-CHMM-t per-state γ_k: min $(round(minimum(γ_k_t), digits=3)), median $(round(median(γ_k_t), digits=3)), max $(round(maximum(γ_k_t), digits=3)); left-skew states $n_left_skew / $K_MAIN, right-skew states $n_right_skew / $K_MAIN");
    println(io, "  Skew-CHMM-L per-state γ_k: min $(round(minimum(γ_k_l), digits=3)), median $(round(median(γ_k_l), digits=3)), max $(round(maximum(γ_k_l), digits=3)); left-skew states $n_left_skew_l / $K_MAIN, right-skew states $n_right_skew_l / $K_MAIN");
    println(io, "");
    println(io, rpad("Variant", 22), " | ", rpad("sim_skew", 10), " | ", rpad("sim_kurt", 10), " | ", rpad("IS_KS%", 8));
    println(io, "-"^70);
    for (name, sk, kt, ks) in panel
        println(io, rpad(name, 22), " | ",
                    rpad(round(sk, digits=3), 10), " | ",
                    rpad(round(kt, digits=3), 10), " | ",
                    rpad(round(100*ks, digits=1), 8));
    end
    println(io, "="^140);
    println(io, "");
    println(io, "Reading: the plug-in approach uses the symmetric CHMM fit (transitions + per-state (μ, σ, ν) or (μ, b))");
    println(io, "and only adds a per-state γ_k by weighted MLE on the EM posterior; it is a one-shot skew correction,");
    println(io, "not full joint EM, but gives the referee the within-state-asymmetry ablation they asked about.");
end

open(joinpath(PAPER_ROBUSTNESS_DIR, "skew_emissions_ablation.csv"), "w") do io
    println(io, "variant,sim_skew,sim_kurt,IS_KS_pct,obs_skew,obs_kurt");
    # K = 1 rows
    println(io, "K=1 sym-t,$(round(k1_results.sim_sym_skew, digits=4)),$(round(k1_results.sim_sym_kurt, digits=4)),NA,$(round(obs_skew, digits=4)),$(round(obs_kurt, digits=4))");
    println(io, "K=1 skew-t (γ=$(round(k1_results.skew_γ, digits=3))),$(round(k1_results.sim_skew_skew, digits=4)),$(round(k1_results.sim_skew_kurt, digits=4)),NA,$(round(obs_skew, digits=4)),$(round(obs_kurt, digits=4))");
    # K = 18 panel
    for (name, sk, kt, ks) in panel
        println(io, "$name,$(round(sk, digits=4)),$(round(kt, digits=4)),$(round(100*ks, digits=2)),$(round(obs_skew, digits=4)),$(round(obs_kurt, digits=4))");
    end
end

println("\n" * "="^72);
println("  Track M9 complete.");
println("  Text report : $(joinpath(TRACK_M9_DIR, "Skew_Emissions.txt"))");
println("  Paper CSV   : $(joinpath(PAPER_ROBUSTNESS_DIR, "skew_emissions_ablation.csv"))");
println("="^72);

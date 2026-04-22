# ========================================================================================= #
# run_vix_generalization.jl
#
# Block D: Volatility-Index Generalization.
# Fits the same CHMM scaffold used for equity returns to the CBOE VIX index,
# using log(VIX) levels y_t = log(VIX_t) as the univariate observable. This is
# the transform used by the log-mean-reverting VIX literature (Papanicolaou &
# Sircar 2014, Kaeck & Alexander 2013, log-Heston / log-OU family) and aligns
# with the stationarity and mean-reversion of the observable (ADF rejects unit
# root on the IS window; AR(1) half-life = 24.7 trading days).
#
# Three emission families (Gaussian, Student-t, Laplace) are trained at K = 18
# and compared against two trivial baselines:
#   - AR(1) on log(VIX) levels (log-OU style, captures half-life)
#   - Gaussian i.i.d. on log(VIX) levels (captures marginal only; no dynamics)
#
# Inputs:
#   data/VIX-Train-10yr.jld2       (from build_vix_train_oos.jl)
#   data/VIX-OoS-Remainder.jld2    (from build_vix_train_oos.jl)
#
# Outputs:
#   results/VIX/K18/N/*.jld2       (CHMM-N model + simulations)
#   results/VIX/K18/t/*.jld2       (CHMM-t model + simulations)
#   results/VIX/K18/L/*.jld2       (CHMM-L model + simulations)
#   results/VIX/baselines/*.jld2   (AR(1), Gaussian i.i.d. simulations)
#
# Usage: julia --project=. run_vix_generalization.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random;
Random.seed!(20260421);

const K             = 18;
const MAX_ITER      = 60;
const N_PATHS       = 500;
const RESULTS_DIR   = joinpath(_ROOT, "results", "VIX");
const TRAIN_IN      = joinpath(_PATH_TO_DATA, "VIX-Train-10yr.jld2");
const OOS_IN        = joinpath(_PATH_TO_DATA, "VIX-OoS-Remainder.jld2");

println("="^72);
println("  Block D: Volatility-Index CHMM Generalization");
println("  Observable: log(VIX) levels   States: K = $K   Paths: $N_PATHS");
println("="^72);

# --- Load VIX train / OoS ---
train_vix = load(TRAIN_IN)["dataset"]["VIX"];
oos_vix   = load(OOS_IN)["dataset"]["VIX"];
log_vix_train = log.(train_vix.close);
log_vix_oos   = log.(oos_vix.close);
y_train = log_vix_train;                # log(VIX) levels, in-sample
y_oos   = log_vix_oos;                  # log(VIX) levels, out-of-sample
dlog_train = diff(log_vix_train);       # retained for diagnostics comparison
n_train = length(y_train);
n_oos   = length(y_oos);
println("[load] log(VIX) train obs: $n_train   OoS obs: $n_oos");
println("[load] levels: μ=$(round(mean(y_train), digits=4))  σ=$(round(std(y_train), digits=4))  " *
        "min=$(round(minimum(y_train), digits=3))  max=$(round(maximum(y_train), digits=3))");
println("[load] increments Δlog(VIX):  skew=$(round(mean(((dlog_train .- mean(dlog_train)) ./ std(dlog_train)).^3), digits=3))  " *
        "excess-kurt=$(round(mean(((dlog_train .- mean(dlog_train)) ./ std(dlog_train)).^4) - 3, digits=3))");

# --- Emission-family dispatch ---
const FAMILY_TAG   = Dict(:gaussian => "N", :student_t => "t", :laplace => "L");
const FAMILY_LABEL = Dict(:gaussian => "CHMM-N (Gaussian)",
                          :student_t => "CHMM-t (Student-t)",
                          :laplace => "CHMM-L (Laplace)");

function _train_family(family::Symbol, obs::Vector{Float64}, K::Int, max_iter::Int)
    if family == :gaussian
        return build(MyContinuousHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :student_t
        return build(MyStudentTHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    elseif family == :laplace
        return build(MyLaplaceHiddenMarkovModel,
            (observations=obs, number_of_states=K, max_iter=max_iter));
    else
        error("Unknown emission family: $family")
    end
end

function _stationary(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return T_mat, Categorical(π_stat);
end

function _simulate_paths(model, start_dist, n_is::Int, n_oos_::Int, n_paths::Int)
    sim_is = Array{Float64,2}(undef, n_is, n_paths);
    sim_oos_ = Array{Float64,2}(undef, n_oos_, n_paths);
    for i in 1:n_paths
        s0 = rand(start_dist);
        st = model(s0, n_is);
        for j in 1:n_is;  sim_is[j, i]  = rand(model.emission[st[j]]); end
        s0 = rand(start_dist);
        st = model(s0, n_oos_);
        for j in 1:n_oos_; sim_oos_[j, i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos_;
end

# --- Train and simulate each emission family ---
println("\n[train] Fitting CHMM-N, CHMM-t, CHMM-L at K = $K...");
mkpath(RESULTS_DIR);

for family in (:gaussian, :student_t, :laplace)
    tag = FAMILY_TAG[family];
    label = FAMILY_LABEL[family];
    out_dir = joinpath(RESULTS_DIR, "K$K", tag);
    mkpath(out_dir);

    println("\n  >>> $label");
    t0 = time();
    model = _train_family(family, y_train, K, MAX_ITER);
    fit_sec = time() - t0;
    println("      EM iterations: $(length(model.log_likelihood_history))  " *
            "final LL: $(round(last(model.log_likelihood_history), digits=2))  " *
            "wall-clock: $(round(fit_sec, digits=1)) s");

    T_mat, start_dist = _stationary(model, K);

    sim_is, sim_oos_ = _simulate_paths(model, start_dist, n_train, n_oos, N_PATHS);

    # Save the fitted model and simulations
    jldsave(joinpath(out_dir, "model.jld2"); model=model,
            T_mat=T_mat, fit_sec=fit_sec);
    jldsave(joinpath(out_dir, "simulations.jld2");
            sim_is=sim_is, sim_oos=sim_oos_);

    println("      saved -> $(out_dir)");
end

# --- Baselines: AR(1) on log(VIX) levels; Gaussian i.i.d. on Δlog(VIX) ---
println("\n[baseline] AR(1) on log(VIX) levels and Gaussian i.i.d. on Δlog(VIX)...");
base_dir = joinpath(RESULTS_DIR, "baselines");
mkpath(base_dir);

# AR(1): log(VIX)_t = c + φ * log(VIX)_{t-1} + ε_t
yL = log_vix_train;
X  = hcat(ones(length(yL)-1), yL[1:end-1]);
βhat = X \ yL[2:end];
c_hat, φ_hat = βhat[1], βhat[2];
σ_hat = std(yL[2:end] - X * βhat);
println("  AR(1) on log(VIX):  φ = $(round(φ_hat, digits=4))   " *
        "c = $(round(c_hat, digits=4))   σ_ε = $(round(σ_hat, digits=4))   " *
        "half-life = $(round(log(2) / (-log(abs(φ_hat))), digits=1)) days");

function _simulate_ar1(c::Float64, φ::Float64, σ::Float64,
        y0::Float64, n::Int, n_paths::Int)
    sims = Array{Float64,2}(undef, n, n_paths);
    for p in 1:n_paths
        yt = y0;
        for t in 1:n
            yt = c + φ * yt + σ * randn();
            sims[t, p] = yt;
        end
    end
    return sims;
end

ar1_is  = _simulate_ar1(c_hat, φ_hat, σ_hat, yL[1], n_train, N_PATHS);
ar1_oos = _simulate_ar1(c_hat, φ_hat, σ_hat, log_vix_oos[1], n_oos, N_PATHS);

# Gaussian i.i.d. on log(VIX) levels (destroys temporal structure)
μG = mean(y_train); σG = std(y_train);
iid_is  = μG .+ σG .* randn(n_train, N_PATHS);
iid_oos = μG .+ σG .* randn(n_oos,   N_PATHS);

jldsave(joinpath(base_dir, "ar1.jld2");
        c=c_hat, phi=φ_hat, sigma=σ_hat,
        sim_is=ar1_is, sim_oos=ar1_oos);
jldsave(joinpath(base_dir, "gaussian_iid.jld2");
        mu=μG, sigma=σG, sim_is=iid_is, sim_oos=iid_oos);

# --- Also save the observed series for downstream diagnostics ---
jldsave(joinpath(RESULTS_DIR, "observed.jld2");
        log_vix_train=log_vix_train, log_vix_oos=log_vix_oos,
        y_train=y_train, y_oos=y_oos,
        dlog_train=dlog_train, dlog_oos=diff(log_vix_oos),
        dates_train=train_vix.date, dates_oos=oos_vix.date);

println();
println("="^72);
println("  DONE  (results -> $(RESULTS_DIR))");
println("="^72);

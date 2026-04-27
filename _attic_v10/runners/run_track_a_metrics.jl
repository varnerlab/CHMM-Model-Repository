# ========================================================================================= #
# run_track_a_metrics.jl
#
# Track A metrics runner (DECISION-MEMO items A1, A2, A3, A6, A7, A9, A10):
#   A1  MMD (RBF, median heuristic) on 20-day windowed paths
#   A2  Signature MMD at depth 3 on time-augmented 2D paths
#   A3  Discriminator AUC (logistic regression on 10 hand-crafted window features, 5-fold CV)
#   A6  Leverage-effect profile corr(r_t^2, r_{t-k}) and down/up asymmetry
#   A7  Aggregational kurtosis at horizons {1, 5, 10, 21}
#   A9  Simulation-based two-sided p-values on stylized-fact statistics
#   A10 Unify simulation counts at N_PATHS = 1000
#
# Models evaluated (matches Table 2): Bootstrap, Gaussian, Laplace, Discrete NJ, Discrete WJ,
#                                     GARCH, CHMM-N, CHMM-t, CHMM-L (all at K=18 for CHMMs).
#
# Outputs:
#   results/track_a/Table-4-Extended-Metrics.txt           extended panel (IS + OoS)
#   results/track_a/leverage_effect.txt                    A6 profile and asymmetry
#   results/track_a/aggregational_kurtosis.txt             A7 per-model per-horizon kurtosis
#   results/track_a/sim_pvalues.txt                        A9 joint and per-stat p-values
#   results/track_a/Track-A-summary.txt                    one-page digest
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER       = "SPY";
const RISK_FREE    = 0.0;
const DT           = 1/252;
const N_PATHS      = 1000;          # A10: unified path count
const L_LAGS       = 252;
const K_MAIN       = 18;
const MAX_ITER     = 60;
# Discrete baseline hyper-parameters (from prior paper arXiv:2603.10202,
# Notebook 3 `3-HMM-WithJumps-Simulation-Notebook.ipynb`, final SPY run)
const DISCRETE_K   = 90;         # number_of_states from prior paper
const EPSILON_JUMP = 0.00005;    # fraction of steps with a jump event
const LAMBDA_JUMP  = 67.0;       # mean number of jump events (Poisson)

const WINDOW_LEN   = 20;            # window length for MMD, sig-MMD, disc AUC
const N_WINDOWS    = 500;
const SIG_DEPTH    = 3;
const MAX_LAG_LEV  = 20;
const HORIZONS_AG  = [1, 5, 10, 21];

const TRACK_A_DIR = joinpath(_ROOT, "results", "track_a");
mkpath(TRACK_A_DIR);

println("="^72)
println("  Track A metrics runner")
println("  Seed:      $SEED")
println("  N paths:   $N_PATHS")
println("  K (CHMMs): $K_MAIN")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data (matches run_baselines_and_cross_asset.jl)
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS + OoS...")

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
# Helpers (mirror run_baselines_and_cross_asset.jl)
# --------------------------------------------------------------------------------------- #
function _stationary_start_dist(model, K::Int)
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(model.transition[i]); end
    π_stat = (T_mat^1000)[1, :];
    return Categorical(π_stat);
end

function _simulate_chmm_paths(model, start_dist, nis::Int, noos::Int, np::Int)
    sim_is = Array{Float64,2}(undef, nis, np);
    sim_oos = Array{Float64,2}(undef, noos, np);
    for i in 1:np
        s0 = rand(start_dist); st = model(s0, nis);
        for j in 1:nis; sim_is[j, i] = rand(model.emission[st[j]]); end
        s0 = rand(start_dist); st = model(s0, noos);
        for j in 1:noos; sim_oos[j, i] = rand(model.emission[st[j]]); end
    end
    return sim_is, sim_oos;
end

function _discrete_archive(T_disc, bin_centers, sd_disc, model, nis::Int, noos::Int, np::Int)
    arch_is = Array{Float64,2}(undef, nis, np);
    arch_oos = Array{Float64,2}(undef, noos, np);
    DK = length(bin_centers);
    for i in 1:np
        s0 = rand(sd_disc); ch = model(s0, nis);
        for t in 1:nis
            eb = rand(model.emission[ch[t]]);
            arch_is[t, i] = bin_centers[eb];
        end
        s0 = rand(sd_disc); ch = model(s0, noos);
        for t in 1:noos
            eb = rand(model.emission[ch[t]]);
            arch_oos[t, i] = bin_centers[eb];
        end
    end
    return arch_is, arch_oos;
end

# --------------------------------------------------------------------------------------- #
# Generate or cache simulation archives
# --------------------------------------------------------------------------------------- #
cache_path = joinpath(TRACK_A_DIR, "sim_archive_cache.jld2");

if isfile(cache_path)
    println("\n[cache] Loading simulation archives from $(basename(cache_path))...");
    archive = load(cache_path)["archive"];
else
    println("\n[sim] Simulating all model archives ($N_PATHS paths each)...");
    archive = Dict{String,NamedTuple}();

    # Observed
    archive["Observed"] = (is=reshape(R_is, :, 1), oos=reshape(R_oos, :, 1));

    # Bootstrap
    println("  Bootstrap...");
    Random.seed!(SEED + 1);
    boot_is = Array{Float64,2}(undef, n_is, N_PATHS);
    boot_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
    for i in 1:N_PATHS
        boot_is[:, i]  = R_is[rand(1:n_is, n_is)];
        boot_oos[:, i] = R_is[rand(1:n_is, n_oos)];
    end
    archive["Bootstrap"] = (is=boot_is, oos=boot_oos);

    # Gaussian
    println("  Gaussian i.i.d....");
    μ_obs = mean(R_is); σ_obs = std(R_is);
    d_gauss = Normal(μ_obs, σ_obs);
    Random.seed!(SEED + 2);
    g_is = Array{Float64,2}(undef, n_is, N_PATHS);
    g_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
    for i in 1:N_PATHS
        g_is[:, i]  = rand(d_gauss, n_is);
        g_oos[:, i] = rand(d_gauss, n_oos);
    end
    archive["Gaussian"] = (is=g_is, oos=g_oos);

    # Laplace
    println("  Laplace i.i.d....");
    μ_lap = median(R_is); b_lap = mean(abs.(R_is .- μ_lap));
    d_lap = Laplace(μ_lap, b_lap);
    Random.seed!(SEED + 3);
    l_is = Array{Float64,2}(undef, n_is, N_PATHS);
    l_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
    for i in 1:N_PATHS
        l_is[:, i]  = rand(d_lap, n_is);
        l_oos[:, i] = rand(d_lap, n_oos);
    end
    archive["Laplace"] = (is=l_is, oos=l_oos);

    # Discrete HMM: no jumps (NJ) and with Poisson jumps (WJ)
    println("  Discrete HMM (NJ + WJ)...");
    qprobs_disc = range(0.0, 1.0, length=DISCRETE_K+1) |> collect;
    bin_edges = quantile(R_is, qprobs_disc);
    function _assign_bin(x, edges)
        Kb = length(edges) - 1;
        for k in 1:Kb; if k == Kb || x < edges[k+1]; return k; end; end
        return Kb;
    end;
    bin_idx_vec = [_assign_bin(r, bin_edges) for r in R_is];
    bin_centers = zeros(DISCRETE_K);
    for k in 1:DISCRETE_K
        mk = findall(==(k), bin_idx_vec);
        bin_centers[k] = isempty(mk) ? 0.0 : mean(R_is[mk]);
    end
    T_counts = zeros(DISCRETE_K, DISCRETE_K);
    for t in 2:length(bin_idx_vec); T_counts[bin_idx_vec[t-1], bin_idx_vec[t]] += 1.0; end
    T_disc = copy(T_counts);
    for i in 1:DISCRETE_K
        rs = sum(T_disc[i, :]);
        T_disc[i, :] = rs > 0 ? T_disc[i, :] ./ rs : fill(1.0/DISCRETE_K, DISCRETE_K);
    end
    E_disc = zeros(DISCRETE_K, DISCRETE_K);
    floor_ = 0.001;
    for i in 1:DISCRETE_K
        for j in 1:DISCRETE_K; E_disc[i, j] = (i == j) ? 1.0 : floor_; end
        E_disc[i, :] ./= sum(E_disc[i, :]);
    end
    π_disc = (T_disc^1000)[1, :];
    sd_disc = Categorical(π_disc);

    Random.seed!(SEED + 4);
    model_nj = build(MyHiddenMarkovModel, (states=collect(1:DISCRETE_K), T=T_disc, E=E_disc));
    nj_is, nj_oos = _discrete_archive(T_disc, bin_centers, sd_disc, model_nj, n_is, n_oos, N_PATHS);
    archive["DiscreteNJ"] = (is=nj_is, oos=nj_oos);

    Random.seed!(SEED + 5);
    model_wj = build(MyHiddenMarkovModelWithJumps, (
        states=collect(1:DISCRETE_K), T=T_disc, E=E_disc,
        ϵ=EPSILON_JUMP, λ=LAMBDA_JUMP));
    wj_is, wj_oos = _discrete_archive(T_disc, bin_centers, sd_disc, model_wj, n_is, n_oos, N_PATHS);
    archive["DiscreteWJ"] = (is=wj_is, oos=wj_oos);

    # GARCH(1,1)
    println("  GARCH(1,1)...");
    Random.seed!(SEED + 6);
    garch_model = build(MyGARCHModel, (observations=R_is,));
    g2_is = Array{Float64,2}(undef, n_is, N_PATHS);
    g2_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
    for i in 1:N_PATHS
        g2_is[:, i]  = simulate_garch(garch_model, n_is);
        g2_oos[:, i] = simulate_garch(garch_model, n_oos);
    end
    archive["GARCH"] = (is=g2_is, oos=g2_oos);

    # CHMM-N
    println("  CHMM-N (K=$K_MAIN)...");
    Random.seed!(SEED + 7);
    chmm_n = build(MyContinuousHiddenMarkovModel, (
        observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
    sd_n = _stationary_start_dist(chmm_n, K_MAIN);
    n_is_sim, n_oos_sim = _simulate_chmm_paths(chmm_n, sd_n, n_is, n_oos, N_PATHS);
    archive["CHMM-N"] = (is=n_is_sim, oos=n_oos_sim);

    # CHMM-t
    println("  CHMM-t (K=$K_MAIN)...");
    Random.seed!(SEED + 8);
    chmm_t = build(MyStudentTHiddenMarkovModel, (
        observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
    sd_t = _stationary_start_dist(chmm_t, K_MAIN);
    t_is_sim, t_oos_sim = _simulate_chmm_paths(chmm_t, sd_t, n_is, n_oos, N_PATHS);
    archive["CHMM-t"] = (is=t_is_sim, oos=t_oos_sim);

    # CHMM-L
    println("  CHMM-L (K=$K_MAIN)...");
    Random.seed!(SEED + 9);
    chmm_l = build(MyLaplaceHiddenMarkovModel, (
        observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
    sd_l = _stationary_start_dist(chmm_l, K_MAIN);
    l_is_sim, l_oos_sim = _simulate_chmm_paths(chmm_l, sd_l, n_is, n_oos, N_PATHS);
    archive["CHMM-L"] = (is=l_is_sim, oos=l_oos_sim);

    save(cache_path, "archive", archive);
    println("[cache] Saved archives to $(basename(cache_path))");
end

MODEL_ORDER = [
    "Bootstrap", "Gaussian", "Laplace",
    "DiscreteNJ", "DiscreteWJ", "GARCH",
    "CHMM-N", "CHMM-t", "CHMM-L",
];

# --------------------------------------------------------------------------------------- #
# A1, A2, A3: MMD, signature-MMD, discriminator AUC on 20-day windows
# --------------------------------------------------------------------------------------- #
println("\n[A1,A2,A3] Computing MMD / sig-MMD / discriminator AUC (W=$WINDOW_LEN, n_windows=$N_WINDOWS, seed=$SEED)");

# Observed windows (fixed across all models)
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
println("  observed windows: IS $n_obs_win_is, OoS $n_obs_win_oos");

mmd_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER
    print("  $m...");
    rng = MersenneTwister(SEED + hash(m) % 10^6);
    s_is, s_oos = archive[m].is, archive[m].oos;

    # Sample windows
    syn_windows_is = sample_windows_from_archive(s_is, WINDOW_LEN, n_obs_win_is; rng=rng);
    syn_windows_oos = sample_windows_from_archive(s_oos, WINDOW_LEN, n_obs_win_oos; rng=rng);

    # A1: MMD on raw windows (with shared bandwidth from pooled sample)
    mmd_is = mmd2_rbf(obs_windows_is, syn_windows_is; rng=rng);
    mmd_oos = mmd2_rbf(obs_windows_oos, syn_windows_oos; rng=rng);

    # A2: signature-MMD at depth 3
    sig_is = sig_mmd2(obs_windows_is, syn_windows_is; depth=SIG_DEPTH, rng=rng);
    sig_oos = sig_mmd2(obs_windows_oos, syn_windows_oos; depth=SIG_DEPTH, rng=rng);

    # A3: discriminator AUC against full IS/OoS series (uses windowed features internally)
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
# A6: leverage effect
# --------------------------------------------------------------------------------------- #
println("\n[A6] Leverage effect (max_lag=$MAX_LAG_LEV)");

lev_obs_is = leverage_effect(R_is; max_lag=MAX_LAG_LEV);
lev_obs_oos = leverage_effect(R_oos; max_lag=MAX_LAG_LEV);
println("  Observed IS  avg_neg=$(round(lev_obs_is.avg_neg, digits=4))  asymmetry=$(round(lev_obs_is.asymmetry, digits=5))");
println("  Observed OoS avg_neg=$(round(lev_obs_oos.avg_neg, digits=4))  asymmetry=$(round(lev_obs_oos.asymmetry, digits=5))");

lev_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER
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
        med_is=median(avg_is),  med_asym_is=median(asym_is),
        med_oos=median(avg_oos), med_asym_oos=median(asym_oos),
        p_avg_is=sim_pvalue(lev_obs_is.avg_neg, avg_is),
        p_avg_oos=sim_pvalue(lev_obs_oos.avg_neg, avg_oos),
    );
    println("  $m  med avg_neg IS=$(round(lev_results[m].med_is, digits=4)) OoS=$(round(lev_results[m].med_oos, digits=4)) pv_is=$(round(lev_results[m].p_avg_is, digits=2))");
end

# --------------------------------------------------------------------------------------- #
# A7: aggregational kurtosis
# --------------------------------------------------------------------------------------- #
println("\n[A7] Aggregational kurtosis (horizons $HORIZONS_AG)");

agg_obs_is = aggregational_kurtosis(R_is; horizons=HORIZONS_AG);
agg_obs_oos = aggregational_kurtosis(R_oos; horizons=HORIZONS_AG);
println("  Observed IS:  ", join(["h=$(h) k=$(round(agg_obs_is[h], digits=2))" for h in HORIZONS_AG], "  "));
println("  Observed OoS: ", join(["h=$(h) k=$(round(agg_obs_oos[h], digits=2))" for h in HORIZONS_AG], "  "));

agg_results = Dict{String,Dict}();
for m in MODEL_ORDER
    s_is, s_oos = archive[m].is, archive[m].oos;
    d = Dict{Symbol,Any}();
    for (tag, arch) in (:is => s_is, :oos => s_oos)
        per_h = Dict{Int,Vector{Float64}}();
        for h in HORIZONS_AG; per_h[h] = Float64[]; end
        for i in 1:size(arch, 2)
            ak = aggregational_kurtosis(arch[:, i]; horizons=HORIZONS_AG);
            for h in HORIZONS_AG; push!(per_h[h], ak[h]); end
        end
        d[tag] = per_h;
    end
    agg_results[m] = d;
end

# --------------------------------------------------------------------------------------- #
# A9: simulation-based p-values on stylized-fact statistics
# --------------------------------------------------------------------------------------- #
println("\n[A9] Simulation-based p-values on {kurt, ACF-MAE |r|, avg_neg lev, agg kurt h=5, agg kurt h=21}");

function _kurt(x)
    μ = mean(x); σ = std(x);
    return σ > 0 ? sum(((x .- μ) ./ σ) .^ 4) / length(x) - 3.0 : 0.0;
end

function _acf_mae_abs(obs, sim; L=L_LAGS)
    a_o = autocor(abs.(obs), 1:min(L, length(obs)-1));
    a_s = autocor(abs.(sim), 1:min(L, length(sim)-1));
    return mean(abs.(a_o .- a_s));
end

# Observed stats
obs_kurt_is = _kurt(R_is);
obs_kurt_oos = _kurt(R_oos);
obs_ak_is_5  = agg_obs_is[5];
obs_ak_is_21 = agg_obs_is[21];
obs_ak_oos_5  = agg_obs_oos[5];
obs_ak_oos_21 = agg_obs_oos[21];

pv_results = Dict{String,NamedTuple}();
for m in MODEL_ORDER
    s_is, s_oos = archive[m].is, archive[m].oos;
    kurts_is = [_kurt(s_is[:, i]) for i in 1:size(s_is, 2)];
    kurts_oos = [_kurt(s_oos[:, i]) for i in 1:size(s_oos, 2)];
    # ACF-MAE not a per-path stat vs observed ACF: use per-path ACF-MAE distribution
    acf_mae_is = [_acf_mae_abs(R_is, s_is[:, i]) for i in 1:size(s_is, 2)];
    acf_mae_oos = [_acf_mae_abs(R_oos, s_oos[:, i]) for i in 1:size(s_oos, 2)];
    # Leverage
    lev_avg_is = lev_results[m];
    # Aggregational kurtosis per path
    ak5_is = agg_results[m][:is][5];
    ak21_is = agg_results[m][:is][21];
    ak5_oos = agg_results[m][:oos][5];
    ak21_oos = agg_results[m][:oos][21];
    pv = (
        p_kurt_is  = sim_pvalue(obs_kurt_is, kurts_is),
        p_kurt_oos = sim_pvalue(obs_kurt_oos, kurts_oos),
        p_acf_is   = sim_pvalue(0.0, acf_mae_is),
        p_acf_oos  = sim_pvalue(0.0, acf_mae_oos),
        p_lev_is   = lev_avg_is.p_avg_is,
        p_lev_oos  = lev_avg_is.p_avg_oos,
        p_ak5_is   = sim_pvalue(obs_ak_is_5,  ak5_is),
        p_ak21_is  = sim_pvalue(obs_ak_is_21, ak21_is),
        p_ak5_oos  = sim_pvalue(obs_ak_oos_5, ak5_oos),
        p_ak21_oos = sim_pvalue(obs_ak_oos_21, ak21_oos),
    );
    pv_joint_is = mean([pv.p_kurt_is, pv.p_lev_is, pv.p_ak5_is, pv.p_ak21_is]);
    pv_joint_oos = mean([pv.p_kurt_oos, pv.p_lev_oos, pv.p_ak5_oos, pv.p_ak21_oos]);
    pv_results[m] = merge(pv, (pv_joint_is=pv_joint_is, pv_joint_oos=pv_joint_oos));
    println("  $m  pv̄ IS=$(round(pv_joint_is, digits=3))  pv̄ OoS=$(round(pv_joint_oos, digits=3))");
end

# --------------------------------------------------------------------------------------- #
# Write outputs
# --------------------------------------------------------------------------------------- #
println("\n[output] Writing Track A tables...");

# Table 4 Extended: headline panel combining existing seven metrics with A1/A2/A3
open(joinpath(TRACK_A_DIR, "Table-4-Extended-Metrics.txt"), "w") do io
    println(io, "="^130);
    println(io, "TABLE 4 (Track A extension). MMD, signature-MMD, and discriminator AUC for SPY generators.");
    println(io, "="^130);
    println(io, "");
    println(io, "Setup   : SPY daily log excess growth; IS n=$n_is, OoS n=$n_oos; seed=$SEED; N_paths=$N_PATHS.");
    println(io, "Windows : length W=$WINDOW_LEN days; observed windows IS $n_obs_win_is, OoS $n_obs_win_oos; stride auto.");
    println(io, "MMD     : RBF kernel, unbiased estimator, bandwidth by median heuristic on pooled sample.");
    println(io, "sig-MMD : depth-$SIG_DEPTH truncated signature of time-augmented 2D path (t/W, r_t / std(r)).");
    println(io, "Disc AUC: logistic regression on 10 hand-crafted window features, 5-fold CV.");
    println(io, "");
    println(io, "Reading: lower MMD / sig-MMD is better; disc AUC near 0.50 means indistinguishable from real.");
    println(io, "-"^130);
    println(io, rpad("Model", 12), " | ",
                rpad("MMD IS",   11), " | ", rpad("MMD OoS",  11), " | ",
                rpad("sig IS",   11), " | ", rpad("sig OoS",  11), " | ",
                rpad("AUC IS",   14), " | ", rpad("AUC OoS",  14));
    println(io, "-"^130);
    for m in MODEL_ORDER
        r = mmd_results[m];
        auc_is_s  = "$(round(r.auc_is,  digits=3)) ± $(round(r.auc_is_std,  digits=3))";
        auc_oos_s = "$(round(r.auc_oos, digits=3)) ± $(round(r.auc_oos_std, digits=3))";
        println(io, rpad(m, 12), " | ",
                    rpad(round(r.mmd_is,  digits=5), 11), " | ",
                    rpad(round(r.mmd_oos, digits=5), 11), " | ",
                    rpad(round(r.sig_is,  digits=5), 11), " | ",
                    rpad(round(r.sig_oos, digits=5), 11), " | ",
                    rpad(auc_is_s, 14), " | ", rpad(auc_oos_s, 14));
    end
    println(io, "="^130);
end

open(joinpath(TRACK_A_DIR, "leverage_effect.txt"), "w") do io
    println(io, "="^110);
    println(io, "A6. Leverage effect: corr(r_t^2, r_{t-k}) profile for k=1..$MAX_LAG_LEV, down/up asymmetry");
    println(io, "="^110);
    println(io, "");
    println(io, "Observed IS  avg corr over k=1..$MAX_LAG_LEV = $(round(lev_obs_is.avg_neg, digits=4))   |E(|r_{t+1}| | r_t<0) - E(|r_{t+1}| | r_t>=0)| = $(round(lev_obs_is.asymmetry, digits=5))");
    println(io, "Observed OoS avg corr over k=1..$MAX_LAG_LEV = $(round(lev_obs_oos.avg_neg, digits=4))   (asymmetry) = $(round(lev_obs_oos.asymmetry, digits=5))");
    println(io, "");
    println(io, rpad("Model", 12), " | ",
                rpad("avg_neg IS",  12), " | ", rpad("asym IS",   12), " | ",
                rpad("pv avg IS",   10), " | ",
                rpad("avg_neg OoS", 12), " | ", rpad("asym OoS",  12), " | ",
                rpad("pv avg OoS",  10));
    println(io, "-"^110);
    for m in MODEL_ORDER
        r = lev_results[m];
        println(io, rpad(m, 12), " | ",
                    rpad(round(r.med_is,       digits=4), 12), " | ",
                    rpad(round(r.med_asym_is,  digits=5), 12), " | ",
                    rpad(round(r.p_avg_is,     digits=3), 10), " | ",
                    rpad(round(r.med_oos,      digits=4), 12), " | ",
                    rpad(round(r.med_asym_oos, digits=5), 12), " | ",
                    rpad(round(r.p_avg_oos,    digits=3), 10));
    end
    println(io, "="^110);
    println(io, "");
    println(io, "Interpretation: a negative avg_neg is the leverage effect (past negative returns predict");
    println(io, "larger subsequent squared returns). The pv column is a two-sided sim-based p-value under");
    println(io, "the model; large pv means the model's distribution covers the observed leverage value.");
end

open(joinpath(TRACK_A_DIR, "aggregational_kurtosis.txt"), "w") do io
    println(io, "="^110);
    println(io, "A7. Aggregational kurtosis at horizons $HORIZONS_AG");
    println(io, "="^110);
    println(io, "");
    println(io, "Observed IS  : ", join(["h=$(h) k=$(round(agg_obs_is[h], digits=2))" for h in HORIZONS_AG], "   "));
    println(io, "Observed OoS : ", join(["h=$(h) k=$(round(agg_obs_oos[h], digits=2))" for h in HORIZONS_AG], "   "));
    println(io, "");
    println(io, "Median simulated kurtosis (IS over $N_PATHS paths)");
    println(io, "-"^110);
    println(io, rpad("Model", 12), " | ", join([rpad("h=$h",8) for h in HORIZONS_AG], " | "));
    println(io, "-"^110);
    for m in MODEL_ORDER
        per_h = agg_results[m][:is];
        row = [rpad(m, 12)];
        for h in HORIZONS_AG
            push!(row, rpad(round(median(per_h[h]), digits=2), 8));
        end
        println(io, join(row, " | "));
    end
    println(io, "");
    println(io, "Median simulated kurtosis (OoS over $N_PATHS paths)");
    println(io, "-"^110);
    println(io, rpad("Model", 12), " | ", join([rpad("h=$h",8) for h in HORIZONS_AG], " | "));
    println(io, "-"^110);
    for m in MODEL_ORDER
        per_h = agg_results[m][:oos];
        row = [rpad(m, 12)];
        for h in HORIZONS_AG
            push!(row, rpad(round(median(per_h[h]), digits=2), 8));
        end
        println(io, join(row, " | "));
    end
    println(io, "="^110);
end

open(joinpath(TRACK_A_DIR, "sim_pvalues.txt"), "w") do io
    println(io, "="^130);
    println(io, "A9. Simulation-based two-sided p-values and joint-coverage summary pv̄");
    println(io, "="^130);
    println(io, "");
    println(io, "pv̄ is the mean of the four per-stat p-values {kurtosis, avg leverage, agg kurt h=5, agg kurt h=21}.");
    println(io, "A well-calibrated generator achieves pv̄ near 0.5; pv̄=0 means the generator fails joint coverage.");
    println(io, "");
    println(io, rpad("Model", 12), " | ",
                rpad("p_kurt",   7), " | ", rpad("p_acf", 7), " | ",
                rpad("p_lev",    7), " | ", rpad("p_ak5", 7), " | ",
                rpad("p_ak21",   7), " | ", rpad("pv̄ IS", 7), " | ", rpad("pv̄ OoS", 7));
    println(io, "-"^130);
    for m in MODEL_ORDER
        pv = pv_results[m];
        println(io, rpad(m, 12), " | ",
                    rpad(round(pv.p_kurt_is,  digits=3), 7), " | ",
                    rpad(round(pv.p_acf_is,   digits=3), 7), " | ",
                    rpad(round(pv.p_lev_is,   digits=3), 7), " | ",
                    rpad(round(pv.p_ak5_is,   digits=3), 7), " | ",
                    rpad(round(pv.p_ak21_is,  digits=3), 7), " | ",
                    rpad(round(pv.pv_joint_is, digits=3), 7), " | ",
                    rpad(round(pv.pv_joint_oos,digits=3), 7));
    end
    println(io, "="^130);
end

# --------------------------------------------------------------------------------------- #
# Summary digest
# --------------------------------------------------------------------------------------- #
open(joinpath(TRACK_A_DIR, "Track-A-summary.txt"), "w") do io
    println(io, "Track A summary digest (extended distributional and tail-fidelity metrics)");
    println(io, "="^90);
    println(io, "");
    println(io, "Setup:  SPY, IS n=$n_is, OoS n=$n_oos; seed=$SEED; $N_PATHS paths/model; K=$K_MAIN.");
    println(io, "");
    println(io, "A1  MMD(RBF)     - lower is better. Best IS: $(argmin([mmd_results[m].mmd_is for m in MODEL_ORDER]) isa Integer ? MODEL_ORDER[argmin([mmd_results[m].mmd_is for m in MODEL_ORDER])] : "-").");
    println(io, "A2  sig-MMD (d=$SIG_DEPTH) - lower is better. Best IS: $(MODEL_ORDER[argmin([mmd_results[m].sig_is for m in MODEL_ORDER])]).");
    println(io, "A3  Disc AUC     - closer to 0.50 is better. Best IS: $(MODEL_ORDER[argmin([abs(mmd_results[m].auc_is - 0.5) for m in MODEL_ORDER])]) (AUC $(round(mmd_results[MODEL_ORDER[argmin([abs(mmd_results[m].auc_is - 0.5) for m in MODEL_ORDER])]].auc_is, digits=3))).");
    println(io, "A6  Leverage avg_neg observed IS $(round(lev_obs_is.avg_neg, digits=4)); best coverage (p_lev): $(MODEL_ORDER[argmax([lev_results[m].p_avg_is for m in MODEL_ORDER])]).");
    println(io, "A7  Aggregational kurt observed IS: ", join(["h=$(h) k=$(round(agg_obs_is[h], digits=2))" for h in HORIZONS_AG], "  "), ".");
    println(io, "A9  pv̄ IS by model:");
    for m in MODEL_ORDER
        println(io, "     $(rpad(m, 12)) pv̄=$(round(pv_results[m].pv_joint_is, digits=3))   pv̄ OoS=$(round(pv_results[m].pv_joint_oos, digits=3))");
    end
    println(io, "");
    println(io, "Per-table details:");
    println(io, "  - Table-4-Extended-Metrics.txt      MMD / sig-MMD / disc AUC panel");
    println(io, "  - leverage_effect.txt               A6 profile and asymmetry");
    println(io, "  - aggregational_kurtosis.txt        A7 kurtosis at horizons $HORIZONS_AG");
    println(io, "  - sim_pvalues.txt                   A9 p-values and joint-coverage pv̄");
end

println("\n" * "="^72);
println("  Track A metrics run complete.");
println("  Results: $TRACK_A_DIR");
println("="^72);

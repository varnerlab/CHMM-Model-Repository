# ========================================================================================= #
# run_acf_horizon_diagnostics.jl
#
# Third-review item 8: the paper's ACF claim rested on a single lag-252 |G_t| ACF-MAE
# number (CHMM-N 0.0462 vs an i.i.d. floor of ~0.063). This runner makes the horizon
# structure explicit for the main text by producing, for CHMM-N at K* = 3 on SPY IS:
#
#   1. the observed IS |G_t| sample ACF (lags 1..252) with a moving-block-bootstrap
#      95% confidence band (block length L = 20, 1000 replicates — the paper's L = 20
#      block convention). The band is reported ONLY for lags <= BAND_MAX = 5, well
#      inside the block length: block-boundary contamination is gradual, not a sharp
#      cutoff at L — at lag h roughly h/L of resampled pairs cross an independently
#      concatenated block boundary, and at h = L every pair does — so a conservative
#      low-lag display range is required (fifth-review finding 3, superseding the
#      fourth-review lags <= L convention). Beyond lag BAND_MAX the CSV records NaN
#      and the figure omits the band;
#   2. a simulated-ACF band: the per-path sample ACF over 1000 IS-length simulated
#      paths (median and 5-95% envelope), same estimator as run_kstar3_headline.jl
#      (StatsBase.autocor of |path| at lags 1..252);
#   3. the THEORETICAL population ACF from the matrix formula (theory.tex identity)
#         rho(tau) = (m' diag(pibar) T^tau m - (pibar'm)^2) / sigma^2_|G|,
#      with per-state folded moments m_k = E[|G| | s=k], M_k = E[G^2 | s=k] sampled
#      exactly as in runners/spectral/spectral_common.jl (n_draw = 200,000);
#   4. horizon-banded ACF MAE (lags 1-5, 6-20, 21-63, 64-252) for CHMM-N
#      simulated-vs-observed and an i.i.d.-permutation baseline (1000 shuffles of the
#      IS series) so each band has its floor.
#
# Seeding: SEED = 20260420 for the fit (as in run_kstar3_headline.jl), SEED + 1 for
# the 1000-path simulation, SEED + 2 for the observed-ACF block bootstrap, SEED + 3
# for the i.i.d. permutations. Fit protocol (data pipeline, K* = 3, max_iter = 60)
# matches run_kstar3_headline.jl exactly.
#
# Output:
#   results/acf_horizon/acf_horizon.txt   (band table; theoretical ACF at reference
#                                          lags; dominant eigenvalue + half-life)
#   results/acf_horizon/acf_curves.csv    (lag, observed, obs_ci_lo, obs_ci_hi,
#                                          sim_median, sim_lo, sim_hi, theoretical)
#   ../CHMM-Paper-Repository/figs/Fig-ACF-Observed-vs-Simulated.pdf  (main-text figure)
#
# Usage: julia --project=. runners/headline/run_acf_horizon_diagnostics.jl
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include(joinpath(@__DIR__, "..", "..", "Include.jl"));
include(joinpath(@__DIR__, "..", "spectral", "spectral_common.jl"));
using Random, Statistics, StatsBase, Printf;

const TICKER         = "SPY";
const RISK_FREE_RATE = 0.0;
const ΔT             = 1/252;
const MAX_ITER       = 60;                 # matches run_kstar3_headline.jl
const N_PATHS        = 1000;               # matches run_kstar3_headline.jl
const L_MAX          = 252;                # max ACF lag
const K_STAR         = 3;
const SEED           = 20260420;           # paper-canonical global seed
const BLOCK_LEN      = 20;                 # paper's L = 20 block-bootstrap convention
const N_BOOT         = 1000;               # moving-block-bootstrap replicates
const N_M_DRAW       = 200_000;            # per-state draws for m_k, M_k (spectral runners)
const BANDS          = [(1, 5), (6, 20), (21, 63), (64, 252)];
const REF_LAGS       = [1, 5, 20, 63, 126, 252];

const OUT_DIR        = joinpath(_ROOT, "results", "acf_horizon");
const PAPER_FIGS_DIR = joinpath(dirname(_ROOT), "CHMM-Paper-Repository", "figs");
mkpath(OUT_DIR);
mkpath(PAPER_FIGS_DIR);

println("="^78)
println("  ACF horizon diagnostics: CHMM-N at K* = $K_STAR on $TICKER IS")
println("  Seed: $SEED   Paths: $N_PATHS   Bootstrap: L = $BLOCK_LEN, B = $N_BOOT")
println("="^78)

# ========================================================================================= #
# [1/6] Load SPY IS data (identical pipeline to run_kstar3_headline.jl)
# ========================================================================================= #
println("\n[1/6] Loading SPY data...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) ∈ train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
list_of_all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, list_of_all_tickers; Δt=ΔT, risk_free_rate=RISK_FREE_RATE);
idx_spy = findfirst(x -> x == TICKER, list_of_all_tickers);
R_is = all_R[:, idx_spy];
n_is = length(R_is);
println("  IS: $n_is obs")

τ = 1:L_MAX;
acf_obs = autocor(abs.(R_is), τ);          # same estimator as run_kstar3_headline.jl

# ========================================================================================= #
# [2/6] Fit CHMM-N at K* = 3 and simulate 1000 IS-length paths
# ========================================================================================= #
println("\n[2/6] Fitting CHMM-N at K* = $K_STAR (seed $SEED, max_iter $MAX_ITER)...")
Random.seed!(SEED);
model = build(MyContinuousHiddenMarkovModel,
    (observations=R_is, number_of_states=K_STAR, max_iter=MAX_ITER));

T_mat = zeros(K_STAR, K_STAR);
for i in 1:K_STAR; T_mat[i, :] = probs(model.transition[i]); end
π_stat = _stationary_pi(T_mat);   # checked left-eigenvector solve (spectral_common.jl)
start_dist = Categorical(π_stat);

println("  Simulating $N_PATHS IS-length paths (seed $(SEED + 1))...")
Random.seed!(SEED + 1);
sim_is = Array{Float64,2}(undef, n_is, N_PATHS);
for i in 1:N_PATHS
    s0 = rand(start_dist);
    st = model(s0, n_is);
    for j in 1:n_is; sim_is[j, i] = rand(model.emission[st[j]]); end
end

# Per-path sample ACF of |path| (headline estimator), median + 5-95% envelope
println("  Computing per-path |G_t| sample ACFs...")
acf_sim = Array{Float64,2}(undef, L_MAX, N_PATHS);
for i in 1:N_PATHS
    acf_sim[:, i] = autocor(abs.(sim_is[:, i]), τ);
end
sim_median = [quantile(acf_sim[t, :], 0.50) for t in 1:L_MAX];
sim_lo     = [quantile(acf_sim[t, :], 0.05) for t in 1:L_MAX];
sim_hi     = [quantile(acf_sim[t, :], 0.95) for t in 1:L_MAX];

# ========================================================================================= #
# [3/6] Theoretical population ACF from the matrix formula + dominant eigenvalue
# ========================================================================================= #
println("\n[3/6] Theoretical population ACF (matrix formula, n_draw = $N_M_DRAW)...")
T, π̄, m, M = _T_pibar_m(model, K_STAR; n_draw=N_M_DRAW, seed=SEED);
σ²_G = π̄' * M - (π̄' * m)^2;
μ_G  = π̄' * m;
acf_theo = zeros(L_MAX);
let Tτ = Matrix{Float64}(I, K_STAR, K_STAR)
    for t in 1:L_MAX
        Tτ *= T;
        acf_theo[t] = (m' * Diagonal(π̄) * Tτ * m - μ_G^2) / σ²_G;
    end
end

λ_all = eigvals(T);
idx_unit = argmin(abs.(λ_all .- 1.0));
λ_rest = λ_all[setdiff(1:K_STAR, idx_unit)];
λ2_abs = maximum(abs.(λ_rest));
half_life = log(0.5) / log(λ2_abs);
println(@sprintf("  |lambda_2| = %.4f   half-life = %.1f trading days", λ2_abs, half_life))

# ========================================================================================= #
# [4/6] Moving-block bootstrap 95% CI for the OBSERVED sample ACF (L = 20, B = 1000)
# ========================================================================================= #
println("\n[4/6] Moving-block bootstrap for the observed ACF (seed $(SEED + 2))...")
Random.seed!(SEED + 2);
n_blocks = ceil(Int, n_is / BLOCK_LEN);
acf_boot = Array{Float64,2}(undef, L_MAX, N_BOOT);
boot_series = Vector{Float64}(undef, n_blocks * BLOCK_LEN);
for b in 1:N_BOOT
    for k in 1:n_blocks
        s = rand(1:(n_is - BLOCK_LEN + 1));
        boot_series[(k-1)*BLOCK_LEN+1 : k*BLOCK_LEN] = R_is[s : s+BLOCK_LEN-1];
    end
    acf_boot[:, b] = autocor(abs.(boot_series[1:n_is]), τ);
end
obs_ci_lo = [quantile(acf_boot[t, :], 0.025) for t in 1:L_MAX];
obs_ci_hi = [quantile(acf_boot[t, :], 0.975) for t in 1:L_MAX];
# Validity restriction (fifth-review finding 3, superseding the fourth-review lags <= L
# convention): block-boundary contamination is GRADUAL, not a sharp cutoff at L. At
# lag h, roughly h/L of the resampled lag-h pairs cross an independently-concatenated
# block boundary; at h = L every pair does. The band is therefore displayed only over
# a conservative low-lag range h <= BAND_MAX << L, where the contaminated fraction is
# small. Record NaN beyond BAND_MAX so no downstream consumer can plot or cite the
# band outside that range.
const BAND_MAX = 5;
for t in (BAND_MAX + 1):L_MAX
    obs_ci_lo[t] = NaN;
    obs_ci_hi[t] = NaN;
end

# ========================================================================================= #
# [5/6] Horizon-banded ACF MAE: CHMM-N vs the i.i.d.-permutation floor
# ========================================================================================= #
println("\n[5/6] Banded ACF MAE (CHMM-N + i.i.d. permutation floor, seed $(SEED + 3))...")
Random.seed!(SEED + 3);
acf_perm = Array{Float64,2}(undef, L_MAX, N_PATHS);
for i in 1:N_PATHS
    acf_perm[:, i] = autocor(abs.(shuffle(R_is)), τ);
end

# Banded MAE = mean over paths of mean(|acf_obs - acf_path|) over the band's lags,
# the same averaging convention as the lag-252 acf_mae in run_kstar3_headline.jl.
_band_mae(acf_paths, lo, hi) =
    mean(mean(abs.(acf_obs[lo:hi] .- acf_paths[lo:hi, i])) for i in 1:size(acf_paths, 2));

band_rows = NamedTuple[];
for (lo, hi) in BANDS
    push!(band_rows, (band="$lo-$hi",
                      chmm=_band_mae(acf_sim, lo, hi),
                      iid=_band_mae(acf_perm, lo, hi)));
end
mae_all_chmm = _band_mae(acf_sim, 1, L_MAX);
mae_all_iid  = _band_mae(acf_perm, 1, L_MAX);

# ========================================================================================= #
# [6/6] Write artefacts: txt table, CSV curves, main-text figure
# ========================================================================================= #
println("\n[6/6] Writing artefacts...")

open(joinpath(OUT_DIR, "acf_horizon.txt"), "w") do io
    println(io, "="^90);
    println(io, "ACF horizon diagnostics: CHMM-N at K* = $K_STAR on $TICKER IS");
    println(io, "="^90);
    println(io);
    println(io, "Setup: IS = $n_is obs, seed = $SEED (fit), seed+1 (sim, $N_PATHS paths),");
    println(io, "seed+2 (moving-block bootstrap, block length $BLOCK_LEN, $N_BOOT replicates),");
    println(io, "seed+3 (i.i.d. permutation floor, $N_PATHS shuffles). ACF estimator:");
    println(io, "StatsBase.autocor of |G_t| at lags 1..$L_MAX, as in run_kstar3_headline.jl.");
    println(io);
    println(io, "Observed-ACF bootstrap band validity: the L = $BLOCK_LEN moving-block bootstrap");
    println(io, "severs dependence across block boundaries, and the contamination is gradual —");
    println(io, "at lag h roughly h/L of resampled pairs cross a boundary, and at h = L every");
    println(io, "pair does — so the percentile band is displayed only over the conservative");
    println(io, "low-lag range lags <= $BAND_MAX (well inside the block length). Entries beyond lag");
    println(io, "$BAND_MAX are NaN in the CSV and omitted from the figure. No claim at any longer");
    println(io, "lag is made from this band; the horizon comparison below rests on the banded");
    println(io, "MAE table alone.");
    println(io);
    println(io, "Horizon-banded |G_t| ACF MAE (mean over paths of the band-mean absolute error):");
    println(io, "-"^60);
    @printf(io, "%-12s %14s %16s\n", "Band (lags)", "CHMM-N MAE", "i.i.d. floor MAE");
    println(io, "-"^60);
    for r in band_rows
        @printf(io, "%-12s %14.4f %16.4f\n", r.band, r.chmm, r.iid);
    end
    @printf(io, "%-12s %14.4f %16.4f\n", "1-252 (all)", mae_all_chmm, mae_all_iid);
    println(io, "-"^60);
    println(io);
    println(io, "Theoretical population ACF (matrix formula, n_draw = $N_M_DRAW per state):");
    println(io, "  rho(tau) = (m' diag(pibar) T^tau m - (pibar'm)^2) / sigma^2_|G|");
    @printf(io, "  sigma^2_|G| = %.6f\n", σ²_G);
    for l in REF_LAGS
        @printf(io, "  rho(%3d) = %+.6f\n", l, acf_theo[l]);
    end
    println(io);
    @printf(io, "Dominant non-unit eigenvalue |lambda_2| = %.4f  =>  half-life = %.1f trading days\n",
            λ2_abs, half_life);
    println(io, "(lambda_2^252 ~ $(@sprintf("%.2e", λ2_abs^252)): the population ACF is ~0 at lag 252;");
    println(io, "long-horizon 'slow decay' credit in the flat lag-1..252 MAE comes from the model");
    println(io, "and the observed ACF both being near zero there, not from matched persistence.)");
    println(io);
    println(io, "Source: results/acf_horizon/acf_curves.csv (run_acf_horizon_diagnostics.jl)");
    println(io, "Figure: figs/Fig-ACF-Observed-vs-Simulated.pdf (paper repo)");
end

open(joinpath(OUT_DIR, "acf_curves.csv"), "w") do io
    println(io, "lag,observed,obs_ci_lo,obs_ci_hi,sim_median,sim_lo,sim_hi,theoretical");
    for t in 1:L_MAX
        @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                t, acf_obs[t], obs_ci_lo[t], obs_ci_hi[t],
                sim_median[t], sim_lo[t], sim_hi[t], acf_theo[t]);
    end
end

# ----- Main-text figure (style of run_figures.jl: Okabe-Ito palette, (700, 500) panel) -----
const TFS = 11;
const _OBS_C  = RGB(0.835, 0.369, 0.0);     # Okabe-Ito vermillion (observed)
const _SIM_C  = RGB(0.0, 0.447, 0.698);     # Okabe-Ito blue (theoretical curve)
const _MEAN_C = RGB(0.0, 0.620, 0.451);     # Okabe-Ito bluish green (simulated)
const _STYLE = (titlefontsize=TFS, guidefontsize=TFS, tickfontsize=TFS-1,
                legendfontsize=TFS-2);

p = plot(xlabel="Lag (trading days, log scale)", ylabel="ACF of |Gₜ|",
    xscale=:log10, xticks=([1, 5, 20, 63, 126, 252], ["1", "5", "20", "63", "126", "252"]),
    xlims=(1, 252), size=(700, 500), legend=:topright; _STYLE...);
# Observed moving-block-bootstrap 95% band — plotted only over its validity range
# (lags <= BLOCK_LEN); beyond the block length the band is not a valid uncertainty
# statement and is omitted.
plot!(p, 1:BAND_MAX, obs_ci_lo[1:BAND_MAX], fillrange=obs_ci_hi[1:BAND_MAX],
    alpha=0.25, color=_OBS_C, lw=0,
    label="Observed 95% CI (block bootstrap, ℓ = $BLOCK_LEN; shown to lag $BAND_MAX)");
# Simulated 5-95% envelope + median
plot!(p, τ, sim_lo, fillrange=sim_hi, alpha=0.20, color=_MEAN_C, lw=0,
    label="CHMM-N 5–95th pctl ($N_PATHS paths)");
plot!(p, τ, sim_median, lw=2, color=_MEAN_C, label="CHMM-N (median path ACF)");
# Observed sample ACF
plot!(p, τ, acf_obs, lw=2, color=_OBS_C, ls=:dash, label="Observed");
# Theoretical population ACF
plot!(p, τ, acf_theo, lw=2.5, color=_SIM_C, label="Population ACF (matrix formula)");
hline!(p, [0.0], lw=1, color=:gray, ls=:dot, label="");
fig_path = joinpath(PAPER_FIGS_DIR, "Fig-ACF-Observed-vs-Simulated.pdf");
savefig(p, fig_path);
savefig(p, joinpath(OUT_DIR, "Fig-ACF-Observed-vs-Simulated.svg"));

# ----- Console report -----
println("\nBand table (|G_t| ACF MAE):");
@printf("  %-12s %12s %14s\n", "Band", "CHMM-N", "iid floor");
for r in band_rows
    @printf("  %-12s %12.4f %14.4f\n", r.band, r.chmm, r.iid);
end
@printf("  %-12s %12.4f %14.4f\n", "1-252 (all)", mae_all_chmm, mae_all_iid);
println("\nTheoretical ACF at reference lags:");
for l in REF_LAGS
    @printf("  rho(%3d) = %+.6f\n", l, acf_theo[l]);
end
@printf("\n|lambda_2| = %.4f   half-life = %.1f days\n", λ2_abs, half_life);
println("\n[done] $(joinpath(OUT_DIR, "acf_horizon.txt"))");
println("[done] $(joinpath(OUT_DIR, "acf_curves.csv"))");
println("[done] $fig_path");

# ========================================================================================= #
# run_leverage_effect.jl
#
# Leverage-effect column for the headline panel: Corr(G_t, |G_{t+1}|) on observed
# (IS, OoS) and per-path on each generator. Cont (2001) stylized fact: equity returns
# carry a negative correlation between today's signed return and tomorrow's absolute
# return. CHMM with symmetric per-state emissions cannot reproduce this by construction
# at the per-state level; the question is whether the regime-mixing weights pick up any
# signal, and whether the asymmetric GARCH variants (EGARCH, GJR-GARCH) close the gap.
#
# Outputs:
#   results/diagnostics/leverage_effect/leverage_effect.txt
#   results/diagnostics/leverage_effect/leverage_effect.csv
#   ../CHMM-paper/results/robustness/leverage_effect.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, StatsBase, Printf
const SEED = 20260420;
Random.seed!(SEED);

const TICKER    = "SPY";
const RISK_FREE = 0.0;
const DT        = 1/252;
const K_MAIN    = 18;
const MAX_ITER  = 60;
const N_PATHS   = 500;

const OUT_DIR              = joinpath(_ROOT, "results", "diagnostics", "leverage_effect");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^72)
println("  Leverage-effect Corr(G_t, |G_{t+1}|) panel")
println("  Seed $SEED, K=$K_MAIN, N_paths=$N_PATHS")
println("="^72)

# --------------------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------------------- #
println("\n[data] Loading SPY IS / OoS...")
train_dataset = MyPortfolioDataSet() |> x -> x["dataset"];
max_days = nrow(train_dataset["AAPL"]);
dataset = Dict{String,DataFrame}();
for (t, data) in train_dataset
    if nrow(data) == max_days; dataset[t] = data; end
end
all_tickers = keys(dataset) |> collect |> sort;
all_R = log_growth_matrix(dataset, all_tickers; Δt=DT, risk_free_rate=RISK_FREE);
idx_spy = findfirst(==("SPY"), all_tickers);
R_is  = Vector{Float64}(all_R[:, idx_spy]);
n_is  = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos")

# --------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------- #
"""
    leverage(x) -> Float64

Cont (2001) leverage effect: ρ(x_t, |x_{t+1}|) at lag 1.
"""
function leverage(x::AbstractVector{<:Real})::Float64
    n = length(x);
    n < 3 && return NaN;
    a = @view x[1:end-1];
    b = abs.(@view x[2:end]);
    return cor(a, b);
end

function leverage_per_path(sim_archive::AbstractMatrix)
    np = size(sim_archive, 2);
    out = Vector{Float64}(undef, np);
    for i in 1:np; out[i] = leverage(sim_archive[:, i]); end
    return out;
end

function summarise(name::String, vals::AbstractVector{Float64})
    q = quantile(vals, [0.05, 0.25, 0.5, 0.75, 0.95]);
    return (name=name, mean=mean(vals), median=q[3], q05=q[1], q95=q[5]);
end

function fmt(s::NamedTuple)
    return @sprintf("%-26s mean=%+.4f  median=%+.4f  [Q5,Q95]=[%+.4f,%+.4f]",
                    s.name, s.mean, s.median, s.q05, s.q95);
end

# --------------------------------------------------------------------------------------- #
# Observed
# --------------------------------------------------------------------------------------- #
lev_obs_is  = leverage(R_is);
lev_obs_oos = leverage(R_oos);
println(@sprintf("\n[observed] IS  Corr(G_t, |G_{t+1}|) = %+.4f", lev_obs_is));
println(@sprintf("[observed] OoS Corr(G_t, |G_{t+1}|) = %+.4f", lev_obs_oos));

# --------------------------------------------------------------------------------------- #
# Generators: train + simulate
# --------------------------------------------------------------------------------------- #
function _simulate_chmm_paths(model, n_is, n_oos, n_paths)
    sim_is  = simulate_returns(model, n_is;  n_paths=n_paths);
    sim_oos = simulate_returns(model, n_oos; n_paths=n_paths);
    return sim_is, sim_oos;
end

println("\n[chmm] Training CHMM-N / -t / -L / -GED at K=$K_MAIN on SPY IS...")
chmm_n   = build(MyContinuousHiddenMarkovModel, (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
chmm_t   = build(MyStudentTHiddenMarkovModel,   (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
chmm_l   = build(MyLaplaceHiddenMarkovModel,    (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
chmm_ged = build(MyGEDHiddenMarkovModel,        (observations=R_is, number_of_states=K_MAIN, max_iter=MAX_ITER));
println("  CHMM training done.")

println("\n[chmm] Simulating $N_PATHS paths from each family...")
sim_n_is,   sim_n_oos   = _simulate_chmm_paths(chmm_n,   n_is, n_oos, N_PATHS);
sim_t_is,   sim_t_oos   = _simulate_chmm_paths(chmm_t,   n_is, n_oos, N_PATHS);
sim_l_is,   sim_l_oos   = _simulate_chmm_paths(chmm_l,   n_is, n_oos, N_PATHS);
sim_ged_is, sim_ged_oos = _simulate_chmm_paths(chmm_ged, n_is, n_oos, N_PATHS);
println("  CHMM paths ready.")

# Bootstrap (i.i.d.; no leverage effect by construction; sanity-check baseline)
println("\n[boot] iid bootstrap of IS series...")
boot_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
boot_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    boot_is[:,  i] = R_is[rand(1:n_is, n_is)];
    boot_oos[:, i] = R_is[rand(1:n_is, n_oos)];
end

# GARCH(1,1) Gaussian
println("\n[garch] Fitting GARCH(1,1) Gaussian + simulating...")
garch_model = build(MyGARCHModel, (observations=R_is,));
garch_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
garch_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    garch_is[:,  i] = simulate_garch(garch_model, n_is);
    garch_oos[:, i] = simulate_garch(garch_model, n_oos);
end

# Asymmetric GARCH: EGARCH and GJR-GARCH
println("\n[asym] Fitting EGARCH + GJR-GARCH + simulating...")
egarch_fit = fit_egarch11(R_is);
gjr_fit    = fit_gjr11(R_is);
egarch_is  = Array{Float64,2}(undef, n_is,  N_PATHS);
egarch_oos = Array{Float64,2}(undef, n_oos, N_PATHS);
gjr_is     = Array{Float64,2}(undef, n_is,  N_PATHS);
gjr_oos    = Array{Float64,2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    egarch_is[:,  i] = simulate_egarch(egarch_fit, n_is);
    egarch_oos[:, i] = simulate_egarch(egarch_fit, n_oos);
    gjr_is[:,  i]    = simulate_gjr(gjr_fit,    n_is);
    gjr_oos[:, i]    = simulate_gjr(gjr_fit,    n_oos);
end

# --------------------------------------------------------------------------------------- #
# Compute leverage per generator (IS and OoS)
# --------------------------------------------------------------------------------------- #
println("\n[compute] Per-path leverage Corr(G_t, |G_{t+1}|)...")
rows = NamedTuple[]
for (name, sim_is_arr, sim_oos_arr) in [
        ("Bootstrap",          boot_is,    boot_oos),
        ("GARCH(1,1) Gaussian", garch_is,   garch_oos),
        ("EGARCH(1,1)",         egarch_is,  egarch_oos),
        ("GJR-GARCH(1,1)",      gjr_is,     gjr_oos),
        ("CHMM-N (K=18)",       sim_n_is,   sim_n_oos),
        ("CHMM-t (K=18)",       sim_t_is,   sim_t_oos),
        ("CHMM-L (K=18)",       sim_l_is,   sim_l_oos),
        ("CHMM-GED (K=18)",     sim_ged_is, sim_ged_oos),
    ]
    lev_is  = leverage_per_path(sim_is_arr);
    lev_oos = leverage_per_path(sim_oos_arr);
    s_is  = summarise(name * " IS",  lev_is);
    s_oos = summarise(name * " OoS", lev_oos);
    push!(rows, s_is); push!(rows, s_oos);
end

# --------------------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------------------- #
out_txt = joinpath(OUT_DIR, "leverage_effect.txt");
out_csv = joinpath(OUT_DIR, "leverage_effect.csv");
out_paper_csv = joinpath(PAPER_ROBUSTNESS_DIR, "leverage_effect.csv");

open(out_txt, "w") do io
    println(io, "Leverage effect Corr(G_t, |G_{t+1}|): SPY $TICKER")
    println(io, "Seed $SEED, K=$K_MAIN, N_paths=$N_PATHS")
    println(io, "IS T = $n_is, OoS T = $n_oos")
    println(io, "")
    println(io, @sprintf("Observed IS  : %+.4f", lev_obs_is))
    println(io, @sprintf("Observed OoS : %+.4f", lev_obs_oos))
    println(io, "")
    println(io, "Per-generator distribution (mean, median, [Q5, Q95]) across $N_PATHS paths:")
    for r in rows
        println(io, fmt(r));
    end
    println(io, "")
    println(io, "Reading: a generator captures the Cont leverage effect when its [Q5, Q95]")
    println(io, "envelope brackets the observed value. Symmetric-emission generators")
    println(io, "(CHMM-N/-t/-L/-GED, GARCH(1,1) Gaussian, Bootstrap) are expected to bracket")
    println(io, "zero rather than the negative observed value; asymmetric GARCH variants")
    println(io, "(EGARCH, GJR-GARCH) are expected to bracket the negative observed value.")
end

open(out_csv, "w") do io
    println(io, "name,window,observed,gen_mean,gen_median,gen_q05,gen_q95")
    obs_map = Dict("IS" => lev_obs_is, "OoS" => lev_obs_oos)
    for r in rows
        # split name like "GARCH(1,1) Gaussian IS"
        nm = String(r.name)
        win = endswith(nm, " IS") ? "IS" : (endswith(nm, " OoS") ? "OoS" : "?");
        gen = strip(replace(nm, " IS" => "", " OoS" => ""));
        obs_v = obs_map[win];
        println(io, @sprintf("%s,%s,%+.6f,%+.6f,%+.6f,%+.6f,%+.6f",
                             gen, win, obs_v, r.mean, r.median, r.q05, r.q95));
    end
end

cp(out_csv, out_paper_csv; force=true);

println("\n[write] $out_txt");
println("[write] $out_csv");
println("[write] $out_paper_csv");
println("\nDone.")

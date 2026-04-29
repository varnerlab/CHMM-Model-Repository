# =========================================================================== #
# run_subdecade_validation.jl
#
# Peer-review item 6 (alternative to the blocked external 1994-2004 ingest):
# split the 2014-2024 SPY IS into two non-overlapping 5-year sub-windows and
# fit the CHMM scaffold on each. Tests whether the four-emission scaffold
# reproduces the headline KS / kurtosis / ACF triple on a non-2014-2024-full
# window without any retuning. This is the spirit of the "independent window"
# validation; the full pre-2014 cross-decade comparison is logged as needing
# external data ingest (Yahoo Finance / similar) which has not been authorised.
#
# Three configurations:
#   A. fit on 2014-2019 (5y), evaluate IS on 2014-2019, OoS on 2019-2024 (5y)
#   B. fit on 2019-2024 (5y), evaluate IS on 2019-2024, OoS on 2024-2026 (~2.5y)
#   C. (control) fit on 2014-2024 (10y), evaluate as in body.
#
# Output: results/subdecade_validation/subdecade_validation.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, StatsBase, HypothesisTests, Printf, Dates;

const SEED      = 20260420;
const N_PATHS   = 1000;
const MAX_ITER  = 60;
const ALPHA_KS  = 0.05;
const L_LAGS    = 252;
const DT        = 1/252;
const RISK_FREE = 0.0;
const TICKER    = "SPY";

const OUT_DIR = joinpath(_ROOT, "results", "subdecade_validation");
mkpath(OUT_DIR);

println("="^70);
println("  Item 6 (alt): sub-decade validation on the 2014-2024 SPY window");
println("="^70);

train_dataset = MyPortfolioDataSet()["dataset"];
oos_dataset   = MyOutOfSamplePortfolioDataSet()["dataset"];
df_train = train_dataset[TICKER];
df_oos   = oos_dataset[TICKER];
df_full  = vcat(df_train, df_oos; cols=:orderequal);
sort!(df_full, :timestamp);
@printf("[data] %s timeline: %s -> %s (%d rows)\n",
    TICKER, df_full[1, :timestamp], df_full[end, :timestamp], nrow(df_full));

function _slice_log_returns(df::DataFrame, t_start::Date, t_end_excl::Date; Δt::Float64=DT)
    ts = Date.(df.timestamp);
    mask = (ts .>= t_start) .& (ts .< t_end_excl);
    sub = df[mask, :];
    P = sub.volume_weighted_average_price;
    R = (1 / Δt) .* (log.(P[2:end] ./ P[1:end-1])) .- 0.0;
    return Vector{Float64}(R);
end

function _eval(R_obs::AbstractVector, sim::AbstractMatrix; L::Int=L_LAGS, α::Float64=ALPHA_KS)
    np = size(sim, 2);
    n_o = length(R_obs);
    L_use = min(L, n_o - 1);
    acf_o_abs = autocor(abs.(R_obs), 1:L_use);
    μ_o = mean(R_obs); σ_o = std(R_obs);
    kurt_o = sum(((R_obs .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    ks_pass = 0; kurt_s = 0.0; acf_mae_s = 0.0;
    for i in 1:np
        s = sim[:, i];
        if pvalue(ApproximateTwoSampleKSTest(R_obs, s)) >= α; ks_pass += 1; end
        μ_s = mean(s); σ_s = std(s);
        kurt_s += sum(((s .- μ_s) ./ σ_s).^4) / length(s) - 3.0;
        acf_mae_s += mean(abs.(autocor(abs.(s), 1:L_use) .- acf_o_abs));
    end
    return (ks_pct = round(100*ks_pass/np, digits=1),
            sim_kurt = round(kurt_s/np, digits=2),
            obs_kurt = round(kurt_o, digits=2),
            acf_mae = round(acf_mae_s/np, digits=4));
end

function _fit_eval(R_train::AbstractVector, R_eval::AbstractVector, K::Int, builder; kwargs...)
    Random.seed!(SEED + 7 * K);
    chmm = build(builder, (observations=R_train, number_of_states=K, max_iter=MAX_ITER, kwargs...));
    T_mat = zeros(K, K);
    for i in 1:K; T_mat[i, :] = probs(chmm.transition[i]); end
    π_bar = (T_mat^2000)[1, :];
    sd = Categorical(π_bar);
    Random.seed!(SEED + 11);
    sim = Matrix{Float64}(undef, length(R_eval), N_PATHS);
    for p in 1:N_PATHS
        s0 = rand(sd);
        st = chmm(s0, length(R_eval));
        for j in 1:length(R_eval); sim[j, p] = rand(chmm.emission[st[j]]); end
    end
    return _eval(R_eval, sim);
end

# Configurations
configs = [
    ("A: 2014-2019 IS, 2019-2024 OoS", Date(2014,1,1), Date(2019,1,1), Date(2019,1,1), Date(2024,1,1)),
    ("B: 2019-2024 IS, 2024-2026 OoS", Date(2019,1,1), Date(2024,1,1), Date(2024,1,1), Date(2026,5,1)),
    ("C: 2014-2024 IS, 2024-2026 OoS (control)", Date(2014,1,1), Date(2024,1,1), Date(2024,1,1), Date(2026,5,1)),
];

results = NamedTuple[];

for (name, ts, te, vs, ve) in configs
    println("\n" * "="^60);
    println("  Config: $name");
    println("="^60);
    R_train = _slice_log_returns(df_full, ts, te);
    R_oos   = _slice_log_returns(df_full, vs, ve);
    @printf("  T_train = %d   T_oos = %d\n", length(R_train), length(R_oos));

    for (label, K, builder, kw) in [
            ("CHMM-N (K=18)",       18, MyContinuousHiddenMarkovModel, ()),
            ("CHMM-N (K*=6)",        6, MyContinuousHiddenMarkovModel, ()),
            ("CHMM-N (K*=3)",        3, MyContinuousHiddenMarkovModel, ()),
            ("CHMM-t pen (K=18)",   18, MyStudentTHiddenMarkovModel,   (ν_shrink_rate = 20.0,)),
        ]
        is_panel  = _fit_eval(R_train, R_train, K, builder; kw...);
        oos_panel = _fit_eval(R_train, R_oos,   K, builder; kw...);
        @printf("  %-20s  IS  KS = %5.1f%%  kurt obs/sim = %5.2f / %5.2f  ACF-MAE = %.4f\n",
            label, is_panel.ks_pct, is_panel.obs_kurt, is_panel.sim_kurt, is_panel.acf_mae);
        @printf("  %-20s  OoS KS = %5.1f%%  kurt obs/sim = %5.2f / %5.2f  ACF-MAE = %.4f\n",
            "", oos_panel.ks_pct, oos_panel.obs_kurt, oos_panel.sim_kurt, oos_panel.acf_mae);
        push!(results, (config=name, model=label,
            T_train=length(R_train), T_oos=length(R_oos),
            is_ks=is_panel.ks_pct, oos_ks=oos_panel.ks_pct,
            is_obs_kurt=is_panel.obs_kurt, is_sim_kurt=is_panel.sim_kurt,
            oos_obs_kurt=oos_panel.obs_kurt, oos_sim_kurt=oos_panel.sim_kurt,
            is_acf=is_panel.acf_mae, oos_acf=oos_panel.acf_mae));
    end
end

open(joinpath(OUT_DIR, "subdecade_validation.txt"), "w") do io
    println(io, "Item 6 (alt): sub-decade validation");
    println(io, "Splits the 2014-2024 SPY window into two non-overlapping 5y sub-windows");
    println(io, "to test the CHMM scaffold's transfer across non-2014-2024-full IS slices.");
    println(io, "$N_PATHS paths, seed = $SEED.");
    println(io, "");
    println(io, "Note: pre-2014 cross-decade validation (1994-2004) requires external");
    println(io, "data ingest (Yahoo Finance / Stooq / similar) outside the user-authorised");
    println(io, "scope. The sub-decade split below is the in-window alternative.");
    println(io, "="^104);
    @printf(io, "%-40s | %-22s | %5s | %5s | %5s | %5s | %5s | %5s | %5s | %5s\n",
        "Config", "Model", "T_tr", "T_oo", "IS%", "OoS%", "IS-K", "OoS-K", "IS-A", "OoS-A");
    println(io, "-"^130);
    for r in results
        @printf(io, "%-40s | %-22s | %5d | %5d | %5.1f | %5.1f | %5.2f | %5.2f | %.4f | %.4f\n",
            r.config, r.model, r.T_train, r.T_oos, r.is_ks, r.oos_ks,
            r.is_sim_kurt, r.oos_sim_kurt, r.is_acf, r.oos_acf);
    end
    println(io);
    println(io, "Reading: configs A and B fit on 5y windows that share no calendar overlap.");
    println(io, "Compare:");
    println(io, "  - IS KS pass rates across configs at fixed K");
    println(io, "  - OoS KS pass rates: config A's OoS is 2019-2024 (covers COVID + 2022 rate-hike),");
    println(io, "    config B's OoS is 2024-2026 (the body's headline OoS), config C is the body row.");
    println(io, "  - Stability of the kurtosis match across configs");
end

@printf("\n[done] %s/subdecade_validation.txt\n", OUT_DIR);

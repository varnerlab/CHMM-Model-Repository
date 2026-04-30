# ========================================================================================= #
# run_quantgan_tcn_lambertw.jl
#
# Closes peer-review item P2.9 / R3 RE1: faithful Wiese et al.\ (2020) QuantGAN with
# Lambert-W input pre-processing. The companion run_quantgan_tcn.jl re-built the
# architecture (5-conv-layer WGAN with weight clipping) at the body baseline's training
# protocol but \emph{without} Lambert-W; this runner adds Lambert-W input pre-processing
# (Goerg, 2011, 2015) on top of the same architecture. The Lambert-W heavy-tail
# transformation maps an approximately-Lambert-W*Gaussian variate to an approximately-
# Gaussian variate before training, which the original Wiese et al.\ authors credit
# with most of the tail-fidelity of QuantGAN; we restore that pipeline here.
#
# Lambert-W heavy-tail forward (heavy-tailed y --> Gaussian z):
#     z = sign(y) * sqrt(W(delta * y^2) / delta)
# inverse (Gaussian z --> heavy-tailed y):
#     y = z * exp(delta/2 * z^2)
# delta >= 0 controls tail heaviness; delta = 0 recovers the identity (Gaussian).
# We fit delta on R_is via simple kurtosis matching: bracketed root-find for the delta
# that makes kurt(z) = 3 within tolerance, which is the standard IGMM (iterated
# generalised method of moments) fit at order 4.
#
# Output:
#   results/quantgan_tcn_lambertw/quantgan_tcn_lambertw_metrics.txt
#   ../CHMM-paper/results/robustness/quantgan_tcn_lambertw.csv
# ========================================================================================= #

using Pkg; Pkg.activate(".");
include("Include.jl");

using Random, Statistics, Flux, Printf
const SEED = 20260422;
Random.seed!(SEED);

const TICKER       = "SPY";
const RISK_FREE    = 0.0;
const DT           = 1/252;
const N_PATHS      = 500;
const L_LAGS       = 252;
const WINDOW_LEN   = 64;
const LATENT_DIM   = 8;
const TCN_CHANNELS = 48;
const TCN_BLOCKS   = 5;
const KERNEL_SIZE  = 3;
const BATCH_SIZE   = 64;
const EPOCHS       = 20;
const STEPS_PER_EPOCH = 80;
const N_CRITIC     = 4;
const LR_G         = 1e-4;
const LR_D         = 1e-4;
const CLIP_VALUE   = 0.01f0;

const OUT_DIR              = joinpath(_ROOT, "results", "quantgan_tcn_lambertw");
const PAPER_ROBUSTNESS_DIR = abspath(joinpath(_ROOT, "..", "CHMM-paper", "results", "robustness"));
mkpath(OUT_DIR);
mkpath(PAPER_ROBUSTNESS_DIR);

println("="^80)
println("  QuantGAN TCN with Lambert-W input pre-processing  (P2.9 / R3 RE1)")
println("="^80)

# --------------------------------------------------------------------------------------- #
# Lambert W function on the principal branch W_0(x) for x >= 0 via Halley's method.
# --------------------------------------------------------------------------------------- #
function lambert_w0(x::Float64; tol::Float64 = 1e-12, max_iter::Int = 30)
    x < 0.0 && error("lambert_w0: argument must be non-negative");
    x == 0.0 && return 0.0;
    # Initial guess: Corless et al.\ (1996) asymptotic
    w = x < 1.0 ? x / (1 + x) : log(x) - log(log(x + 1) + 1e-15);
    for _ in 1:max_iter
        e = exp(w);
        we = w * e;
        diff = we - x;
        if abs(diff) < tol; break; end
        w -= diff / (e * (w + 1) - (w + 2) * diff / (2 * (w + 1)));
    end
    return w;
end
lambert_w0(x::Float32; kwargs...) = Float32(lambert_w0(Float64(x); kwargs...));

# --------------------------------------------------------------------------------------- #
# Lambert-W heavy-tail transform (Goerg, 2011/2015).
# Forward: heavy-tailed y -> Gaussian z = sign(y) * sqrt(W(δ y^2) / δ).
# Inverse: Gaussian z      -> heavy-tailed y = z * exp(δ z^2 / 2).
# --------------------------------------------------------------------------------------- #
function lambertw_forward(y::Real, δ::Real)
    δ <= 0.0 && return Float64(y);
    z2 = lambert_w0(δ * Float64(y)^2) / δ;
    z2 = max(z2, 0.0);
    return sign(y) * sqrt(z2);
end

function lambertw_inverse(z::Real, δ::Real)
    δ <= 0.0 && return Float64(z);
    return Float64(z) * exp(δ * Float64(z)^2 / 2.0);
end

# IGMM-order-4 fit: find δ in [0, δ_max] such that kurtosis of forward-transformed series
# equals the Gaussian target of 3 (excess 0). Bracket-bisection because the relationship
# is monotone decreasing in δ.
function fit_lambertw_delta(y::AbstractVector; δ_lo = 0.0, δ_hi = 0.5, tol = 1e-4,
                            max_iter = 60)
    function transformed_kurtosis(δ)
        z = [lambertw_forward(yt, δ) for yt in y];
        zs = (z .- mean(z)) ./ std(z);
        return sum(zs .^ 4) / length(zs);  # raw kurtosis (Gaussian = 3)
    end
    target = 3.0;
    a = δ_lo; b = δ_hi;
    fa = transformed_kurtosis(a) - target;
    fb = transformed_kurtosis(b) - target;
    if fa * fb > 0
        # Both same sign: pick δ that minimises |k - 3| within bracket
        return abs(fa) < abs(fb) ? a : b;
    end
    for _ in 1:max_iter
        m = 0.5 * (a + b);
        fm = transformed_kurtosis(m) - target;
        if abs(fm) < tol; return m; end
        if fa * fm < 0
            b = m; fb = fm;
        else
            a = m; fa = fm;
        end
    end
    return 0.5 * (a + b);
end

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

obs_kurt_is = sum(((R_is .- mean(R_is)) ./ std(R_is)).^4) / n_is;  # raw (Gaussian = 3)
@printf("  observed raw kurtosis IS = %.3f (target Gaussian = 3.0)\n", obs_kurt_is);

# --------------------------------------------------------------------------------------- #
# Lambert-W pre-processing: fit delta on IS, transform IS to z space, train on z space
# --------------------------------------------------------------------------------------- #
δ_hat = fit_lambertw_delta(R_is; δ_lo = 0.0, δ_hi = 0.5);
@printf("\n[lambertw] fitted δ̂ = %.4f via IGMM order-4 (bracket-bisection on kurt(z) = 3)\n", δ_hat);

# Forward-transform R_is → z_is (heavy-tailed → approximately Gaussian)
z_is = [lambertw_forward(yt, δ_hat) for yt in R_is];
μ_z = mean(z_is); σ_z = std(z_is);
Z_is_norm = Float32.((z_is .- μ_z) ./ σ_z);
post_kurt = sum(((z_is .- μ_z) ./ σ_z) .^ 4) / length(z_is);
@printf("  pre  Lambert-W: raw kurtosis = %.3f\n", obs_kurt_is);
@printf("  post Lambert-W: raw kurtosis = %.3f  (target 3.0)\n", post_kurt);

function rolling_windows(z::Vector{Float32}, w::Int)
    n = length(z); nw = n - w + 1;
    X = Array{Float32, 3}(undef, w, 1, nw);
    for i in 1:nw
        X[:, 1, i] = z[i:(i + w - 1)];
    end
    return X;
end

train_windows = rolling_windows(Z_is_norm, WINDOW_LEN);
n_train_windows = size(train_windows, 3);
println("  train windows: $n_train_windows");

# --------------------------------------------------------------------------------------- #
# Architecture: same 5-conv-layer WGAN as run_quantgan_tcn.jl
# --------------------------------------------------------------------------------------- #
function make_generator(latent_dim::Int, hidden::Int)
    return Flux.Chain(
        Flux.Conv((KERNEL_SIZE,), latent_dim => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((1,), hidden => 1; pad = Flux.SamePad()),
    );
end

function make_critic(hidden::Int)
    return Flux.Chain(
        Flux.Conv((KERNEL_SIZE,), 1 => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        Flux.Conv((KERNEL_SIZE,), hidden => hidden, x -> leakyrelu(x, 0.2f0); pad = Flux.SamePad()),
        x -> mean(x; dims = 1),
        x -> dropdims(x; dims = 1),
        Flux.Dense(hidden => 1),
    );
end

generator = make_generator(LATENT_DIM, TCN_CHANNELS);
critic    = make_critic(TCN_CHANNELS);
n_g = sum(length, Flux.params(generator));
n_d = sum(length, Flux.params(critic));
println("[model] generator parameters: $n_g, critic parameters: $n_d");

opt_g = Flux.setup(Flux.Adam(LR_G), generator);
opt_d = Flux.setup(Flux.Adam(LR_D), critic);

function clip_critic!(critic, clip_value::Float32)
    for p in Flux.params(critic)
        p .= clamp.(p, -clip_value, clip_value);
    end
end

critic_scores(critic, x::Array{Float32, 3}) = vec(critic(x));

sample_noise(batch::Int) = randn(Float32, WINDOW_LEN, LATENT_DIM, batch);

function sample_real_batch(windows::Array{Float32, 3}, batch::Int)
    idx = rand(1:size(windows, 3), batch);
    return windows[:, :, idx];
end

# --------------------------------------------------------------------------------------- #
# Train (WGAN with weight clipping)
# --------------------------------------------------------------------------------------- #
println("\n[train] WGAN TCN with Lambert-W pre-processing, weight clipping...");
g_loss_hist = Float64[];
d_loss_hist = Float64[];
for epoch in 1:EPOCHS
    d_epoch = 0.0; g_epoch = 0.0;
    for step in 1:STEPS_PER_EPOCH
        for c in 1:N_CRITIC
            real = sample_real_batch(train_windows, BATCH_SIZE);
            z    = sample_noise(BATCH_SIZE);
            dloss, grads = Flux.withgradient(critic) do crit
                fake = generator(z);
                s_real = critic_scores(crit, real);
                s_fake = critic_scores(crit, fake);
                mean(s_fake) - mean(s_real)
            end
            Flux.update!(opt_d, critic, grads[1]);
            clip_critic!(critic, CLIP_VALUE);
            d_epoch += dloss;
        end
        z = sample_noise(BATCH_SIZE);
        gloss, grads = Flux.withgradient(generator) do gen
            fake = gen(z);
            -mean(critic_scores(critic, fake))
        end
        Flux.update!(opt_g, generator, grads[1]);
        g_epoch += gloss;
    end
    push!(d_loss_hist, d_epoch / (STEPS_PER_EPOCH * N_CRITIC));
    push!(g_loss_hist, g_epoch / STEPS_PER_EPOCH);
    @printf("  epoch %2d / %d  D = %+.4f  G = %+.4f\n",
            epoch, EPOCHS, d_loss_hist[end], g_loss_hist[end]);
end

# --------------------------------------------------------------------------------------- #
# Synthesis: generate paths in z space, de-standardise, INVERSE Lambert-W to return scale
# --------------------------------------------------------------------------------------- #
function synthesise_paths(generator, T::Int, n_paths::Int; window::Int = WINDOW_LEN,
                          μ_z::Float64, σ_z::Float64, δ::Float64)
    paths = Array{Float64, 2}(undef, T, n_paths);
    n_windows_per_path = ceil(Int, T / window);
    for p in 1:n_paths
        z_noise = sample_noise(n_windows_per_path);
        x_z = vec(generator(z_noise));
        x_z = x_z[1:T];
        # de-standardise back to the natural z scale
        x_z_unstd = Float64.(x_z) .* σ_z .+ μ_z;
        # inverse Lambert-W: z (Gaussian) -> y (heavy-tailed)
        for t in 1:T
            paths[t, p] = lambertw_inverse(x_z_unstd[t], δ);
        end
    end
    return paths;
end

println("\n[synth] Synthesising $N_PATHS IS-length and OoS-length paths under Lambert-W inverse...");
paths_is  = synthesise_paths(generator, n_is,  N_PATHS;
                              μ_z = μ_z, σ_z = σ_z, δ = δ_hat);
paths_oos = synthesise_paths(generator, n_oos, N_PATHS;
                              μ_z = μ_z, σ_z = σ_z, δ = δ_hat);

# --------------------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------------------- #
using StatsBase, HypothesisTests
function eval_panel(observed::Vector{Float64}, sim_archive::Matrix{Float64}; L_val::Int = L_LAGS)
    np = size(sim_archive, 2);
    n_o = length(observed);
    μ_o = mean(observed); σ_o = std(observed);
    kurt_o = sum(((observed .- μ_o) ./ σ_o).^4) / n_o - 3.0;
    L_use = min(L_val, n_o - 1);
    acf_o_abs = autocor(abs.(observed), 1:L_use);
    acf_o_raw = autocor(observed, 1:L_use);
    ks_pass = 0; kurt_s = 0.0; acf_mae_abs = 0.0; acf_mae_raw = 0.0;
    for i in 1:np
        sim = sim_archive[:, i];
        # Skip pathological paths with NaN / Inf (Lambert-W inverse can blow up far in the tail)
        if any(!isfinite, sim); continue; end
        pval = pvalue(ApproximateTwoSampleKSTest(observed, sim));
        if pval > 0.05; ks_pass += 1; end
        μ_s = mean(sim); σ_s = std(sim);
        kurt_s += sum(((sim .- μ_s) ./ σ_s).^4) / length(sim) - 3.0;
        acf_sim_abs = autocor(abs.(sim), 1:L_use);
        acf_sim_raw = autocor(sim, 1:L_use);
        acf_mae_abs += mean(abs.(acf_o_abs .- acf_sim_abs));
        acf_mae_raw += mean(abs.(acf_o_raw .- acf_sim_raw));
    end
    n_finite = sum(all(isfinite, sim_archive[:, i]) for i in 1:np);
    return (
        ks_pass_pct = 100 * ks_pass / np,
        kurt_obs    = kurt_o,
        kurt_sim    = n_finite > 0 ? kurt_s / n_finite : NaN,
        acf_mae_abs = n_finite > 0 ? acf_mae_abs / n_finite : NaN,
        acf_mae_raw = n_finite > 0 ? acf_mae_raw / n_finite : NaN,
        n_finite    = n_finite,
    );
end

m_is  = eval_panel(R_is,  paths_is);
m_oos = eval_panel(R_oos, paths_oos);

println("\n[result] QuantGAN TCN + Lambert-W:");
@printf("  IS  : KS pass%% = %.1f, kurt sim = %.2f vs obs %.2f, |G| ACF-MAE = %.4f, finite paths = %d / %d\n",
        m_is.ks_pass_pct, m_is.kurt_sim, m_is.kurt_obs, m_is.acf_mae_abs, m_is.n_finite, N_PATHS);
@printf("  OoS : KS pass%% = %.1f, kurt sim = %.2f vs obs %.2f, |G| ACF-MAE = %.4f, finite paths = %d / %d\n",
        m_oos.ks_pass_pct, m_oos.kurt_sim, m_oos.kurt_obs, m_oos.acf_mae_abs, m_oos.n_finite, N_PATHS);

# --------------------------------------------------------------------------------------- #
out_path = joinpath(OUT_DIR, "quantgan_tcn_lambertw_metrics.txt");
open(out_path, "w") do io
    println(io, "QuantGAN TCN with Lambert-W input pre-processing on SPY (P2.9 / R3 RE1).");
    println(io, "Goerg (2011, 2015) Lambert-W * Gaussian heavy-tail transformation: forward pre-train,");
    println(io, "inverse post-synthesis.  delta fit on IS by IGMM-order-4 (kurt(z) = 3) bracket-bisection.");
    println(io);
    @printf(io, "  Fitted Lambert-W delta = %.4f\n", δ_hat);
    @printf(io, "  Pre-Lambert-W IS raw kurtosis = %.3f (target 3.0 for Gaussian)\n", obs_kurt_is);
    @printf(io, "  Post-Lambert-W IS raw kurtosis = %.3f\n", post_kurt);
    println(io);
    @printf(io, "  Architecture : 5-conv WGAN, hidden = %d, kernel = %d, window = %d, epochs = %d\n",
            TCN_CHANNELS, KERNEL_SIZE, WINDOW_LEN, EPOCHS);
    @printf(io, "  Generator parameters = %d, critic parameters = %d\n", n_g, n_d);
    println(io);
    @printf(io, "  IS  : KS pass%% = %.1f, kurt sim = %.2f vs obs %.2f, |G| ACF-MAE = %.4f, finite paths = %d / %d\n",
            m_is.ks_pass_pct, m_is.kurt_sim, m_is.kurt_obs, m_is.acf_mae_abs, m_is.n_finite, N_PATHS);
    @printf(io, "  OoS : KS pass%% = %.1f, kurt sim = %.2f vs obs %.2f, |G| ACF-MAE = %.4f, finite paths = %d / %d\n",
            m_oos.ks_pass_pct, m_oos.kurt_sim, m_oos.kurt_obs, m_oos.acf_mae_abs, m_oos.n_finite, N_PATHS);
    println(io);
    println(io, "Comparison reference (run_quantgan_tcn.jl, same architecture, no Lambert-W):");
    println(io, "  see results/quantgan_tcn/quantgan_tcn_metrics.txt");
end

csv_path = joinpath(PAPER_ROBUSTNESS_DIR, "quantgan_tcn_lambertw.csv");
open(csv_path, "w") do io
    println(io, "scope,delta,ks_pass_pct,kurt_obs,kurt_sim,acf_mae_abs,acf_mae_raw,n_finite_paths");
    @printf(io, "IS,%.5f,%.2f,%.3f,%.3f,%.5f,%.5f,%d\n",
            δ_hat, m_is.ks_pass_pct, m_is.kurt_obs, m_is.kurt_sim,
            m_is.acf_mae_abs, m_is.acf_mae_raw, m_is.n_finite);
    @printf(io, "OoS,%.5f,%.2f,%.3f,%.3f,%.5f,%.5f,%d\n",
            δ_hat, m_oos.ks_pass_pct, m_oos.kurt_obs, m_oos.kurt_sim,
            m_oos.acf_mae_abs, m_oos.acf_mae_raw, m_oos.n_finite);
end

println("\n" * "="^80);
println("  QuantGAN TCN + Lambert-W complete.");
@printf("  Human-readable: %s\n", out_path);
@printf("  Paper CSV     : %s\n", csv_path);
println("="^80);

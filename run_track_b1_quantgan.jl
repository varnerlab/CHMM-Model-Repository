# ========================================================================================= #
# run_track_b1_quantgan.jl
#
# Track B1 (first pass): QuantGAN-style deep baseline in pure Julia / Flux.
#
# This is a practical repo-native approximation to QuantGAN:
#   - train on standardised rolling windows of SPY returns,
#   - convolutional generator and critic,
#   - Wasserstein loss with critic weight clipping,
#   - synthesize long paths by stitching generated windows.
#
# The goal is not exact architectural reproduction of the original paper; the
# goal is a serious GAN baseline row that is reproducible in this codebase.
#
# Outputs:
#   results/track_b1/Table-4-Extended-Metrics-B1.txt
#   results/track_b1/VaR_LR_tests_b1.txt
#   results/track_b1/sim_pvalues_b1.txt
#   results/track_b1/Track-B1-summary.txt
#   results/track_b1/Loss.svg
# ========================================================================================= #

using Pkg; Pkg.activate(".");
ENV["GKSwstype"] = "100";
include("Include.jl");

using Random
const SEED = 20260422;
Random.seed!(SEED);

const TICKER       = "SPY";
const RISK_FREE    = 0.0;
const DT           = 1/252;
const N_PATHS      = 1000;
const L_LAGS       = 252;
const WINDOW_LEN   = 64;
const LATENT_DIM   = 8;
const G_CHANNELS   = 32;
const D_CHANNELS   = 32;
const BATCH_SIZE   = 64;
const EPOCHS       = 15;
const STEPS_PER_EPOCH = 120;
const N_CRITIC     = 4;
const LR_G         = 1e-4;
const LR_D         = 1e-4;
const CLIP_VALUE   = 0.01f0;
const N_WINDOWS_METRICS = 500;
const SIG_DEPTH    = 3;
const MAX_LAG_LEV  = 20;
const HORIZONS_AG  = [1, 5, 10, 21];

const TRACK_A_DIR = joinpath(_ROOT, "results", "track_a");
const TRACK_B_DIR = joinpath(_ROOT, "results", "track_b1");
mkpath(TRACK_B_DIR);

println("="^72)
println("  Track B1: QuantGAN-style convolutional WGAN baseline")
println("  Seed=$SEED, W=$WINDOW_LEN, latent=$LATENT_DIM, epochs=$EPOCHS")
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
n_is = length(R_is);

oos_dataset = MyOutOfSamplePortfolioDataSet() |> x -> x["dataset"];
R_oos = Vector{Float64}(log_growth_matrix(oos_dataset, "SPY"; Δt=DT, risk_free_rate=RISK_FREE));
n_oos = length(R_oos);
println("  IS $n_is, OoS $n_oos");

μ_is = mean(R_is);
σ_is = std(R_is);
Z_is = Float32.((R_is .- μ_is) ./ σ_is);

function rolling_windows(z::Vector{Float32}, w::Int)
    n = length(z);
    nw = n - w + 1;
    X = Array{Float32, 3}(undef, w, 1, nw);
    for i in 1:nw
        X[:, 1, i] = z[i:(i + w - 1)];
    end
    return X;
end

train_windows = rolling_windows(Z_is, WINDOW_LEN);
n_train_windows = size(train_windows, 3);
println("  train windows: $n_train_windows");

function make_generator(latent_dim::Int, channels::Int)
    return Flux.Chain(
        Flux.Conv((5,), latent_dim => channels, relu; pad=Flux.SamePad()),
        Flux.Conv((5,), channels => channels, relu; pad=Flux.SamePad()),
        Flux.Conv((5,), channels => channels, relu; pad=Flux.SamePad()),
        Flux.Conv((1,), channels => 1; pad=Flux.SamePad())
    );
end

function make_critic(channels::Int)
    return Flux.Chain(
        Flux.Conv((5,), 1 => channels, x -> leakyrelu(x, 0.2f0); pad=Flux.SamePad()),
        Flux.Conv((5,), channels => channels, x -> leakyrelu(x, 0.2f0); pad=Flux.SamePad()),
        Flux.Conv((5,), channels => channels, x -> leakyrelu(x, 0.2f0); pad=Flux.SamePad()),
        x -> mean(x; dims=1),
        x -> dropdims(x; dims=1),
        Flux.Dense(channels => 1)
    );
end

function critic_scores(critic, x::Array{Float32, 3})
    return vec(critic(x));
end

function sample_noise(batch::Int)
    return randn(Float32, WINDOW_LEN, LATENT_DIM, batch);
end

function sample_real_batch(windows::Array{Float32, 3}, batch::Int)
    idx = rand(1:size(windows, 3), batch);
    return windows[:, :, idx];
end

generator = make_generator(LATENT_DIM, G_CHANNELS);
critic = make_critic(D_CHANNELS);

opt_g = Flux.setup(Flux.Adam(LR_G), generator);
opt_d = Flux.setup(Flux.Adam(LR_D), critic);

g_loss_hist = Float64[];
d_loss_hist = Float64[];

function clip_critic!(critic, clip_value::Float32)
    for layer in critic.layers
        if hasproperty(layer, :weight)
            layer.weight .= clamp.(layer.weight, -clip_value, clip_value);
        end
        if hasproperty(layer, :bias) && !isnothing(layer.bias)
            layer.bias .= clamp.(layer.bias, -clip_value, clip_value);
        end
    end
end

println("\n[train] QuantGAN-style WGAN...");
for epoch in 1:EPOCHS
    d_epoch = 0.0;
    g_epoch = 0.0;
    for _ in 1:STEPS_PER_EPOCH
        for _ in 1:N_CRITIC
            real = sample_real_batch(train_windows, BATCH_SIZE);
            z = sample_noise(BATCH_SIZE);
            dloss, grads = Flux.withgradient(critic) do c
                fake = generator(z);
                s_real = critic_scores(c, real);
                s_fake = critic_scores(c, fake);
                mean(s_fake) - mean(s_real)
            end
            Flux.update!(opt_d, critic, grads[1]);
            clip_critic!(critic, CLIP_VALUE);
            d_epoch += Float64(dloss);
        end

        z = sample_noise(BATCH_SIZE);
        gloss, grads = Flux.withgradient(generator) do g
            fake = g(z);
            -mean(critic_scores(critic, fake))
        end
        Flux.update!(opt_g, generator, grads[1]);
        g_epoch += Float64(gloss);
    end
    push!(d_loss_hist, d_epoch / (STEPS_PER_EPOCH * N_CRITIC));
    push!(g_loss_hist, g_epoch / STEPS_PER_EPOCH);
    println("  epoch $epoch / $EPOCHS  D=$(round(d_loss_hist[end], digits=4))  G=$(round(g_loss_hist[end], digits=4))");
end

loss_fig = plot(1:EPOCHS, d_loss_hist, lw=2, color=:darkred, label="Critic",
    xlabel="Epoch", ylabel="Loss", title="QuantGAN-style WGAN training loss",
    titlefontsize=10, size=(760, 420));
plot!(loss_fig, 1:EPOCHS, g_loss_hist, lw=2, color=:navy, label="Generator");
savefig(loss_fig, joinpath(TRACK_B_DIR, "Loss.svg"));

function generate_window(g)
    z = sample_noise(1);
    x = g(z);
    return Vector{Float64}(μ_is .+ σ_is .* vec(x[:, 1, 1]));
end

function generate_path(g, n_steps::Int)
    n_blocks = cld(n_steps, WINDOW_LEN);
    out = Vector{Float64}(undef, n_blocks * WINDOW_LEN);
    cursor = 1;
    for _ in 1:n_blocks
        win = generate_window(g);
        out[cursor:(cursor + WINDOW_LEN - 1)] = win;
        cursor += WINDOW_LEN;
    end
    return out[1:n_steps];
end

println("\n[sim] Generating $N_PATHS IS + OoS paths...");
qgan_is = Array{Float64, 2}(undef, n_is, N_PATHS);
qgan_oos = Array{Float64, 2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    Random.seed!(SEED + i);
    qgan_is[:, i] = generate_path(generator, n_is);
    Random.seed!(SEED + 1_000_000 + i);
    qgan_oos[:, i] = generate_path(generator, n_oos);
    if i % 100 == 0
        println("  path $i / $N_PATHS");
    end
end

cache_path = joinpath(TRACK_A_DIR, "sim_archive_cache.jld2");
base_archive = load(cache_path)["archive"];
archive = merge(base_archive, Dict("QuantGAN" => (is=qgan_is, oos=qgan_oos)));

MODEL_ORDER = [
    "GARCH", "QuantGAN", "CHMM-N", "CHMM-t", "CHMM-L"
];

println("\n[A1,A2,A3] MMD / sig-MMD / discriminator AUC...");
Random.seed!(SEED + 400);
obs_windows_is = let
    W_all = windowize(R_is, 20; stride=max(1, (n_is - 20) ÷ N_WINDOWS_METRICS));
    idx = randperm(size(W_all, 2))[1:min(N_WINDOWS_METRICS, size(W_all, 2))];
    W_all[:, idx];
end;
obs_windows_oos = let
    W_all = windowize(R_oos, 20; stride=max(1, (n_oos - 20) ÷ N_WINDOWS_METRICS));
    idx = randperm(size(W_all, 2))[1:min(N_WINDOWS_METRICS, size(W_all, 2))];
    W_all[:, idx];
end;

mmd_results = Dict{String, NamedTuple}();
for m in MODEL_ORDER
    rng = MersenneTwister(SEED + hash(m) % 10^6);
    s_is, s_oos = archive[m].is, archive[m].oos;
    syn_windows_is  = sample_windows_from_archive(s_is, 20, size(obs_windows_is, 2); rng=rng);
    syn_windows_oos = sample_windows_from_archive(s_oos, 20, size(obs_windows_oos, 2); rng=rng);
    mmd_results[m] = (
        mmd_is=mmd2_rbf(obs_windows_is, syn_windows_is; rng=rng),
        mmd_oos=mmd2_rbf(obs_windows_oos, syn_windows_oos; rng=rng),
        sig_is=sig_mmd2(obs_windows_is, syn_windows_is; depth=SIG_DEPTH, rng=rng),
        sig_oos=sig_mmd2(obs_windows_oos, syn_windows_oos; depth=SIG_DEPTH, rng=rng),
        auc_is=discriminator_auc(R_is, s_is; window=20, n_windows=N_WINDOWS_METRICS, rng=rng).auc,
        auc_oos=discriminator_auc(R_oos, s_oos; window=20, n_windows=N_WINDOWS_METRICS, rng=rng).auc,
    );
    println("  $m  MMD=$(round(mmd_results[m].mmd_is, digits=5))  sig=$(round(mmd_results[m].sig_is, digits=5))  AUC=$(round(mmd_results[m].auc_is, digits=3))");
end

println("\n[A9] p-value coverage...");
function _kurt(x)
    μ = mean(x); σ = std(x);
    return σ > 0 ? sum(((x .- μ) ./ σ) .^ 4) / length(x) - 3.0 : 0.0;
end

obs_kurt_is = _kurt(R_is); obs_kurt_oos = _kurt(R_oos);
obs_ak_is = aggregational_kurtosis(R_is; horizons=HORIZONS_AG);
obs_ak_oos = aggregational_kurtosis(R_oos; horizons=HORIZONS_AG);
obs_lev_is = leverage_effect(R_is; max_lag=MAX_LAG_LEV);
obs_lev_oos = leverage_effect(R_oos; max_lag=MAX_LAG_LEV);

pv_results = Dict{String, NamedTuple}();
for m in MODEL_ORDER
    s_is, s_oos = archive[m].is, archive[m].oos
    kurt_is_s = [_kurt(s_is[:, i]) for i in 1:size(s_is, 2)];
    kurt_oos_s = [_kurt(s_oos[:, i]) for i in 1:size(s_oos, 2)];
    lev_is_s = [leverage_effect(s_is[:, i]; max_lag=MAX_LAG_LEV).avg_neg for i in 1:size(s_is, 2)];
    lev_oos_s = [leverage_effect(s_oos[:, i]; max_lag=MAX_LAG_LEV).avg_neg for i in 1:size(s_oos, 2)];
    ak5_is_s = [aggregational_kurtosis(s_is[:, i]; horizons=HORIZONS_AG)[5] for i in 1:size(s_is, 2)];
    ak21_is_s = [aggregational_kurtosis(s_is[:, i]; horizons=HORIZONS_AG)[21] for i in 1:size(s_is, 2)];
    ak5_oos_s = [aggregational_kurtosis(s_oos[:, i]; horizons=HORIZONS_AG)[5] for i in 1:size(s_oos, 2)];
    ak21_oos_s = [aggregational_kurtosis(s_oos[:, i]; horizons=HORIZONS_AG)[21] for i in 1:size(s_oos, 2)];
    pv_results[m] = (
        pv_joint_is=mean([sim_pvalue(obs_kurt_is, kurt_is_s), sim_pvalue(obs_lev_is.avg_neg, lev_is_s),
                          sim_pvalue(obs_ak_is[5], ak5_is_s), sim_pvalue(obs_ak_is[21], ak21_is_s)]),
        pv_joint_oos=mean([sim_pvalue(obs_kurt_oos, kurt_oos_s), sim_pvalue(obs_lev_oos.avg_neg, lev_oos_s),
                           sim_pvalue(obs_ak_oos[5], ak5_oos_s), sim_pvalue(obs_ak_oos[21], ak21_oos_s)]),
    );
end

println("\n[A8] Unconditional VaR LR tests...");
var_results = Dict{String, NamedTuple}();
for m in MODEL_ORDER
    v01 = quantile(vec(archive[m].is), 0.01);
    v05 = quantile(vec(archive[m].is), 0.05);
    br01 = R_oos .<= v01; br05 = R_oos .<= v05;
    var_results[m] = (
        v01=v01, v05=v05,
        k01=kupiec_lr(br01, 0.01), k05=kupiec_lr(br05, 0.05),
        c01=christoffersen_lr(br01), c05=christoffersen_lr(br05)
    );
end

open(joinpath(TRACK_B_DIR, "Table-4-Extended-Metrics-B1.txt"), "w") do io
    println(io, "="^120);
    println(io, "TABLE 4 (Track B1). QuantGAN-style convolutional WGAN baseline added.");
    println(io, "="^120);
    println(io, "Setup: SPY daily log excess growth; IS n=$n_is, OoS n=$n_oos; seed=$SEED; N_paths=$N_PATHS.");
    println(io, "B1 model: 3-layer 1D convolutional generator + critic, Wasserstein loss with critic weight clipping on rolling windows (W=$WINDOW_LEN).");
    println(io, "Generator training loss final: $(round(g_loss_hist[end], digits=4)); critic: $(round(d_loss_hist[end], digits=4)).");
    println(io, "");
    println(io, rpad("Model", 10), " | ", rpad("MMD IS", 10), " | ", rpad("MMD OoS", 10), " | ",
                rpad("sig IS", 10), " | ", rpad("sig OoS", 10), " | ", rpad("AUC IS", 7), " | ", rpad("AUC OoS", 7));
    println(io, "-"^120);
    for m in MODEL_ORDER
        r = mmd_results[m];
        println(io, rpad(m, 10), " | ", rpad(round(r.mmd_is, digits=5), 10), " | ",
                    rpad(round(r.mmd_oos, digits=5), 10), " | ",
                    rpad(round(r.sig_is, digits=5), 10), " | ",
                    rpad(round(r.sig_oos, digits=5), 10), " | ",
                    rpad(round(r.auc_is, digits=3), 7), " | ",
                    rpad(round(r.auc_oos, digits=3), 7));
    end
end

open(joinpath(TRACK_B_DIR, "sim_pvalues_b1.txt"), "w") do io
    println(io, "Track B1. Joint p-value coverage pv̄ with QuantGAN row.");
    println(io, rpad("Model", 10), " | pv̄ IS   | pv̄ OoS");
    for m in MODEL_ORDER
        pv = pv_results[m];
        println(io, rpad(m, 10), " | ", rpad(round(pv.pv_joint_is, digits=3), 7), " | ", rpad(round(pv.pv_joint_oos, digits=3), 7));
    end
end

open(joinpath(TRACK_B_DIR, "VaR_LR_tests_b1.txt"), "w") do io
    println(io, "Track B1. Unconditional VaR LR tests with QuantGAN row (α ∈ {0.01, 0.05}).");
    println(io, rpad("Model", 10), " | VaR01   | br%01 | LR_uc01 | LR_ind01 | VaR05   | br%05 | LR_uc05 | LR_ind05");
    for m in MODEL_ORDER
        r = var_results[m];
        println(io, rpad(m, 10), " | ", rpad(round(r.v01, digits=3), 7), " | ",
                    rpad(round(100*r.k01.breach_rate, digits=1), 5), " | ",
                    rpad(round(r.k01.LR, digits=2), 7), " | ",
                    rpad(round(r.c01.LR, digits=2), 8), " | ",
                    rpad(round(r.v05, digits=3), 7), " | ",
                    rpad(round(100*r.k05.breach_rate, digits=1), 5), " | ",
                    rpad(round(r.k05.LR, digits=2), 7), " | ",
                    rpad(round(r.c05.LR, digits=2), 8));
    end
end

open(joinpath(TRACK_B_DIR, "Track-B1-summary.txt"), "w") do io
    println(io, "Track B1 summary: QuantGAN-style convolutional WGAN baseline");
    println(io, "="^80);
    println(io, "");
    println(io, "Training setup:");
    println(io, "  rolling-window WGAN with critic weight clipping on standardised SPY returns, W=$WINDOW_LEN, latent=$LATENT_DIM, epochs=$EPOCHS.");
    println(io, "  final generator loss $(round(g_loss_hist[end], digits=4)), critic loss $(round(d_loss_hist[end], digits=4)).");
    println(io, "");
    for m in ("GARCH", "QuantGAN", "CHMM-t")
        r = mmd_results[m];
        println(io, "  $(rpad(m, 8)) MMD IS $(round(r.mmd_is, digits=5)) sig-MMD IS $(round(r.sig_is, digits=5)) AUC IS $(round(r.auc_is, digits=3)) pv̄ OoS $(round(pv_results[m].pv_joint_oos, digits=3))");
    end
    println(io, "");
    println(io, "Interpretation: QuantGAN is the deep generative row to compare against CHMM-t on");
    println(io, "distributional fidelity. The stitched-window simulation scheme is simple and reproducible,");
    println(io, "but it may understate long-horizon dependence relative to an autoregressive generator.");
end

println("\n" * "="^72);
println("  Track B1 complete.");
println("  Results: $TRACK_B_DIR");
println("="^72);

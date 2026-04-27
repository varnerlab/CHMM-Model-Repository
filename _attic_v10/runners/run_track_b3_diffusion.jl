# ========================================================================================= #
# run_track_b3_diffusion.jl
#
# Track B3 (first pass): window-based diffusion baseline in pure Julia / Flux.
#
# This is a practical TimeGrad-style approximation for this repo:
#   - train on standardised rolling windows of SPY returns,
#   - epsilon-prediction denoiser on a fixed diffusion schedule,
#   - generate windows by reverse diffusion,
#   - stitch windows into full paths for the existing metrics harness.
#
# Outputs:
#   results/track_b3/Table-4-Extended-Metrics-B3.txt
#   results/track_b3/VaR_LR_tests_b3.txt
#   results/track_b3/sim_pvalues_b3.txt
#   results/track_b3/Track-B3-summary.txt
#   results/track_b3/Loss.svg
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
const WINDOW_LEN   = 64;
const CHANNELS     = 32;
const DIFF_STEPS   = 50;
const BATCH_SIZE   = 64;
const EPOCHS       = 18;
const STEPS_PER_EPOCH = 140;
const LR           = 2e-4;
const N_WINDOWS_METRICS = 500;
const SIG_DEPTH    = 3;
const MAX_LAG_LEV  = 20;
const HORIZONS_AG  = [1, 5, 10, 21];

const TRACK_A_DIR = joinpath(_ROOT, "results", "track_a");
const TRACK_B_DIR = joinpath(_ROOT, "results", "track_b3");
mkpath(TRACK_B_DIR);

println("="^72)
println("  Track B3: window-based diffusion baseline")
println("  Seed=$SEED, W=$WINDOW_LEN, steps=$DIFF_STEPS, epochs=$EPOCHS")
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
println("  train windows: ", size(train_windows, 3));

β = Float32.(collect(range(1e-4, 0.02; length=DIFF_STEPS)));
α = 1 .- β;
ᾱ = accumulate(*, α);

function make_denoiser(channels::Int)
    return Flux.Chain(
        Flux.Conv((5,), 2 => channels, relu; pad=Flux.SamePad()),
        Flux.Conv((5,), channels => channels, relu; pad=Flux.SamePad()),
        Flux.Conv((5,), channels => channels, relu; pad=Flux.SamePad()),
        Flux.Conv((1,), channels => 1; pad=Flux.SamePad())
    );
end

function sample_real_batch(windows::Array{Float32, 3}, batch::Int)
    idx = rand(1:size(windows, 3), batch);
    return windows[:, :, idx];
end

function timestep_channel(t::Int, batch::Int)
    τ = Float32(t / DIFF_STEPS);
    return fill(τ, WINDOW_LEN, 1, batch);
end

function denoiser_input(x::Array{Float32, 3}, t::Int)
    return cat(x, timestep_channel(t, size(x, 3)); dims=2);
end

denoiser = make_denoiser(CHANNELS);
opt = Flux.setup(Flux.Adam(LR), denoiser);
loss_hist = Float64[];

println("\n[train] DDPM-style epsilon prediction...");
for epoch in 1:EPOCHS
    epoch_loss = 0.0;
    for _ in 1:STEPS_PER_EPOCH
        x0 = sample_real_batch(train_windows, BATCH_SIZE);
        t = rand(1:DIFF_STEPS);
        eps = randn(Float32, size(x0));
        x_t = sqrt(ᾱ[t]) .* x0 .+ sqrt(1 - ᾱ[t]) .* eps;

        loss, grads = Flux.withgradient(denoiser) do d
            epŝ = d(denoiser_input(x_t, t));
            mean((epŝ .- eps) .^ 2)
        end
        Flux.update!(opt, denoiser, grads[1]);
        epoch_loss += Float64(loss);
    end
    push!(loss_hist, epoch_loss / STEPS_PER_EPOCH);
    println("  epoch $epoch / $EPOCHS  loss=$(round(loss_hist[end], digits=5))");
end

loss_fig = plot(1:EPOCHS, loss_hist, lw=2, marker=:circle, ms=3, color=:darkgreen,
    xlabel="Epoch", ylabel="MSE", title="Window diffusion training loss", titlefontsize=10, size=(760, 420));
savefig(loss_fig, joinpath(TRACK_B_DIR, "Loss.svg"));

function sample_window(d)
    x = randn(Float32, WINDOW_LEN, 1, 1);
    for t in DIFF_STEPS:-1:1
        epŝ = d(denoiser_input(x, t));
        z = t > 1 ? randn(Float32, size(x)) : zeros(Float32, size(x));
        x = (1 / sqrt(α[t])) .* (x .- ((1 - α[t]) / sqrt(1 - ᾱ[t])) .* epŝ) .+ sqrt(β[t]) .* z;
    end
    return Vector{Float64}(μ_is .+ σ_is .* vec(x[:, 1, 1]));
end

function sample_path(d, n_steps::Int)
    n_blocks = cld(n_steps, WINDOW_LEN);
    out = Vector{Float64}(undef, n_blocks * WINDOW_LEN);
    cursor = 1;
    for _ in 1:n_blocks
        win = sample_window(d);
        out[cursor:(cursor + WINDOW_LEN - 1)] = win;
        cursor += WINDOW_LEN;
    end
    return out[1:n_steps];
end

println("\n[sim] Generating $N_PATHS IS + OoS paths...");
diff_is = Array{Float64, 2}(undef, n_is, N_PATHS);
diff_oos = Array{Float64, 2}(undef, n_oos, N_PATHS);
for i in 1:N_PATHS
    Random.seed!(SEED + i);
    diff_is[:, i] = sample_path(denoiser, n_is);
    Random.seed!(SEED + 1_000_000 + i);
    diff_oos[:, i] = sample_path(denoiser, n_oos);
    if i % 100 == 0
        println("  path $i / $N_PATHS");
    end
end

cache_path = joinpath(TRACK_A_DIR, "sim_archive_cache.jld2");
base_archive = load(cache_path)["archive"];
archive = merge(base_archive, Dict("Diffusion" => (is=diff_is, oos=diff_oos)));

MODEL_ORDER = ["GARCH", "QuantGAN", "Diffusion", "CHMM-N", "CHMM-t", "CHMM-L"];

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
    if !haskey(archive, m)
        continue;
    end
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
    if !haskey(archive, m)
        continue;
    end
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
    if !haskey(archive, m)
        continue;
    end
    v01 = quantile(vec(archive[m].is), 0.01);
    v05 = quantile(vec(archive[m].is), 0.05);
    br01 = R_oos .<= v01; br05 = R_oos .<= v05;
    var_results[m] = (
        v01=v01, v05=v05,
        k01=kupiec_lr(br01, 0.01), k05=kupiec_lr(br05, 0.05),
        c01=christoffersen_lr(br01), c05=christoffersen_lr(br05)
    );
end

open(joinpath(TRACK_B_DIR, "Table-4-Extended-Metrics-B3.txt"), "w") do io
    println(io, "="^120);
    println(io, "TABLE 4 (Track B3). Window diffusion baseline added.");
    println(io, "="^120);
    println(io, "Setup: SPY daily log excess growth; IS n=$n_is, OoS n=$n_oos; seed=$SEED; N_paths=$N_PATHS.");
    println(io, "B3 model: DDPM-style 1D convolutional denoiser on rolling windows (W=$WINDOW_LEN, T=$DIFF_STEPS).");
    println(io, "Final denoising loss: $(round(loss_hist[end], digits=5)).");
    println(io, "");
    println(io, rpad("Model", 10), " | ", rpad("MMD IS", 10), " | ", rpad("MMD OoS", 10), " | ",
                rpad("sig IS", 10), " | ", rpad("sig OoS", 10), " | ", rpad("AUC IS", 7), " | ", rpad("AUC OoS", 7));
    println(io, "-"^120);
    for m in MODEL_ORDER
        if !haskey(mmd_results, m)
            continue;
        end
        r = mmd_results[m];
        println(io, rpad(m, 10), " | ", rpad(round(r.mmd_is, digits=5), 10), " | ",
                    rpad(round(r.mmd_oos, digits=5), 10), " | ",
                    rpad(round(r.sig_is, digits=5), 10), " | ",
                    rpad(round(r.sig_oos, digits=5), 10), " | ",
                    rpad(round(r.auc_is, digits=3), 7), " | ",
                    rpad(round(r.auc_oos, digits=3), 7));
    end
end

open(joinpath(TRACK_B_DIR, "sim_pvalues_b3.txt"), "w") do io
    println(io, "Track B3. Joint p-value coverage pv̄ with diffusion row.");
    println(io, rpad("Model", 10), " | pv̄ IS   | pv̄ OoS");
    for m in MODEL_ORDER
        if !haskey(pv_results, m)
            continue;
        end
        pv = pv_results[m];
        println(io, rpad(m, 10), " | ", rpad(round(pv.pv_joint_is, digits=3), 7), " | ", rpad(round(pv.pv_joint_oos, digits=3), 7));
    end
end

open(joinpath(TRACK_B_DIR, "VaR_LR_tests_b3.txt"), "w") do io
    println(io, "Track B3. Unconditional VaR LR tests with diffusion row (α ∈ {0.01, 0.05}).");
    println(io, rpad("Model", 10), " | VaR01   | br%01 | LR_uc01 | LR_ind01 | VaR05   | br%05 | LR_uc05 | LR_ind05");
    for m in MODEL_ORDER
        if !haskey(var_results, m)
            continue;
        end
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

open(joinpath(TRACK_B_DIR, "Track-B3-summary.txt"), "w") do io
    println(io, "Track B3 summary: window-based diffusion baseline");
    println(io, "="^80);
    println(io, "");
    println(io, "Training setup:");
    println(io, "  DDPM-style denoiser on standardised SPY return windows, W=$WINDOW_LEN, T=$DIFF_STEPS, epochs=$EPOCHS.");
    println(io, "  final denoising loss $(round(loss_hist[end], digits=5)).");
    println(io, "");
    for m in ("QuantGAN", "Diffusion", "CHMM-t")
        if !haskey(mmd_results, m)
            continue;
        end
        r = mmd_results[m];
        println(io, "  $(rpad(m, 9)) MMD IS $(round(r.mmd_is, digits=5)) sig-MMD IS $(round(r.sig_is, digits=5)) AUC IS $(round(r.auc_is, digits=3)) pv̄ OoS $(round(pv_results[m].pv_joint_oos, digits=3))");
    end
    println(io, "");
    println(io, "Interpretation: diffusion is the remaining serious deep baseline to compare against");
    println(io, "CHMM-t after the first-pass QuantGAN row. The stitched-window generation scheme is");
    println(io, "simple and reproducible but does not encode long-run path memory explicitly.");
end

println("\n" * "="^72);
println("  Track B3 complete.");
println("  Results: $TRACK_B_DIR");
println("="^72);

# ========================================================================================= #
# Metrics.jl
#
# Track A evaluation metrics for the synthetic-data-generator pipeline:
#   A1  mmd2_rbf                       Maximum Mean Discrepancy, RBF kernel, median heuristic
#   A2  path_signature, sig_mmd2       Truncated path signature (time-lifted), signature MMD
#   A3  discriminator_auc              Real-vs-synthetic MLP classifier, 5-fold CV AUC
#   A6  leverage_effect                corr(r_t^2, r_{t-k}) and gain/loss asymmetry
#   A7  aggregational_kurtosis         Kurtosis at multiple aggregation horizons
#   A8  kupiec_lr, christoffersen_lr   VaR unconditional + independence LR tests
#   A9  sim_pvalue                     Two-sided simulation-based p-value
#
# Pure helpers with no I/O. Consumers: run_track_a.jl and run_diagnostics.jl.
# ========================================================================================= #

using LinearAlgebra;
using Statistics;
using StatsBase;
using Random;
using Distributions;

# ------------------------------------------------------------------------------------------- #
# Window extraction
# ------------------------------------------------------------------------------------------- #

"""
    windowize(x::AbstractVector, W::Int; stride::Int=1)

Return a matrix of shape (W, n_windows) where column j is x[start_j : start_j + W - 1].
"""
function windowize(x::AbstractVector, W::Int; stride::Int=1)
    T = length(x);
    if T < W; return Array{Float64,2}(undef, W, 0); end
    starts = collect(1:stride:(T - W + 1));
    M = Array{Float64,2}(undef, W, length(starts));
    for (j, s) in enumerate(starts)
        M[:, j] = x[s:s + W - 1];
    end
    return M;
end

"""
    sample_windows_from_archive(sim_archive::AbstractMatrix, W::Int, n_windows::Int; stride::Int=1, rng=Random.GLOBAL_RNG)

Collect `n_windows` windows of length W from a (T x n_paths) matrix, random stride and random
path for each window.
"""
function sample_windows_from_archive(sim_archive::AbstractMatrix, W::Int, n_windows::Int;
                                     stride::Int=1, rng=Random.GLOBAL_RNG)
    T, np = size(sim_archive);
    if T < W; return Array{Float64,2}(undef, W, 0); end
    out = Array{Float64,2}(undef, W, n_windows);
    for j in 1:n_windows
        p = rand(rng, 1:np);
        s = rand(rng, 1:(T - W + 1));
        out[:, j] = sim_archive[s:s + W - 1, p];
    end
    return out;
end

# ------------------------------------------------------------------------------------------- #
# A1. MMD with RBF kernel, median-heuristic bandwidth
# ------------------------------------------------------------------------------------------- #

"""
    median_bandwidth(Z::AbstractMatrix)

Median-heuristic squared bandwidth for an RBF kernel on columns of Z.
Uses a random sub-sample of pairwise distances when the input is large.
"""
function median_bandwidth(Z::AbstractMatrix; max_pairs::Int=5000,
                          rng=Random.GLOBAL_RNG)
    n = size(Z, 2);
    if n <= 1; return 1.0; end
    n_pairs = min(max_pairs, n*(n-1) ÷ 2);
    ds2 = Vector{Float64}(undef, n_pairs);
    for k in 1:n_pairs
        i = rand(rng, 1:n); j = rand(rng, 1:n);
        while j == i; j = rand(rng, 1:n); end
        d = 0.0;
        for r in 1:size(Z, 1); d += (Z[r, i] - Z[r, j])^2; end
        ds2[k] = d;
    end
    m = median(ds2);
    return m > 0 ? m : 1.0;
end

function _rbf_gram(X::AbstractMatrix, Y::AbstractMatrix, γ::Float64)
    # X: d x nx, Y: d x ny. Returns K of size nx x ny with K[i,j] = exp(-||X_i - Y_j||^2 / γ).
    nx = size(X, 2); ny = size(Y, 2);
    xsq = vec(sum(X .^ 2, dims=1));
    ysq = vec(sum(Y .^ 2, dims=1));
    K = X' * Y;
    @inbounds for j in 1:ny, i in 1:nx
        d = xsq[i] + ysq[j] - 2.0 * K[i, j];
        K[i, j] = exp(-max(d, 0.0) / γ);
    end
    return K;
end

"""
    mmd2_rbf(X::AbstractMatrix, Y::AbstractMatrix; γ=nothing, unbiased::Bool=true)

Maximum Mean Discrepancy squared with RBF kernel. Columns of X and Y are samples (each column
is a d-vector). γ is the kernel bandwidth (scale = 2 σ²); if nothing, uses the median heuristic
on the pooled sample.
"""
function mmd2_rbf(X::AbstractMatrix, Y::AbstractMatrix; γ=nothing,
                  unbiased::Bool=true, rng=Random.GLOBAL_RNG)
    if γ === nothing
        Z = hcat(X, Y);
        γ = median_bandwidth(Z; rng=rng);
    end
    Kxx = _rbf_gram(X, X, γ);
    Kyy = _rbf_gram(Y, Y, γ);
    Kxy = _rbf_gram(X, Y, γ);
    m = size(X, 2); n = size(Y, 2);
    if unbiased && m > 1 && n > 1
        sxx = (sum(Kxx) - sum(diag(Kxx))) / (m * (m - 1));
        syy = (sum(Kyy) - sum(diag(Kyy))) / (n * (n - 1));
    else
        sxx = sum(Kxx) / (m * m);
        syy = sum(Kyy) / (n * n);
    end
    sxy = sum(Kxy) / (m * n);
    return max(sxx + syy - 2.0 * sxy, 0.0);
end

# ------------------------------------------------------------------------------------------- #
# A2. Truncated path signatures (time-lifted 2D path), signature MMD
# ------------------------------------------------------------------------------------------- #

"""
    path_signature(x::AbstractVector; depth::Int=3, normalize::Bool=true)

Truncated signature of the time-augmented 2D path X_t = (t/W, x_t). Returns a flat feature
vector containing signature components up to `depth`. Uses Chen's iterated-sum formula on
discrete increments.

Dimensionality: Σ_{k=1}^{depth} 2^k. At depth=3 this is 14 features.
"""
function path_signature(x::AbstractVector; depth::Int=3, normalize::Bool=true)
    @assert depth >= 1 && depth <= 5 "depth must be in 1..5";
    W = length(x);
    if W < 2; return zeros(sum(2^k for k in 1:depth)); end
    # Time-augmented 2D path with time in [0, 1]
    t_axis = collect(range(0.0, 1.0; length=W));
    x_axis = normalize ? (x .- first(x)) ./ max(std(x), 1e-12) : collect(x);
    # Increments: (W-1) x 2
    dt = diff(t_axis);
    dx = diff(x_axis);
    T = length(dt);
    d = 2;

    # Level 1
    s1 = Vector{Float64}(undef, d);
    s1[1] = sum(dt); s1[2] = sum(dx);
    feats = copy(s1);

    if depth >= 2
        # Level 2: s2[i,j] = sum_{t1 < t2} incr[t1,i] * incr[t2,j]
        #        = sum_t incr[t,j] * (prefix_i up to t-1)
        s2 = zeros(d, d);
        pref = zeros(d);
        for tt in 1:T
            inc = (dt[tt], dx[tt]);
            for i in 1:d, j in 1:d
                s2[i, j] += pref[i] * inc[j];
            end
            for i in 1:d; pref[i] += inc[i]; end
        end
        append!(feats, vec(s2));
    end

    if depth >= 3
        # Level 3: s3[i,j,k] accumulates incr[t,k] * Sig^(i,j)(0, t-1)
        s3 = zeros(d, d, d);
        level1 = zeros(d);
        level2 = zeros(d, d);
        for tt in 1:T
            inc = (dt[tt], dx[tt]);
            for i in 1:d, j in 1:d, k in 1:d
                s3[i, j, k] += level2[i, j] * inc[k];
            end
            for i in 1:d, j in 1:d
                level2[i, j] += level1[i] * inc[j];
            end
            for i in 1:d; level1[i] += inc[i]; end
        end
        append!(feats, vec(s3));
    end

    if depth >= 4
        s4 = zeros(d, d, d, d);
        l1 = zeros(d); l2 = zeros(d, d); l3 = zeros(d, d, d);
        for tt in 1:T
            inc = (dt[tt], dx[tt]);
            for i in 1:d, j in 1:d, k in 1:d, l in 1:d
                s4[i, j, k, l] += l3[i, j, k] * inc[l];
            end
            for i in 1:d, j in 1:d, k in 1:d
                l3[i, j, k] += l2[i, j] * inc[k];
            end
            for i in 1:d, j in 1:d
                l2[i, j] += l1[i] * inc[j];
            end
            for i in 1:d; l1[i] += inc[i]; end
        end
        append!(feats, vec(s4));
    end

    if depth >= 5
        s5 = zeros(d, d, d, d, d);
        l1 = zeros(d); l2 = zeros(d, d); l3 = zeros(d, d, d); l4 = zeros(d, d, d, d);
        for tt in 1:T
            inc = (dt[tt], dx[tt]);
            for i in 1:d, j in 1:d, k in 1:d, l in 1:d, m in 1:d
                s5[i, j, k, l, m] += l4[i, j, k, l] * inc[m];
            end
            for i in 1:d, j in 1:d, k in 1:d, l in 1:d
                l4[i, j, k, l] += l3[i, j, k] * inc[l];
            end
            for i in 1:d, j in 1:d, k in 1:d
                l3[i, j, k] += l2[i, j] * inc[k];
            end
            for i in 1:d, j in 1:d
                l2[i, j] += l1[i] * inc[j];
            end
            for i in 1:d; l1[i] += inc[i]; end
        end
        append!(feats, vec(s5));
    end

    return feats;
end

"""
    signature_features(windows::AbstractMatrix; depth::Int=3)

Apply `path_signature` to each column of `windows`. Returns a matrix of shape
(sig_dim x n_windows).
"""
function signature_features(windows::AbstractMatrix; depth::Int=3)
    n = size(windows, 2);
    first_sig = path_signature(windows[:, 1]; depth=depth);
    sig_dim = length(first_sig);
    F = Array{Float64,2}(undef, sig_dim, n);
    F[:, 1] = first_sig;
    for j in 2:n
        F[:, j] = path_signature(windows[:, j]; depth=depth);
    end
    return F;
end

"""
    sig_mmd2(X_windows::AbstractMatrix, Y_windows::AbstractMatrix; depth::Int=3)

Signature-MMD: compute truncated signatures on each window, then RBF-MMD on signature vectors.
"""
function sig_mmd2(X_windows::AbstractMatrix, Y_windows::AbstractMatrix;
                  depth::Int=3, rng=Random.GLOBAL_RNG)
    FX = signature_features(X_windows; depth=depth);
    FY = signature_features(Y_windows; depth=depth);
    return mmd2_rbf(FX, FY; rng=rng);
end

# ------------------------------------------------------------------------------------------- #
# A3. Discriminator AUC (real vs synthetic)
# ------------------------------------------------------------------------------------------- #

"""
    _feature_vector(w::AbstractVector)

Hand-crafted features for a single window: location, scale, shape, tail, ACF, vol clustering.
Used for the fast logistic-regression discriminator.
"""
function _feature_vector(w::AbstractVector)
    n = length(w);
    μ = mean(w); σ = std(w) + 1e-12;
    z = (w .- μ) ./ σ;
    skew_v = sum(z .^ 3) / n;
    kurt_v = sum(z .^ 4) / n - 3.0;
    mx = maximum(w); mn = minimum(w);
    cw = cumsum(w);
    max_dd = maximum(cw) - minimum(cw);
    abs_w = abs.(w);
    ac1 = n >= 2 ? cor(w[1:end-1], w[2:end]) : 0.0;
    aa1 = n >= 2 ? cor(abs_w[1:end-1], abs_w[2:end]) : 0.0;
    aa5 = n >= 6 ? cor(abs_w[1:end-5], abs_w[6:end]) : 0.0;
    return [μ, σ, skew_v, kurt_v, mx, mn, max_dd, ac1, aa1, aa5];
end

function _featurize(windows::AbstractMatrix)
    nw = size(windows, 2);
    first_feat = _feature_vector(windows[:, 1]);
    fd = length(first_feat);
    F = Array{Float64,2}(undef, fd, nw);
    F[:, 1] = first_feat;
    for j in 2:nw
        F[:, j] = _feature_vector(windows[:, j]);
    end
    return F;
end

"""
    _logistic_cv_auc(X::AbstractMatrix, y::AbstractVector; nfolds::Int=5, lr=0.05, steps=600, rng)

Fit logistic regression via gradient descent with an intercept term. Features standardized per
fold; returns mean AUC across `nfolds` stratified folds.
"""
function _logistic_cv_auc(X::AbstractMatrix, y::AbstractVector; nfolds::Int=5,
                           lr=0.05, steps=600, rng=Random.GLOBAL_RNG)
    n = size(X, 2); d = size(X, 1);
    pos = findall(==(1), y); neg = findall(==(0), y);
    shuffle!(rng, pos); shuffle!(rng, neg);
    folds_pos = [pos[i:nfolds:end] for i in 1:nfolds];
    folds_neg = [neg[i:nfolds:end] for i in 1:nfolds];
    aucs = Float64[];
    for f in 1:nfolds
        test_idx = sort(vcat(folds_pos[f], folds_neg[f]));
        train_idx = setdiff(1:n, test_idx);
        Xtr = X[:, train_idx]; ytr = y[train_idx];
        Xte = X[:, test_idx];  yte = y[test_idx];
        μ = mean(Xtr, dims=2); σ = std(Xtr, dims=2) .+ 1e-8;
        Xtr_n = (Xtr .- μ) ./ σ;
        Xte_n = (Xte .- μ) ./ σ;
        w = zeros(d); b = 0.0;
        m = size(Xtr_n, 2);
        for _ in 1:steps
            z = vec(w' * Xtr_n .+ b);
            p = 1.0 ./ (1.0 .+ exp.(-z));
            grad_w = (Xtr_n * (p .- ytr)) ./ m;
            grad_b = mean(p .- ytr);
            w .-= lr .* grad_w;
            b  -= lr * grad_b;
        end
        zte = vec(w' * Xte_n .+ b);
        push!(aucs, _auc(zte, yte));
    end
    return mean(aucs), std(aucs);
end

"""
    _auc(scores::AbstractVector, y::AbstractVector)

ROC-AUC via Mann-Whitney U statistic with tie handling.
"""
function _auc(scores::AbstractVector, y::AbstractVector)
    pos = findall(==(1), y); neg = findall(==(0), y);
    np = length(pos); nn = length(neg);
    np == 0 || nn == 0 && return 0.5;
    r = tiedrank(scores);
    sum_rank_pos = sum(r[pos]);
    u = sum_rank_pos - np * (np + 1) / 2.0;
    return u / (np * nn);
end

"""
    discriminator_auc(R_obs::AbstractVector, sim_archive::AbstractMatrix;
                      window::Int=20, n_windows::Int=500, nfolds::Int=5, rng)

Build a balanced real-vs-synthetic dataset of windowed paths, featurize, and report the mean
5-fold CV AUC of a logistic-regression classifier on hand-crafted features.

AUC near 0.5 means the generator is indistinguishable from real under this test; AUC near 1.0
means reviewers can easily tell them apart.
"""
function discriminator_auc(R_obs::AbstractVector, sim_archive::AbstractMatrix;
                           window::Int=20, n_windows::Int=500, nfolds::Int=5,
                           rng=Random.GLOBAL_RNG)
    Wreal = windowize(R_obs, window; stride=max(1, (length(R_obs) - window) ÷ n_windows));
    if size(Wreal, 2) > n_windows
        idx = randperm(rng, size(Wreal, 2))[1:n_windows];
        Wreal = Wreal[:, idx];
    end
    Wsynth = sample_windows_from_archive(sim_archive, window, size(Wreal, 2); rng=rng);
    Freal = _featurize(Wreal);
    Fsyn = _featurize(Wsynth);
    X = hcat(Freal, Fsyn);
    y = vcat(ones(size(Freal, 2)), zeros(size(Fsyn, 2)));
    μ, σ = _logistic_cv_auc(X, y; nfolds=nfolds, rng=rng);
    return (auc=μ, auc_std=σ, n_windows=size(Wreal, 2));
end

# ------------------------------------------------------------------------------------------- #
# A6. Leverage effect
# ------------------------------------------------------------------------------------------- #

"""
    leverage_effect(r::AbstractVector; max_lag::Int=20)

Returns a NamedTuple:
- `profile::Vector{Float64}` of length max_lag, the correlation corr(r_t^2, r_{t-k}) for k=1..max_lag.
- `avg_neg::Float64` mean correlation at positive lags k=1..max_lag (a negative value indicates
  leverage effect: past returns predict subsequent absolute/squared movements).
- `asymmetry::Float64` = mean(|ACF(|r|) on down-preceded| - |ACF(|r|) on up-preceded|).
"""
function leverage_effect(r::AbstractVector; max_lag::Int=20)
    n = length(r);
    r2 = r .^ 2;
    profile = zeros(max_lag);
    for k in 1:max_lag
        a = r[1:n-k]; b = r2[k+1:n];
        profile[k] = cor(a, b);
    end
    avg_neg = mean(profile);
    # Asymmetry: ACF of |r| after down-days vs up-days
    down = findall(x -> x < 0, r[1:end-1]);
    up   = findall(x -> x >= 0, r[1:end-1]);
    abs_next = abs.(r[2:end]);
    mean_down = isempty(down) ? 0.0 : mean(abs_next[down]);
    mean_up = isempty(up) ? 0.0 : mean(abs_next[up]);
    asymmetry = mean_down - mean_up;
    return (profile=profile, avg_neg=avg_neg, asymmetry=asymmetry);
end

# ------------------------------------------------------------------------------------------- #
# A7. Aggregational Gaussianity
# ------------------------------------------------------------------------------------------- #

"""
    aggregational_kurtosis(r::AbstractVector; horizons=[1,5,10,21])

Excess kurtosis of non-overlapping h-day sums of r for each h in horizons.
Standard test: if returns are aggregationally Gaussian, kurtosis should decay toward 0 (excess).
"""
function aggregational_kurtosis(r::AbstractVector; horizons::AbstractVector=[1, 5, 10, 21])
    out = Dict{Int, Float64}();
    for h in horizons
        m = length(r) ÷ h;
        if m < 5; out[h] = NaN; continue; end
        agg = [sum(r[(i-1)*h+1 : i*h]) for i in 1:m];
        μ = mean(agg); σ = std(agg);
        k = σ > 0 ? sum(((agg .- μ) ./ σ) .^ 4) / length(agg) - 3.0 : 0.0;
        out[h] = k;
    end
    return out;
end

# ------------------------------------------------------------------------------------------- #
# A8. VaR LR tests (Kupiec unconditional + Christoffersen independence)
# ------------------------------------------------------------------------------------------- #

"""
    kupiec_lr(breaches::AbstractVector{Bool}, α::Float64)

Unconditional-coverage likelihood-ratio test (Kupiec 1995). Returns (LR, p_value).
Null: the breach rate equals α. Under H0, LR ~ χ²(1); 5 % crit value = 3.841.
"""
function kupiec_lr(breaches::AbstractVector{Bool}, α::Float64)
    n = length(breaches);
    x = sum(breaches);
    π_hat = x / n;
    if π_hat == 0.0 || π_hat == 1.0
        # Degenerate; use the standard clamp
        π_hat = clamp(π_hat, 1e-10, 1 - 1e-10);
    end
    ll_null  = x * log(α) + (n - x) * log(1 - α);
    ll_alt   = x * log(π_hat) + (n - x) * log(1 - π_hat);
    LR = -2 * (ll_null - ll_alt);
    p = 1 - cdf(Chisq(1), LR);
    return (LR=LR, pvalue=p, breach_rate=x/n, n=n, x=x);
end

"""
    christoffersen_lr(breaches::AbstractVector{Bool})

Christoffersen (1998) independence LR test. Checks whether VaR breaches cluster in time.
Returns (LR, p_value). Under H0 (independence), LR ~ χ²(1).
"""
function christoffersen_lr(breaches::AbstractVector{Bool})
    n = length(breaches);
    n00 = 0; n01 = 0; n10 = 0; n11 = 0;
    for t in 2:n
        prev = breaches[t-1]; cur = breaches[t];
        if !prev && !cur; n00 += 1;
        elseif !prev && cur; n01 += 1;
        elseif prev && !cur; n10 += 1;
        else n11 += 1;
        end
    end
    π01 = (n00 + n01) > 0 ? n01 / (n00 + n01) : 0.0;
    π11 = (n10 + n11) > 0 ? n11 / (n10 + n11) : 0.0;
    π   = (n01 + n11) / max(1, n00 + n01 + n10 + n11);
    if π == 0 || π == 1
        return (LR=0.0, pvalue=1.0, n01=n01, n11=n11);
    end
    function _logbern(p::Float64, k::Int, nn::Int)
        if nn == 0; return 0.0; end
        p = clamp(p, 1e-12, 1 - 1e-12);
        return k * log(p) + (nn - k) * log(1 - p);
    end
    ll_null = _logbern(π, n01 + n11, n00 + n01 + n10 + n11);
    ll_alt  = _logbern(π01, n01, n00 + n01) + _logbern(π11, n11, n10 + n11);
    LR = -2 * (ll_null - ll_alt);
    p  = 1 - cdf(Chisq(1), LR);
    return (LR=LR, pvalue=p, n01=n01, n11=n11);
end

"""
    christoffersen_cc(breaches, α)

Christoffersen conditional coverage (joint Kupiec + independence). LR ~ χ²(2).
"""
function christoffersen_cc(breaches::AbstractVector{Bool}, α::Float64)
    k = kupiec_lr(breaches, α);
    c = christoffersen_lr(breaches);
    LR = k.LR + c.LR;
    p  = 1 - cdf(Chisq(2), LR);
    return (LR=LR, pvalue=p, LR_uc=k.LR, LR_ind=c.LR);
end

# ------------------------------------------------------------------------------------------- #
# A9. Simulation-based p-value
# ------------------------------------------------------------------------------------------- #

"""
    sim_pvalue(observed::Real, sim_stats::AbstractVector)

Two-sided simulation-based p-value: the fraction of simulated statistics at least as extreme
as observed, measured from the median of the simulation distribution.
"""
function sim_pvalue(observed::Real, sim_stats::AbstractVector)
    n = length(sim_stats);
    if n == 0; return 1.0; end
    med = median(sim_stats);
    obs_dev = abs(observed - med);
    sim_dev = abs.(sim_stats .- med);
    p = mean(sim_dev .>= obs_dev);
    return p;
end

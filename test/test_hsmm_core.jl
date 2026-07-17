include(joinpath(@__DIR__, "..", "runners", "baselines", "hsmm_core.jl"))

# --------------------------------------------------------------------------- #
# Brute-force enumeration of all right-censored segmentations of a tiny window:
# completed segments carry p_s(d) and a transition factor; the terminal segment
# carries the survival S_s(dc). Ground truth for the forward-backward E-step.
# --------------------------------------------------------------------------- #
function _enum_hsmm_paths!(paths, states, durs, used, T, D, K)
    s = states[end];
    remaining = T - used;
    if remaining <= D
        push!(paths, (copy(states), vcat(durs, remaining)));   # terminal (censored)
    end
    for d in 1:min(D, remaining - 1)
        for snew in 1:K
            snew == s && continue;
            _enum_hsmm_paths!(paths, vcat(states, snew), vcat(durs, d), used + d, T, D, K);
        end
    end
    return paths;
end

@testset "HSMM censored E-step matches brute-force enumeration" begin
    T = 6; K = 2; D = 3;
    R = [0.3, -1.1, 0.6, 2.0, -0.4, 0.9];
    π0 = [0.6, 0.4];
    A0 = [0.0 1.0; 1.0 0.0];
    μ0 = [-0.5, 0.7];
    σ0 = [0.8, 1.2];
    α0 = [0.7, 1.6];
    m = _init_hsmm(R, K, D; init=(π=π0, A=A0, μ=μ0, σ=σ0, α=α0));

    lp = [truncated_pareto_logpmf(α0[s], D) for s in 1:K];
    ls = [truncated_pareto_logsf(α0[s], D) for s in 1:K];

    paths = [];
    for s1 in 1:K
        _enum_hsmm_paths!(paths, [s1], Int[], 0, T, D, K);
    end

    # Enumerated statistics.
    logw = Float64[];
    occ  = zeros(K, T);            # occupancy posterior numerator
    xi   = zeros(K, K);            # expected transition counts numerator
    wcnt = zeros(K, D);            # expected completed duration counts numerator
    ccnt = zeros(K, D);            # expected censored terminal counts numerator
    pini = zeros(K);               # first-segment state numerator
    for (states, durs) in paths
        n = length(states);
        lw = log(π0[states[1]]);
        t = 1;
        for i in 1:n
            s = states[i]; d = durs[i];
            lw += (i < n) ? lp[s][d] : ls[s][d];
            for τ in t:(t + d - 1)
                lw += -0.5*((R[τ]-μ0[s])/σ0[s])^2 - log(σ0[s]) - 0.5*log(2π);
            end
            if i < n
                lw += log(A0[states[i], states[i+1]]);
            end
            t += d;
        end
        push!(logw, lw);
        w = exp(lw);
        t = 1;
        for i in 1:n
            s = states[i]; d = durs[i];
            for τ in t:(t + d - 1); occ[s, τ] += w; end
            if i < n
                wcnt[s, d] += w;
                xi[states[i], states[i+1]] += w;
            else
                ccnt[s, d] += w;
            end
            t += d;
        end
        pini[states[1]] += w;
    end
    mx = maximum(logw);
    ll_enum = mx + log(sum(exp.(logw .- mx)));
    Z = exp(ll_enum);
    occ ./= Z; xi ./= Z; wcnt ./= Z; ccnt ./= Z; pini ./= Z;

    post = _e_step(m, R, D);

    # Likelihood, both recursion directions.
    @test post.ll ≈ ll_enum atol=1e-10
    @test _logsumexp(log.(π0) .+ post.logg[:, 1]) ≈ ll_enum atol=1e-10

    # Occupancy posterior: exact match and unit column sums (including t = T).
    γ = exp.(post.log_γ);
    @test maximum(abs.(γ .- occ)) < 1e-10
    @test all(abs.(sum(γ, dims=1) .- 1.0) .< 1e-10)

    # Expected transition counts.
    ξsum = [sum(exp.(post.log_ξ[sp, s, :])) for sp in 1:K, s in 1:K];
    @test maximum(abs.(ξsum .- xi)) < 1e-10

    # Completed and censored duration counts; censored mass sums to one.
    w_est = zeros(K, D); c_est = zeros(K, D);
    for s in 1:K, t in 1:(T-1), d in 1:min(t, D)
        w_est[s, d] += exp(post.log_eta[s, t, d]);
    end
    for s in 1:K, dc in 1:min(D, T)
        c_est[s, dc] = exp(post.log_eta_c[s, dc]);
    end
    @test maximum(abs.(w_est .- wcnt)) < 1e-10
    @test maximum(abs.(c_est .- ccnt)) < 1e-10
    @test sum(c_est) ≈ 1.0 atol=1e-10

    # First-segment posterior (π M-step numerator): η(s, d, d) + η_c(s, T).
    pini_est = zeros(K);
    for s in 1:K
        for d in 1:min(D, T - 1)
            pini_est[s] += exp(post.log_eta[s, d, d]);
        end
        if T <= D
            pini_est[s] += exp(post.log_eta_c[s, T]);
        end
    end
    # (here T = 6 > D = 3, so the censored whole-series branch is inactive)
    @test maximum(abs.(pini_est .- pini)) < 1e-10
end

@testset "Truncated Pareto survival and censored M-step" begin
    D = 20; α = 1.2;
    lp = truncated_pareto_logpmf(α, D);
    ls = truncated_pareto_logsf(α, D);
    @test exp(ls[1]) ≈ 1.0 atol=1e-12
    @test all(diff(ls) .< 0.0)                       # survival strictly decreasing
    @test ls[D] ≈ lp[D] atol=1e-12
    @test exp(ls[5]) ≈ sum(exp.(lp[5:D])) atol=1e-12

    # No-censoring path unchanged: recovers the exact-counts optimum.
    w_exact = exp.(truncated_pareto_logpmf(1.2, D)) .* 500.0;
    @test abs(fit_truncated_pareto_alpha(w_exact, D) - 1.2) < 1e-3

    # Zero censored counts must agree with the no-censoring path.
    α_nc = fit_truncated_pareto_alpha(w_exact, D);
    α_zc = fit_truncated_pareto_alpha(w_exact, D; censored_counts=zeros(D));
    @test abs(α_nc - α_zc) < 1e-4

    # Grid maximality with genuine censored mass.
    c = zeros(D); c[3] = 4.0; c[10] = 2.0; c[20] = 1.0;
    α̂ = fit_truncated_pareto_alpha(w_exact ./ 10.0, D; censored_counts=c);
    ℓ(a) = sum((w_exact[d] / 10.0) * truncated_pareto_logpmf(a, D)[d] +
               c[d] * truncated_pareto_logsf(a, D)[d] for d in 1:D);
    grid = 0.06:0.005:6.0;
    @test ℓ(α̂) >= maximum(ℓ.(grid)) - 1e-6

    # Censored mass at long durations must pull α̂ DOWN vs completed-only.
    α_heavy = fit_truncated_pareto_alpha(w_exact, D; censored_counts=vcat(zeros(D-1), 50.0));
    @test α_heavy < α_nc
end

@testset "HSMM EM monotone under the self-consistent censored likelihood" begin
    Random.seed!(4242);
    K = 2; D = 50;
    m_true = _init_hsmm(randn(100), K, D;
                        init=(π=[0.5, 0.5], A=[0.0 1.0; 1.0 0.0],
                              μ=[0.0, 0.1], σ=[0.6, 2.0], α=[0.6, 1.8]));
    R = vec(simulate_hsmm(m_true; T=400, D=D, n_paths=1));
    mdl = fit_hsmm_ml(R, K; D=D, max_iter=15, tol=0.0, verbose=false);
    @test all(diff(mdl.ll_history) .> -1e-8)
    @test length(mdl.ll_history) == 16          # max_iter + 1 evaluations at tol = 0
end

@testset "HSMM multistart contract" begin
    Random.seed!(777);
    R = randn(300) .* (1.0 .+ 0.5 .* (rand(300) .> 0.7));
    best, diags = fit_hsmm_ml_multistart(R, 2; n_starts=3, D=30, max_iter=8,
                                         tol=1e-6, seed=99, verbose=false);
    @test length(diags) == 3
    @test best.ll_history[end] ≈ maximum(d.ll for d in diags) atol=1e-12
    for d in diags
        @test d.converged == (abs(d.final_increment) / 300 < 1e-6)
    end
end

# =========================================================================== #
# run_christoffersen_power.jl
#
# Peer-review item 4 (R1 Q4 / R2 W2-RE1 / R3 Q5):
# Monte Carlo power calibration of the Christoffersen-cc joint conditional-coverage
# test at T_OoS = 572 across a range of breach-clustering levels.
#
# Construction:
#   simulate breach indicator sequence I_t in {0,1} of length T = 572 from a
#   two-state Markov chain {breach, no-breach} with marginal breach probability
#   alpha (1% or 5%) and second eigenvalue rho. rho = 0 is the iid null;
#   rho > 0 means breaches cluster (positive serial correlation in I_t).
#
# Test:
#   Christoffersen-cc combines unconditional Kupiec (LR_uc) + breach-independence
#   (LR_ind) into LR_cc ~ chi^2_2 under the null.
#
# Output: results/diagnostics/christoffersen_power/christoffersen_power.txt
# =========================================================================== #

using Pkg; Pkg.activate(".");
using Random, Statistics, Printf, Distributions;

const SEED   = 20260420;
const OUTDIR = joinpath(@__DIR__, "results", "diagnostics", "christoffersen_power");
mkpath(OUTDIR);

const T_OOS  = 572;          # OoS length used in the paper
const B      = 5_000;        # MC replicates per (alpha, rho) cell
const ALPHAS = [0.01, 0.05];
const RHOS   = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50];

# Christoffersen tests (standard formulation).
function _christoffersen(I::Vector{Int}, alpha::Real)
    T = length(I);
    n1 = sum(I);
    n0 = T - n1;
    pi_hat = n1 / T;

    # LR_uc = -2 [(n1 log(alpha) + n0 log(1-alpha)) - (n1 log(pi_hat) + n0 log(1-pi_hat))]
    # Guard against pi_hat == 0 or 1.
    if pi_hat <= 0.0 || pi_hat >= 1.0
        LR_uc = 0.0;  # degenerate; do not contribute
    else
        ll_null = n1 * log(alpha)   + n0 * log(1 - alpha);
        ll_alt  = n1 * log(pi_hat)  + n0 * log(1 - pi_hat);
        LR_uc = -2 * (ll_null - ll_alt);
    end

    # Transitions for LR_ind
    n00 = 0; n01 = 0; n10 = 0; n11 = 0;
    @inbounds for t in 2:T
        prev, cur = I[t-1], I[t];
        if prev == 0 && cur == 0
            n00 += 1;
        elseif prev == 0 && cur == 1
            n01 += 1;
        elseif prev == 1 && cur == 0
            n10 += 1;
        elseif prev == 1 && cur == 1
            n11 += 1;
        end
    end
    pi01 = (n00 + n01) > 0 ? n01 / (n00 + n01) : 0.0;
    pi11 = (n10 + n11) > 0 ? n11 / (n10 + n11) : 0.0;
    pi_uc = (n01 + n11) / (n00 + n01 + n10 + n11);

    # LR_ind under null pi01 == pi11 == pi_uc
    function _safe_log_term(n, p)
        return n == 0 ? 0.0 : n * log(max(p, 1e-300));
    end
    if pi_uc <= 0.0 || pi_uc >= 1.0
        LR_ind = 0.0;
    else
        ll_null_ind = _safe_log_term(n00, 1 - pi_uc) + _safe_log_term(n01, pi_uc) +
                      _safe_log_term(n10, 1 - pi_uc) + _safe_log_term(n11, pi_uc);
        ll_alt_ind  = _safe_log_term(n00, 1 - pi01)  + _safe_log_term(n01, pi01) +
                      _safe_log_term(n10, 1 - pi11)  + _safe_log_term(n11, pi11);
        LR_ind = -2 * (ll_null_ind - ll_alt_ind);
    end

    LR_cc = LR_uc + LR_ind;
    return (LR_uc = LR_uc, LR_ind = LR_ind, LR_cc = LR_cc, pi_hat = pi_hat,
            n1 = n1, n00 = n00, n01 = n01, n10 = n10, n11 = n11);
end

# Simulate breach indicator sequence with marginal alpha and second eigenvalue rho.
# Two-state Markov chain on I_t with transition matrix P:
#   P_00 = 1 - p01, P_01 = p01, P_10 = p10, P_11 = 1 - p10
# Marginal: alpha = p01 / (p01 + p10);
# Second eigenvalue: 1 - p01 - p10 = rho  =>  p01 + p10 = 1 - rho
# Combine: p01 = alpha (1 - rho),  p10 = (1 - alpha)(1 - rho)
function _simulate_breach_chain(alpha::Real, rho::Real, T::Int; rng::AbstractRNG)
    p01 = alpha * (1 - rho);
    p10 = (1 - alpha) * (1 - rho);
    I = Vector{Int}(undef, T);
    # initialise from stationary
    I[1] = rand(rng) < alpha ? 1 : 0;
    @inbounds for t in 2:T
        if I[t-1] == 0
            I[t] = rand(rng) < p01 ? 1 : 0;
        else
            I[t] = rand(rng) < (1 - p10) ? 1 : 0;
        end
    end
    return I;
end

# Critical values
const CHI2_1_05 = quantile(Chisq(1), 0.95);
const CHI2_2_05 = quantile(Chisq(2), 0.95);

println("=" ^ 78);
println("  Christoffersen-cc MC power calibration at T_OoS = $T_OOS");
println("  Replicates per cell: $B   alphas = $ALPHAS   rhos = $RHOS");
println("=" ^ 78);

results = NamedTuple{(:alpha, :rho, :reject_uc, :reject_ind, :reject_cc, :mean_breaches),
                     Tuple{Float64,Float64,Float64,Float64,Float64,Float64}}[];

for alpha in ALPHAS
    for rho in RHOS
        rng = Random.MersenneTwister(SEED + Int(round(1e6 * alpha + 1e3 * rho)));
        rej_uc  = 0;
        rej_ind = 0;
        rej_cc  = 0;
        breaches = 0;
        for b in 1:B
            I = _simulate_breach_chain(alpha, rho, T_OOS; rng = rng);
            r = _christoffersen(I, alpha);
            breaches += r.n1;
            if r.LR_uc  > CHI2_1_05; rej_uc  += 1; end
            if r.LR_ind > CHI2_1_05; rej_ind += 1; end
            if r.LR_cc  > CHI2_2_05; rej_cc  += 1; end
        end
        r_uc  = rej_uc  / B;
        r_ind = rej_ind / B;
        r_cc  = rej_cc  / B;
        m_br  = breaches / B;
        push!(results, (alpha = alpha, rho = rho,
                        reject_uc = r_uc, reject_ind = r_ind, reject_cc = r_cc,
                        mean_breaches = m_br));
        @printf("  alpha=%.2f  rho=%.2f  | breaches~%5.1f  | rej UC=%5.1f%%  IND=%5.1f%%  CC=%5.1f%%\n",
                alpha, rho, m_br, 100 * r_uc, 100 * r_ind, 100 * r_cc);
    end
end

OUT = joinpath(OUTDIR, "christoffersen_power.txt");
open(OUT, "w") do io
    println(io, "Christoffersen-cc / -ind / Kupiec MC power calibration");
    println(io, "Peer-review item 4 (R1 Q4 / R2 W2-RE1 / R3 Q5).");
    println(io, "T_OoS = $T_OOS, replicates B = $B per (alpha, rho) cell, seed = $SEED.");
    println(io, "Critical values: chi^2_1(0.05) = $(round(CHI2_1_05, digits=3)),  chi^2_2(0.05) = $(round(CHI2_2_05, digits=3)).");
    println(io);
    println(io, "Construction: breach indicator I_t simulated from a two-state Markov chain");
    println(io, "with marginal alpha and second eigenvalue rho. rho = 0 is the iid null;");
    println(io, "rho > 0 means breaches cluster (positive serial correlation in I_t).");
    println(io);
    println(io, "Empirical rejection rate at nominal alpha = 0.05:");
    println(io, "alpha    rho     mean_breaches   rej_UC%   rej_IND%   rej_CC%");
    println(io, "-"^70);
    for r in results
        @printf(io, "%5.2f  %5.2f   %12.1f   %7.2f   %8.2f   %7.2f\n",
                r.alpha, r.rho, r.mean_breaches,
                100 * r.reject_uc, 100 * r.reject_ind, 100 * r.reject_cc);
    end
    println(io);
    println(io, "Reading:");
    println(io, "  rho = 0 row: empirical Type-I error under iid null.");
    println(io, "    Should sit at 5% if the asymptotic chi^2 calibration holds at T = 572.");
    println(io, "  rho > 0 rows: power against Markov-clustered breach sequences.");
    println(io);
    println(io, "Headline interpretation for the paper:");
    # find rho needed for 80% power at alpha=0.05
    for alpha in ALPHAS
        block = filter(r -> r.alpha == alpha, results);
        rho_for_80 = nothing;
        for r in block
            if r.reject_cc >= 0.80
                rho_for_80 = r.rho; break;
            end
        end
        if rho_for_80 !== nothing
            @printf(io, "  alpha = %.2f: Christoffersen-cc reaches 80%% power at rho >= %.2f.\n",
                    alpha, rho_for_80);
        else
            @printf(io, "  alpha = %.2f: Christoffersen-cc does not reach 80%% power within tested rho range; ", alpha);
            r_max = block[end];
            @printf(io, "even at rho = %.2f, CC rejection rate is %.1f%%.\n", r_max.rho, 100 * r_max.reject_cc);
        end
    end
end

println("[done] $OUT");

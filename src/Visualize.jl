"""
    plot_acf_comparison(observed::Vector, simulated::Vector, title_text::String, random_index::Int; is_absolute::Bool=false, L::Int=252)

Plots the autocorrelation function (ACF) for an observed time series and a single simulated time series on the same graph.

### Arguments
- `observed::Vector`: The vector of observed data (e.g., historical returns).
- `simulated::Vector`: A single vector representing one path of simulated data.
- `title_text::String`: The title for the plot.
- `random_index::Int`: The index of the simulated path, used for the legend.
- `is_absolute::Bool=false`: (Keyword) If `true`, the ACF of the absolute values of the series is plotted to show volatility clustering. Defaults to `false`.
- `L::Int=252`: (Keyword) The maximum lag to calculate the autocorrelation for. Defaults to 252 (approx. 1 trading year).

### Returns
- `Plots.Plot`: The generated plot object.
"""
function plot_acf_comparison(observed::Vector, simulated::Vector, title_text::String, random_index::Int; is_absolute::Bool=false, L::Int=252)

    data_obs = is_absolute ? abs.(observed) : observed
    data_sim = is_absolute ? abs.(simulated) : simulated

    # L is now passed as a keyword argument (default 252)
    τ = 1:(L-1)
    ci = 2.576 / sqrt(length(data_obs))

    ac_obs = autocor(data_obs, τ)
    ac_sim = autocor(data_sim, τ)
    
    # Create the plot with a smaller title font size
    p = plot(τ, ac_obs, label="Observed", linetype=:steppost, lw=2, c=:red, legend=:topright, title=title_text, titlefontsize=10)
    
    # Update the label to include the random_index
    plot!(p, τ, ac_sim, label="Simulation (Path $(random_index))", linetype=:steppost, lw=2, c=:blue)
    
    plot!(p, τ, ci * ones(length(τ)), label="99% CI", lw=1.5, c=:gray, ls=:dash)
    plot!(p, τ, -ci * ones(length(τ)), label="", lw=1.5, c=:gray, ls=:dash)
    xlabel!(p, "Lag (trading day)")
    
    # Add "(AU)" to the ylabel for Arbitrary Units / Unitless
    ylabel!(p, "Autocorrelation (AU)")

    return p
end


"""
    plot_regime_overlay(dates, prices, states, ticker; title_text="")

Plots price time series with regime-colored background shading.

### Arguments
- `dates::Vector`: Date index for x-axis.
- `prices::Vector{Float64}`: Price series (e.g., close prices).
- `states::Vector{Int64}`: Decoded hidden state sequence.
- `ticker::String`: Ticker label for axis/title.
- `title_text::String`: Optional custom title.

### Returns
- `Plots.Plot`: The generated plot object.
"""
function plot_regime_overlay(dates::Vector, prices::Vector{Float64}, states::Vector{Int64}, ticker::String; title_text::String="")

    n_states = length(unique(states))
    palette = cgrad(:RdYlGn, n_states, categorical=true)

    if isempty(title_text)
        title_text = "$(ticker) — Regime Overlay (K=$(n_states))"
    end

    p = plot(dates, prices, label=ticker, lw=1.5, c=:black, title=title_text, titlefontsize=10, legend=:topleft)

    # shade by regime
    for i in 1:length(dates)
        s = states[i]
        vspan!(p, [dates[max(1,i)], dates[min(length(dates),i)]], alpha=0.15, c=palette[s], label="")
    end

    ylabel!(p, "$(ticker) Level")

    return p
end


"""
    plot_emission_pdfs(model::MyContinuousHiddenMarkovModel, ticker::String; xlabel="Log Return")

Plots the Gaussian emission PDF for each hidden state in the model.
X-range is computed adaptively from the emission parameters (μ ± 4σ).

### Returns
- `Plots.Plot`
"""
function plot_emission_pdfs(model::MyContinuousHiddenMarkovModel, ticker::String; xlabel::String="Log Return")

    K = length(model.states)

    # Data-adaptive x-range: cover μ ± 4σ of the widest emission
    all_lo = minimum(mean(model.emission[s]) - 4*std(model.emission[s]) for s in model.states)
    all_hi = maximum(mean(model.emission[s]) + 4*std(model.emission[s]) for s in model.states)
    x = range(all_lo, all_hi, length=1000)

    p = plot(title="Emission Distributions — $(ticker) (K=$(K))", titlefontsize=10,
             xlabel=xlabel, ylabel="Probability Density (AU)", legend=:topright)

    palette = cgrad(:RdYlGn, K, categorical=true)

    for s in model.states
        d = model.emission[s]
        μ_s = round(mean(d), digits=5)
        σ_s = round(std(d), digits=5)
        plot!(p, x, pdf.(d, x), label="State $(s) (μ=$(μ_s), σ=$(σ_s))", lw=2, c=palette[s], fillalpha=0.15, fill=true)
    end

    return p
end


# --- PRICING VISUALIZATION --------------------------------------------------- #

"""
    plot_implied_vol_surface(strikes, expiries, iv_matrix, title_text) -> Plots.Plot

Heatmap of the implied volatility surface (strikes × expiries).
"""
function plot_implied_vol_surface(strikes::Array{Float64,1}, expiries::Array{Float64,1},
    iv_matrix::Array{Float64,2}, title_text::String)

    strike_labels = [string(round(k, digits=0)) for k in strikes];
    expiry_labels = [string(round(t, digits=2)) for t in expiries];

    p = heatmap(expiry_labels, strike_labels, iv_matrix .* 100,
        title=title_text, titlefontsize=10,
        xlabel="Expiry (years)", ylabel="Strike",
        color=:viridis, colorbar_title="Implied Vol (%)",
        size=(600, 450));

    return p;
end


"""
    plot_pricing_comparison(chmm_result, bs_price, title_text) -> Plots.Plot

Bar chart comparing the CHMM regime-switching Monte Carlo price against the
Black-Scholes analytical benchmark, with 95% confidence intervals on the MC
estimate.
"""
function plot_pricing_comparison(chmm_result::MyPricingResult,
    bs_price::Float64, title_text::String)

    labels = ["CHMM", "Black-Scholes"];
    prices = [chmm_result.price, bs_price];
    errors = [1.96 * chmm_result.std_error, 0.0];

    p = bar(labels, prices, yerr=errors, title=title_text, titlefontsize=10,
        ylabel="Option Price (\$)", legend=false,
        color=[:steelblue, :gray], alpha=0.8,
        bar_width=0.6, size=(500, 400));

    return p;
end


"""
    plot_mc_convergence(result::MyPricingResult, title_text::String) -> Plots.Plot

Running average of discounted payoffs vs. path count, showing Monte Carlo convergence.
"""
function plot_mc_convergence(result::MyPricingResult, title_text::String)

    n = result.n_paths;
    running_mean = cumsum(result.payoffs) ./ (1:n);

    p = plot(1:n, running_mean, lw=2, color=:navy, label="Running Mean",
        title=title_text, titlefontsize=10,
        xlabel="Number of Paths", ylabel="Price Estimate (\$)");
    hline!(p, [result.price], lw=1.5, color=:red, ls=:dash, label="Final: \$$(round(result.price, digits=4))");

    return p;
end
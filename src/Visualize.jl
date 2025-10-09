"""
    plot_acf_comparison(observed::Vector, simulated::Vector, title_text::String, random_index::Int; is_absolute::Bool=false)

Plots the autocorrelation function (ACF) for an observed time series and a single simulated time series on the same graph.

### Arguments
- `observed::Vector`: The vector of observed data (e.g., historical returns).
- `simulated::Vector`: A single vector representing one path of simulated data.
- `title_text::String`: The title for the plot.
- `random_index::Int`: The index of the simulated path, used for the legend.
- `is_absolute::Bool=false`: (Keyword) If `true`, the ACF of the absolute values of the series is plotted to show volatility clustering. Defaults to `false`.

### Returns
- `Plots.Plot`: The generated plot object.
"""
function plot_acf_comparison(observed::Vector, simulated::Vector, title_text::String, random_index::Int; is_absolute::Bool=false)

    data_obs = is_absolute ? abs.(observed) : observed
    data_sim = is_absolute ? abs.(simulated) : simulated

    L = 252
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
end;
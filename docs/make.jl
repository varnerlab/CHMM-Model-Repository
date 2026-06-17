using Documenter

# Load the project
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "Include.jl"))

makedocs(
    sitename = "ContinuousHMM",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Fitting" => "fitting.md",
        "Simulation" => "simulation.md",
        "Validation" => "validation.md",
        "Finance Utilities" => "finance.md",
        "API Reference" => "api.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
)

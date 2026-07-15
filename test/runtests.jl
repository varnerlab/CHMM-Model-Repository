using Test

# Headless GR workstation so the Visualize tests run without a display
# (CI, SSH, Pkg.test sandboxes); same guard as runners/headline/run_quantgan_baseline.jl.
ENV["GKSwstype"] = "100"

# Load the project
include(joinpath(@__DIR__, "..", "Include.jl"))

@testset "ContinuousJumpHMM" begin
    include("test_types.jl")
    include("test_factory.jl")
    include("test_compute.jl")
    include("test_crossasset.jl")
    include("test_visualize.jl")
    include("test_msgarch_reference.jl")
end

using Test

# Load the project
include(joinpath(@__DIR__, "..", "Include.jl"))

@testset "ContinuousJumpHMM" begin
    include("test_types.jl")
    include("test_factory.jl")
    include("test_compute.jl")
    include("test_visualize.jl")
end

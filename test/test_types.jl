@testset "Types" begin

    @testset "Abstract type hierarchy" begin
        @test MyContinuousHiddenMarkovModel <: AbstractMarkovModel
        @test StudentTModel <: AbstractDistributionModel
        @test LaplaceModel <: AbstractDistributionModel
    end

    @testset "Constructors create empty instances" begin
        m1 = MyContinuousHiddenMarkovModel()
        @test m1 isa MyContinuousHiddenMarkovModel
    end
end

@testset "Types" begin

    @testset "Abstract type hierarchy" begin
        @test MyContinuousHiddenMarkovModel <: AbstractMarkovModel
        @test MyContinuousHiddenMarkovModelWithJumps <: AbstractMarkovModel
        @test MyHiddenMarkovModel <: AbstractMarkovModel
        @test MyHiddenMarkovModelWithJumps <: AbstractMarkovModel
        @test StudentTModel <: AbstractDistributionModel
        @test LaplaceModel <: AbstractDistributionModel
    end

    @testset "Constructors create empty instances" begin
        m1 = MyContinuousHiddenMarkovModel()
        @test m1 isa MyContinuousHiddenMarkovModel

        m2 = MyContinuousHiddenMarkovModelWithJumps()
        @test m2 isa MyContinuousHiddenMarkovModelWithJumps

        m3 = MyHiddenMarkovModel()
        @test m3 isa MyHiddenMarkovModel

        m4 = MyHiddenMarkovModelWithJumps()
        @test m4 isa MyHiddenMarkovModelWithJumps
    end
end

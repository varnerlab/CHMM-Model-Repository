@testset "Types" begin

    @testset "Abstract type hierarchy" begin
        @test MyContinuousHiddenMarkovModel <: AbstractMarkovModel
        @test MyHiddenMarkovModel <: AbstractMarkovModel
        @test MyHiddenMarkovModelWithJumps <: AbstractMarkovModel
        @test StudentTModel <: AbstractDistributionModel
        @test LaplaceModel <: AbstractDistributionModel
    end

    @testset "Constructors create empty instances" begin
        m1 = MyContinuousHiddenMarkovModel()
        @test m1 isa MyContinuousHiddenMarkovModel

        m2 = MyHiddenMarkovModel()
        @test m2 isa MyHiddenMarkovModel

        m3 = MyHiddenMarkovModelWithJumps()
        @test m3 isa MyHiddenMarkovModelWithJumps

        m4 = MyGARCHModel()
        @test m4 isa MyGARCHModel
    end
end

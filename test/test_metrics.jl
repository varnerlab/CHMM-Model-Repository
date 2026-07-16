@testset "Metrics utilities" begin

    @testset "_auc degenerate label sets return 0.5" begin
        # (2026-07-16 fifth review, code finding 8: `np == 0 || nn == 0 && return 0.5`
        # short-circuited so an empty POSITIVE class fell through to a 0/0 division.)
        scores = [0.1, 0.4, 0.35, 0.8, 0.65]
        @test _auc(scores, zeros(Int, 5)) == 0.5   # no positives
        @test _auc(scores, ones(Int, 5)) == 0.5    # no negatives
    end

    @testset "_auc separable and random labelings" begin
        y = [0, 0, 0, 1, 1, 1]
        @test _auc([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], y) == 1.0   # perfectly separable
        @test _auc([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], y) == 0.0   # perfectly inverted
        @test _auc([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], y) == 0.5   # all tied
    end
end

@testset "Pricing" begin

    function _make_contract(; S0=100.0, K=100.0, T=1.0, r=0.05, is_call=true)
        return build(MyEuropeanOptionContract, (
            S0=S0, K=K, T=T, r=r, is_call=is_call))
    end


    @testset "Black-Scholes put-call parity" begin
        call = _make_contract(is_call=true)
        put = _make_contract(is_call=false)
        σ = 0.20

        C = black_scholes(call, σ)
        P = black_scholes(put, σ)

        parity = call.S0 - call.K * exp(-call.r * call.T)
        @test (C - P) ≈ parity atol=1e-10
    end

    @testset "Black-Scholes ATM sanity" begin
        contract = _make_contract()
        σ = 0.20

        price_val = black_scholes(contract, σ)

        @test price_val > 0.0
        approx = contract.S0 * σ * sqrt(contract.T) * 0.4
        @test abs(price_val - approx) < 5.0
    end

    @testset "Implied volatility inversion" begin
        contract = _make_contract()
        σ_true = 0.20

        bs_price = black_scholes(contract, σ_true)
        σ_recovered = implied_volatility(contract, bs_price)

        @test σ_recovered ≈ σ_true atol=1e-5
    end
end

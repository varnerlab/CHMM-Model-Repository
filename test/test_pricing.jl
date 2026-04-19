@testset "Pricing" begin

    # --- Helpers: build synthetic CHMM pricing model ---
    function _make_chmm_pricer(; n_states=5, n_paths=1000, seed=42)
        rng = Random.MersenneTwister(seed)
        obs = vcat(randn(rng, 200) .* 0.01, randn(rng, 200) .* 0.05)
        hmm = build(MyContinuousHiddenMarkovModel, (
            observations = shuffle(rng, obs),
            number_of_states = n_states,
            max_iter = 15
        ))

        # Synthetic VIX prices + Viterbi-decoded states
        vix_prices = 15.0 .+ 5.0 .* randn(rng, 400)
        vix_prices = max.(vix_prices, 5.0)  # VIX can't go below ~5
        vix_states = viterbi(shuffle(rng, obs), hmm)

        return build(MyCHMMPricingModel, (
            hmm = hmm,
            vix_prices = vix_prices,
            vix_states = vix_states,
            n_paths = n_paths,
            n_steps_per_year = 252
        ))
    end

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

        # Put-call parity: C - P = S0 - K*exp(-r*T)
        parity = call.S0 - call.K * exp(-call.r * call.T)
        @test (C - P) ≈ parity atol=1e-10
    end

    @testset "Black-Scholes ATM sanity" begin
        contract = _make_contract()
        σ = 0.20

        price_val = black_scholes(contract, σ)

        @test price_val > 0.0
        # ATM call ≈ S0 * σ * √T * 0.4 for small r
        approx = contract.S0 * σ * sqrt(contract.T) * 0.4
        @test abs(price_val - approx) < 5.0  # within $5 of approximation
    end

    @testset "CHMM pricer returns valid result" begin
        pricer = _make_chmm_pricer(n_paths=500)
        contract = _make_contract()

        result = price(pricer, contract)

        @test result.price > 0.0
        @test result.std_error > 0.0
        @test result.n_paths == 500
        @test length(result.payoffs) == 500
    end

    @testset "Implied volatility inversion" begin
        contract = _make_contract()
        σ_true = 0.20

        # Price at known vol, then invert
        bs_price = black_scholes(contract, σ_true)
        σ_recovered = implied_volatility(contract, bs_price)

        @test σ_recovered ≈ σ_true atol=1e-5
    end

    @testset "CHMM MC convergence: more paths reduce std error" begin
        pricer_small = _make_chmm_pricer(n_paths=100, seed=101)
        pricer_large = _make_chmm_pricer(n_paths=5000, seed=101)
        contract = _make_contract()

        result_small = price(pricer_small, contract)
        result_large = price(pricer_large, contract)

        @test result_large.std_error < result_small.std_error
    end
end

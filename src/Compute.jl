function _simulate(m::MyHiddenMarkovModel, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    chain[1] = start;

    # main loop -
    for i ∈ 2:steps
        chain[i] = rand(m.transition[chain[i-1]]);
    end

    return chain;
end

function _simulate(m::MyHiddenMarkovModelWithJumps, start::Int64, steps::Int64)::Array{Int64,1}

    # initialize -
    chain = Array{Int64,1}(undef, steps);
    tmp_chain = Dict{Int64,Int64}();
    tmp_chain[1] = start;
    counter = 2;

    # main -
    jump_state = start;
    while (counter ≤ steps)
        
        if (rand() < m.ϵ)

            # # jump: find the next state. It is lowest probability state from here
            number_of_jumps = rand(m.jump_distribution);
            number_of_states = length(m.states);
            bottom_states = [1,2,3]; # super bad
            top_states = [number_of_states-2,number_of_states-1,number_of_states]; # super good
 
            @show number_of_jumps

            for _ ∈ 1:number_of_jumps
                if (rand() < 0.52)
                    tmp_chain[counter] = rand(bottom_states) # a jump transition to bottom states
                else
                    tmp_chain[counter] = rand(top_states) # a jump transition to top states
                end
                counter += 1;
            end
        else
            tmp_chain[counter] = rand(m.transition[jump_state]); # a normal transition
            counter += 1; # increment counter
        end

        jump_state = tmp_chain[counter-1]; # get the last state
    end

    # populate the chain from tmp_chain -
    for i ∈ 1:steps
        chain[i] = tmp_chain[i];
    end

    # return -
    return chain;
end

(m::MyHiddenMarkovModel)(start::Int64, steps::Int64) = _simulate(m, start, steps); 
(m::MyHiddenMarkovModelWithJumps)(start::Int64, steps::Int64) = _simulate(m, start, steps); 

# function log_growth_matrix(dataset::Dict{String, DataFrame}, 
#     firms::Array{String,1}; Δt::Float64 = (1.0/252.0), risk_free_rate::Float64 = 0.0, 
#     testfirm="AAPL", keycol::Symbol = :volume_weighted_average_price)::Array{Float64,2}

#     # initialize -
#     number_of_firms = length(firms);
#     number_of_trading_days = nrow(dataset[testfirm]);
#     return_matrix = Array{Float64,2}(undef, number_of_trading_days-1, number_of_firms);

#     # main loop -
#     for i ∈ eachindex(firms) 

#         # get the firm data -
#         firm_index = firms[i];
#         firm_data = dataset[firm_index];

#         # compute the log returns -
#         for j ∈ 2:number_of_trading_days
#             S₁ = firm_data[j-1, keycol];
#             S₂ = firm_data[j, keycol];
#             return_matrix[j-1, i] = (1/Δt)*(log(S₂/S₁)) - risk_free_rate;
#         end
#     end

#     # return -
#     return return_matrix;
# end

function vwap(df::DataFrame)::Array{Float64,1}

    # Get the number of rows in the DataFrame
    n = nrow(df)
    
    # Initialize an array to store the VWAP values
    vwap_array = Array{Float64,1}(undef, n)
    
    # Initialize cumulative price and volume
    cumulative_pv = 0.0  # sum of price * volume
    cumulative_volume = 0.0

    # Calculate VWAP for each row
    for i in 1:n
        typical_price = (df.high[i] + df.low[i] + df.close[i]) / 3
        volume = df.volume[i]

        cumulative_pv += typical_price * volume
        cumulative_volume += volume

        vwap_array[i] = cumulative_pv / cumulative_volume
    end

    # Return the VWAP array
    return vwap_array
end


"""
    learn_return_distribution_mcmc(returns::Vector{Float64}; samples::Int = 2000)

Uses a Bayesian MCMC approach to learn the parameters of a Student's t-distribution
fitted to the equity returns data.

Returns a Turing.jl `Chain` object containing the posterior distributions of the parameters.
"""
function learn_distribution_mcmc(model_type::AbstractDistributionModel, returns::Vector{Float64}; samples::Int = 2000)
    
    # 1. Build the correct model based on the input type (e.g., StudentTModel())
    #    Julia's multiple dispatch calls the correct function from Factory.jl
    model_instance = build_turing_model(model_type, returns)

    # 2. Run the MCMC sampler
    chain = Turing.sample(model_instance, NUTS(), samples)

    # 3. Return the resulting chain
    return chain
end
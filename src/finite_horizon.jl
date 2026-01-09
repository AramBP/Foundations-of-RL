Base.@kwdef struct WithTime{S}
    state::S
    time::Int = 0
end

"""
    This function accepts 
"""
function unwrap_finite_horizon_mrp(process::FiniteMarkovRewardProcess{WithTime{S}}) where {S}
    time(x::WithTime) = x.time

    function single_without_time(state_reward::Tuple{State{WithTime{S}}, Float64})
        if isa(state_reward[1], NonTerminal)
            return (NonTerminal(state_reward[1].state.state), state_reward[2])
        else
            return (Terminal(state_reward[1].state.state), state_reward[2])
        end
    end

    function without_time(arg::FiniteDistribution)
        map()
    end
end


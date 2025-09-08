module MarkovRewardProcesses

using Distributions, DataStructures, LinearAlgebra
using RL.MarkovProcesses: MarkovProcess, FiniteMarkovProcess, NonTerminal, Terminal, State, get_transition_matrix
using RL.DistributionsExt: FiniteDistribution, LabeledCategorical

export MarkovRewardProcess, transition_reward, simulate, transition, FiniteMarkovRewardProcess, transition_reward, get_value_function_vec

struct TransitionStep{S}
    state::NonTerminal{S}
    next_state::State{S}
    reward::Float64
end

abstract type MarkovRewardProcess{S} <: MarkovProcess{S} end

transition_reward(mrp::MarkovRewardProcess, state::NonTerminal) = error("transition function not specified for $(typeof(mrp)), state: $(state)")
function simulate(mrp::MarkovRewardProcess, start_state_distribution::Distribution, n::Int)
    reward = 0
    sample_trace = []
    state = rand(start_state_distribution)
    for _ in 1:n
        isa(state, NonTerminal) || break
        next_distribution = transition_reward(mrp, state)
        (next_state, reward) = rand(next_distribution)
        push!(sample_trace, TransitionStep(state, next_state, reward))
        state = next_state
    end
    return sample_trace
end

function transition(mrp::MarkovRewardProcess, state::NonTerminal)
    function next_state(d=transition_reward(mrp, state))
        (next_s, _) = rand(d(mrp, state))
        return next_s
    end
    return SampledDistribution(next_state)
end

struct FiniteMarkovRewardProcess{S} <: MarkovRewardProcess{S}
    transition_reward_map::Dict{NonTerminal{S}, FiniteDistribution} 
    reward_function_vec::Vector
    fmp::FiniteMarkovProcess
end

function FiniteMarkovRewardProcess(transition_reward_map::Dict{S,T}) where {S, T<:FiniteDistribution{Tuple{S, Float64}}}
    transition_map::Dict{S, FiniteDistribution{S}} = Dict()
    for (state, trans) in transition_reward_map
        probabilities = DefaultDict{S, Float64}(0.0)
        for ((next_state, _), probability) in trans.dict
            probabilities[next_state] += probability
        end
        transition_map[state] = LabeledCategorical(Dict(probabilities))
    end
    fmp = FiniteMarkovProcess(transition_map)
    nt = collect(keys(transition_reward_map))
    transition_rm = Dict(
        NonTerminal(s) => LabeledCategorical(
            Dict(
                ((s1 in nt ? NonTerminal(s1) : Terminal(s1)), r) => p for ((s1, r), p) in v.dict
            ) 
        ) for (s, v) in transition_reward_map
    )


    reward_function_vec = [
        sum(p * r for ((_, r), p) in transition_rm[NonTerminal(s)].dict) for s in fmp.non_terminals
    ]

    return FiniteMarkovRewardProcess{S}(transition_rm, reward_function_vec, fmp)
end

transition_reward(fmrp::FiniteMarkovRewardProcess, state::NonTerminal) = fmrp.transition_reward_map[state]


function get_value_function_vec(fmrp::FiniteMarkovRewardProcess, gamma::Float64)
    return (I(length(fmrp.fmp.non_terminals)) - gamma * get_transition_matrix(fmrp.fmp)) \ fmrp.reward_function_vec
end

end
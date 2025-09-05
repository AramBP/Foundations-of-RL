using Random: SamplerUnion
using Distributions
include("mp.jl")

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

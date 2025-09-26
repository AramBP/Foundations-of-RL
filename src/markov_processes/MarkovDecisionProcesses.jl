module MarkovDecisionProcesses

using Distributions

using RL.MarkovProcesses: NonTerminal, State, Terminal
using RL.Policies: Policy, act
using RL.DistributionsExt: apply, FiniteDistribution, SampledDistribution, LabeledCategorical
using RL.MarkovRewardProcesses: MarkovRewardProcess

export MarkovDecisionProcess, actions, step, transition_reward, ImpliedMRP, simulate_actions, FiniteMarkovDecisionProcess

struct TransitionStep{S, A}
    state::NonTerminal{S}
    action::A
    next_state::T where {T <: State{S}}
    reward::Float64
end

abstract type MarkovDecisionProcess{S, A} end

# returns iterable of type A
actions(mdp::MarkovDecisionProcess, state::NonTerminal) = 
    error("function actions not defined for $(typeof(mdp))")

# specifies a distribution of pairs of next state and reward
step(mdp::MarkovDecisionProcess, state::NonTerminal, action) =
    error("function step not defined for $(typeof(mdp))")

# implied MRP of the passed in MDP and fixed policy
struct ImpliedMRP{S} <: MarkovRewardProcess{S}
    mdp::MarkovDecisionProcess
    policy::Policy
end

function transition_reward(mrp::ImpliedMRP, state::NonTerminal)
    actions = act(policy, state)
    return apply(actions, a -> step(mrp.mdp, state, a))
end

function simulate_actions(mdp::MarkovDecisionProcess, start_states::Distribution, policy::Policy, n::Int)
    sample_trace = []
    state = rand(start_states)
    for _ in 1:n
        isa(state, NonTerminal) || break

        action_distribution = act(policy, state)
        action = rand(action_distribution)

        next_state_distribition = step(mdp, state, action)
        next_state, reward = rand(next_state_distribition)

        push!(sample_trace, TransitionStep(state, action, next_state, reward))
        state = next_state
    end
    return sample_trace
end

struct FiniteMarkovDecisionProcess{S, A} <: MarkovDecisionProcess{S, A}
    mapping::Dict{NonTerminal{S}, Dict{A, FiniteDistribution}}
    non_terminals::Vector{S}
end

function FiniteMarkovDecisionProcess(transition_mapping::Dict{S, Dict{A, T}}) where {S, A, T <: FiniteDistribution{Tuple{S, Float64}}}
    non_terminals_transition_mapping = collect(keys(transition_mapping))
    mapping = Dict(NonTerminal(s) => {a => LabeledCategorical(
        Dict(((s1 in non_terminals_transition_mapping) ? NonTerminal(s1) : Terminal(s1), r) => p) for ((s1, r), p) in v
    ) for (a, v) in d} for (s, d) in transition_mapping)
    non_terminals_mapping = collect(keys(mapping))
    return FiniteMarkovDecisionProcess(mapping, non_terminals_mapping)
end

function Base.show(io::IO, fmdp::FiniteMarkovDecisionProcess)
    display = ""
    for (s, d) in fmdp.mapping
        display *= "From State $(s.state): \n"
        for (a, d1) in d
            display *= "  With Action $(a): \n"
            for ((s1, r), p) in d1 
                opt = isa(s1, Terminal) ? "Terminal " : ""
                display *= "  To [$(opt) State $(s1.state) and Reward $(round(r; digits = 3))] with Probability $(round(p; digits = 3))" 
            end
        end
    end
    print(io, display)
end

end
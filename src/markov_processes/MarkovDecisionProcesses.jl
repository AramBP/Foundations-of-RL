module MarkovDecisionProcesses

using Distributions

using RL.MarkovProcesses: NonTerminal
using RL.Policies: Policy, act
using RL.DistributionsExt: apply
using RL.MarkovRewardProcesses: MarkovRewardProcess

export MarkovDecisionProcess, actions, step, transition_reward, ImpliedMRP, simulate_actions

struct TransitionStep{S, A}
    state::NonTerminal{S}
    action::A
    next_state::S
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
end

end
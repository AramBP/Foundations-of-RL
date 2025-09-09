module Policies

using RL.DistributionsExt: Constant, Choose

export Policy, DeterministicPolicy, UniformPolicy, act, action_for

abstract type Policy{S, A} end
abstract type DeterministicPolicy{S, A} <: Policy{S, A} end
abstract type UniformPolicy{S, A} <: Policy{S, A} end

act(policy::Policy) = error("The function act is not defined for $(typeof(policy))")
action_for(policy::Policy, s::NonTerminal) = error("The function action_for is not defined for $(typeof(policy))")

act(policy::DeterministicPolicy, state::NonTerminal) = Constant(policy.action_for(state.state))
act(policy::UniformPolicy, state::NonTerminal) = Choose(policy.valid_actions(state.state))

end
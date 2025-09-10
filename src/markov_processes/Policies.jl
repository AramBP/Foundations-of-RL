module Policies

using RL.DistributionsExt: Constant, Choose
using RL.MarkovProcesses: NonTerminal

export Policy, DeterministicPolicy, UniformPolicy, act, action_for

abstract type Policy{S, A} end
abstract type DeterministicPolicy{S, A} <: Policy{S, A} end
abstract type UniformPolicy{S, A} <: Policy{S, A} end

act(policy::Policy, s::NonTerminal) = error("The function act is not defined for $(typeof(policy))")
action_for(policy::Policy, s::NonTerminal) = error("The function action_for is not defined for $(typeof(policy))")

act(policy::DeterministicPolicy, state::NonTerminal) = Constant(action_for(policy, state.state))
act(policy::UniformPolicy, state::NonTerminal) = Choose(policy.valid_actions(state.state))

end
module Policies

using RL.DistributionsExt: Constant, Choose

export Policy, DeterministicPolicy, UniformPolicy, act

abstract type Policy{S, A} end

act(policy::Policy) = error("The function act is not defined for $(typeof(policy))")


struct DeterministicPolicy{S, A} <: Policy{S, A}
    action_for::Function # f: S -> A
end

act(policy::DeterministicPolicy, state::NonTerminal) = Constant(policy.action_for(state.state))

struct UniformPolicy{S, A} <: Policy{S, A}
    valid_actions::Function # f: S -> Array{A}
end

act(policy::UniformPolicy, state::NonTerminal) = Choose(policy.valid_actions(state.state))

end
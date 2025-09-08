module Policies

export Policy, DeterministicPolicy, act

abstract type Policy{S, A} end

act(policy::Policy) = error("The function act is not defined for $(typeof(policy))")


struct DeterministicPolicy{S, A} <: Policy{S, A}
    action_for::Function
end

#act(policy::DeterministicPolicy, state::NonTerminal) = Constant(policy.action_for(state.state))

end
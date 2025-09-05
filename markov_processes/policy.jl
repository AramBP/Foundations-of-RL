using FunctionWrappers
abstract type Policy{S, A} end

act(policy::Policy) = error("The function act is not defined for $(typeof(policy))")


struct DeterministicPolicy{S, A} <: Policy{S, A}
    action_for::FunctionWrapper{S, A}
end


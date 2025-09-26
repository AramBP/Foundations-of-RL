module Policies

using RL.DistributionsExt: Constant, Choose, FiniteDistribution
using RL.MarkovProcesses: NonTerminal

export Policy, DeterministicPolicy, UniformPolicy, act, action_for, FiniteDeterministicPolicy

abstract type Policy{S, A} end
act(policy::Policy, s::NonTerminal) = error("The function act is not defined for $(typeof(policy))")
action_for(policy::Policy, s::NonTerminal) = error("The function action_for is not defined for $(typeof(policy))")

abstract type DeterministicPolicy{S, A} <: Policy{S, A} end
abstract type UniformPolicy{S, A} <: Policy{S, A} end
act(policy::DeterministicPolicy, state::NonTerminal) = Constant(action_for(policy, state.state))
act(policy::UniformPolicy, state::NonTerminal) = Choose(policy.valid_actions(state.state))

struct FinitePolicy{S, A} <: Policy{S, A}
    policy_map::Dict{S, FiniteDistribution{A}}
end
act(policy::FinitePolicy, state::NonTerminal) = policy.policy_map[state.state]

struct FiniteDeterministicPolicy{S, A} <: Policy{S, A}
    action_for::Dict{S, A}
    fp::FinitePolicy
end

function FiniteDeterministicPolicy(action_for::Dict{S, A}) where {S, A}
    fp = FinitePolicy(Dict(s => Constant(a) for (s, a) in mapping))
    return FiniteDeterministicPolicy(action_for, fp)
end


function Base.show(io::IO, fp::FinitePolicy) 
    display = ""
    for (s, d) in fp.policy_map
        display *= "For State $(s): \n"
        for (a, p) in d.dict
            display *= "  Do Action $(a) with Probability $(round(p; digits = 3))\n"
        end
    end
    print(io, display)
end



end
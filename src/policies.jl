abstract type Policy{S, A} end
abstract type DeterministicPolicy{S, A} <: Policy{S, A} end
abstract type UniformPolicy{S, A} <: Policy{S, A} end
act(policy::DeterministicPolicy, state::NonTerminal) = Constant(action_for(policy, state.state))
act(policy::UniformPolicy, state::NonTerminal) = Choose(policy.valid_actions(state.state))

struct FinitePolicy{S, A} <: Policy{S, A}
    policy_map::Dict{S, <:FiniteDistribution{A}}
end
act(policy::FinitePolicy, state::NonTerminal) = policy.policy_map[state.state]

struct FiniteDeterministicPolicy{S, A} <: Policy{S, A}
    action_for::Dict{S, A}
    fp::FinitePolicy
end

function FiniteDeterministicPolicy(action_for::Dict{S, A}) where {S, A}
    fp = FinitePolicy(Dict(s => Constant(a) for (s, a) in action_for))
    return FiniteDeterministicPolicy(action_for, fp)
end

act(policy::FiniteDeterministicPolicy, state::NonTerminal) = act(policy.fp, state)


function Base.show(io::IO, fp::FinitePolicy) 
    display = ""
    for (s, d) in fp.policy_map
        display *= "For State $(s): \n"
        for (a, p) in d
            display *= "  Do Action $(a) with Probability $(round(p; digits = 3))\n"
        end
    end
    print(io, display)
end

function Base.show(io::IO, fdp::FiniteDeterministicPolicy) 
    display = ""
    for (s, a) in fdp.action_for
        display *= "For State $(s): do Action $(a)\n"
    end
    print(io, display)
end

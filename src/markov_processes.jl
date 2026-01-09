"""
    This file includes different abstract types and structs that describe markov processes and markov reward processes.
    It also includes functions to interact with them.
"""
# State types
abstract type State{S} end

struct Terminal{S} <: State{S}
    state::S
end

struct NonTerminal{S} <: State{S}
    state::S
end

function on_non_terminal(state::State, f::Function, default)
    if isa(state, NonTerminal)
        return f(state)
    else
        return default
    end
end

################# Markov Processes #################

abstract type MarkovProcess{S} end

transition(mp::MarkovProcess, state::NonTerminal{S}) where {S} = error("transistion function not specified for $(typeof(mp)), state: $(state)")

function simulate(mp::MarkovProcess, start_state_distribution::Distribution, n::Int)
    sample_trace = []
    state = rand(start_state_distribution)
    push!(sample_trace, state)
    for _ in 1:n
        isa(state, NonTerminal) || break
        state = rand(transition(mp, state))
        push!(sample_trace, state)
    end
    return sample_trace
end


# Finite Markov Process

struct FiniteMarkovProcess{S} <: MarkovProcess{S}
    non_terminals::Vector{S}
    transition_map::Dict
end

function FiniteMarkovProcess(transition_map::Dict)
    non_terminals = collect(keys(transition_map))
    transition = Dict(
        NonTerminal(s) => LabeledCategorical(
            Dict(
                (s1 in non_terminals ? NonTerminal(s1) : Terminal(s1)) => p for (s1, p) in v.dict)
        ) for (s, v) in transition_map)
    return FiniteMarkovProcess(non_terminals, transition)
end

function Base.show(io::IO, fmp::FiniteMarkovProcess)
    display = ""
    for (s, d) in fmp.transition_map
        display *= "From state $(s.state):\n"
        for (s1, p) in d.dict
            opt = isa(s1, Terminal) ? "Terminal State" : "State"
            display *= "   To $(opt) $(s1.state) with Probability $(round(p;digits=3))\n"
        end
    end
    print(io, display)
end

transition(fmp::FiniteMarkovProcess, state::NonTerminal) = fmp.transition_map[state]
function get_transition_matrix(fmp::FiniteMarkovProcess)
    size = length(fmp.non_terminals)
    mat = Matrix{Float64}(undef, size, size)
    for (i, s1) in enumerate(fmp.non_terminals)
        for(j, s2) in enumerate(fmp.non_terminals)
            mat[i,j] = probability(transition(fmp, NonTerminal(s1)), NonTerminal(s2))
        end
    end
    return mat
end

function get_stationairy_distribution(fmp::FiniteMarkovProcess)
    (eig_vals, eig_vecs) = eigen(Transpose(get_transition_matrix(fmp)))

    first_unit_eigen_val = filter((x) -> abs(x - 1) < 1e-8, eig_vals)[1]
    index_of_first_unit_eig_val = findfirst(==(first_unit_eigen_val), eig_vals)
    eig_vec_of_unit_eig_val = real.(eig_vecs[:, index_of_first_unit_eig_val])
    return LabeledCategorical(
        Dict(
            fmp.non_terminals[i] => ev for (i, ev) in enumerate(eig_vec_of_unit_eig_val ./ sum(eig_vec_of_unit_eig_val))
        )
    )
end

################# Markov Reward Processes #################

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

struct FiniteMarkovRewardProcess{S} <: MarkovRewardProcess{S}
    transition_reward_map::Dict{NonTerminal, FiniteDistribution} 
    reward_function_vec::Vector
    fmp::FiniteMarkovProcess
end

function FiniteMarkovRewardProcess(transition_reward_map::Dict{S,T}) where {S, T}
    transition_map::Dict{S, FiniteDistribution{S}} = Dict()
    for (state, trans) in transition_reward_map
        probabilities = DefaultDict{S, Float64}(0.0)
        for ((next_state, _), probability) in trans.dict
            probabilities[next_state] += probability
        end
        transition_map[state] = LabeledCategorical(Dict(probabilities))
    end
    fmp = FiniteMarkovProcess(transition_map)
    nt = collect(keys(transition_reward_map))
    transition_rm = Dict(
        NonTerminal(s) => LabeledCategorical(
            Dict(
                ((s1 in nt ? NonTerminal(s1) : Terminal(s1)), r) => p for ((s1, r), p) in v.dict
            ) 
        ) for (s, v) in transition_reward_map
    )


    reward_function_vec = [
        sum(p * r for ((_, r), p) in transition_rm[NonTerminal(s)].dict) for s in fmp.non_terminals
    ]

    return FiniteMarkovRewardProcess{S}(transition_rm, reward_function_vec, fmp)
end

transition_reward(fmrp::FiniteMarkovRewardProcess, state::NonTerminal) = fmrp.transition_reward_map[state]


function get_value_function_vec(fmrp::FiniteMarkovRewardProcess, gamma::Float64)
    return (I(length(fmrp.fmp.non_terminals)) - gamma * get_transition_matrix(fmrp.fmp)) \ fmrp.reward_function_vec
end

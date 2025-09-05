using Distributions, LinearAlgebra, Statistics
include("../distributions.jl")

# State types
abstract type State{S} end

struct Terminal{S} <: State{S}
    state::S
end

struct NonTerminal{S} <: State{S}
    state::S
end

# Markov Process
abstract type MarkovProcess{S} end

transition(mp::MarkovProcess{S}, state::NonTerminal{S}) where {S} =
    error("transistion function not specified for $(typeof(mp)), passed in state: $(state.state)")

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
            Dict((s1 in non_terminals ? NonTerminal(s1) : Terminal(s1)) => p for (s1, p) in v.dict)
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


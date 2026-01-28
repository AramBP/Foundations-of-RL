struct TransitionStepDecision{S, A}
    state::NonTerminal{S}
    action::A
    next_state::T where {T <: State{S}}
    reward::Float64
end

abstract type MarkovDecisionProcess{S, A} end

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

        push!(sample_trace, TransitionStepDecision(state, action, next_state, reward))
        state = next_state
    end
    return sample_trace
end

struct FiniteMarkovDecisionProcess{S, A} <: MarkovDecisionProcess{S, A}
    mapping::Dict{NonTerminal{S}, Dict{A, LabeledCategorical}}
    non_terminals::Vector{NonTerminal{S}}
end

function FiniteMarkovDecisionProcess(transition_mapping::Dict)
    non_terminals_transition_mapping = collect(keys(transition_mapping))
    mapping = Dict(
        NonTerminal(s) => Dict{typeof(first(keys(d))), LabeledCategorical}(a => 
            LabeledCategorical( 
                Dict(((s1 in non_terminals_transition_mapping ? NonTerminal(s1) : Terminal(s1)), r) => p for ((s1, r), p) in v.dict)
            ) for (a, v) in d) 
        for (s, d) in transition_mapping)
    non_terminals_mapping = collect(keys(mapping))
    return FiniteMarkovDecisionProcess(mapping, non_terminals_mapping)
end

function step(fmdp::FiniteMarkovDecisionProcess, state::NonTerminal, action::A) where {A}
    action_map = fmdp.mapping[state]
    return action_map
end

function actions(fmdp::FiniteMarkovDecisionProcess, state::NonTerminal)
    return collect(keys(fmdp.mapping[state]))
end

function Base.show(io::IO, fmdp::FiniteMarkovDecisionProcess)
    display = ""
    for (s, d) in fmdp.mapping
        display *= "From State $(s.state): \n"
        for (a, d1) in d
            display *= "  With Action $(a): \n"
            for ((s1, r), p) in d1.dict
                opt = isa(s1, Terminal) ? "Terminal " : ""
                display *= "  To [$(opt) State $(s1.state) and Reward $(round(r; digits = 3))] with Probability $(round(p; digits = 3)) \n" 
            end
        end
    end
    print(io, display)
end

function apply_finite_policy(fmdp::FiniteMarkovDecisionProcess, fp::FinitePolicy)
    transition_mapping::Dict = Dict()
    for state in collect(keys(fmdp.mapping))
        action_map = fmdp.mapping[state]
        outcomes = DefaultDict{Tuple, Float64}(0.0)
        actions = act(fp, state)
        if isa(actions, LabeledCategorical)
            for (action, p_action) in actions.dict
                for ((s1, r), p) in action_map[action].dict
                    outcomes[(s1.state, r)] += p_action * p
                end
            end
        elseif isa(actions, Choose)
            options = actions.options
            p_action = 1/length(options)
            for action in options
                for ((s1, r), p) in action_map[action].dict
                    outcomes[(s1.state, r)] += p_action * p
                end
            end            
        else
            for ((s1, r), p) in action_map[actions.value].dict
                outcomes[(s1.state, r)] += p
            end
        end
        transition_mapping[state.state] = LabeledCategorical(Dict(outcomes))
    end
    return FiniteMarkovRewardProcess(transition_mapping)
end


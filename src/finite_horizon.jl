Base.@kwdef struct WithTime{S}
    state::S
    time::Int = 0
end


function finite_horizon_MRP(process::FiniteMarkovRewardProcess, limit::Int)
    transition_map = Dict()
    for time in 0:(limit - 1)
        for s in process.fmp.non_terminals
            s = NonTerminal(s)
            result = transition_reward(process, s)
            s_time = WithTime(s.state, time)
            transition_map[s_time] = map(result, s_r -> (WithTime(s_r[1].state, time+1), s_r[2]))
        end
    end
    return FiniteMarkovRewardProcess(transition_map)
end

function unwrap_finite_horizon_MRP(process::FiniteMarkovRewardProcess)
    time(x::WithTime) = x.time

    function single_without_time(state_reward::Tuple)
        if isa(state_reward[1], NonTerminal)
            return (NonTerminal(state_reward[1].state.state), state_reward[2])
        else
            return (Terminal(state_reward[1].state.state), state_reward[2])
        end
    end

    function without_time(arg::FiniteDistribution)
        map(arg, single_without_time)
    end

    return [
        Dict(
            NonTerminal(s.state) => without_time(
                transition_reward(process, NonTerminal(s))
            )
            for s in states
        )
        for states in groupby(
            x -> x.time, sort(process.fmp.non_terminals, by = x -> x.time)
        )
    ]
end

function evaluate(steps, gamma)
    v = []
    for step in reverse(steps)
        push!(v, 
            Dict(
                s => expectation(res, s_r -> length(v) > 0 ? s_r[2] + gamma * extended_vf(v[end], NonTerminal(s_r[1])) : s_r[2])
                for (s, res) in collect(step)
            )
        )
    end
    return reverse(v)
end

function finite_horizon_MDP(process::FiniteMarkovDecisionProcess, limit::Int)
    mapping = Dict()
    for time in 0:(limit - 1)
        for s in process.non_terminals
            s_time = WithTime(s.state, time)
            mapping[s_time] = Dict(
                a => map(result, s_r -> (WithTime(s_r[1].state, time+1), s_r[2]))
                for (a, result) in collect(process.mapping[s])
            )
        end
    end
    return FiniteMarkovDecisionProcess(mapping)
end

function unwrap_finite_horizon_MDP(process::FiniteMarkovDecisionProcess)
    time(x::WithTime) = x.time

    function single_without_time(state_reward::Tuple{State{WithTime{S}}, Float64}) where {S}
        if isa(state_reward[1], NonTerminal)
            return (NonTerminal(state_reward[1].state.state), state_reward[2])
        else
            return (Terminal(state_reward[1].state.state), state_reward[2])
        end
    end

    function without_time(arg)
        return Dict(a => map(sr_distr, single_without_time) for (a, sr_distr) in collect(arg))
    end

    return [
        Dict(
            NonTerminal(s.state) => without_time(
                process.mapping[NonTerminal(s)]
            )
            for s in states
        )
        for states in groupby(
            x -> x.time, sort([nt.state for nt in process.non_terminals], by = x -> x.time)
        )
    ]
end

function optimal_vf_and_policy(steps, gamma)
    v_p = []
    for step in reverse(steps)
        this_v = Dict()
        this_a = Dict()
        for (s, actions_map) in collect(step)
            action_values = [
                (expectation(res, s_r -> length(v_p) > 0 ? s_r[2] + gamma * (extended_vf(v_p[end][1], NonTerminal(s_r[1]))) : s_r[2]), a) 
                for (a, res) in collect(actions_map)
            ]
            (v_star, a_star) = argmax(first, sort(action_values, by= x -> x[2]))
            this_v[s] = v_star
            this_a[s.state] = a_star
        end
        push!(v_p, (this_v, FiniteDeterministicPolicy(this_a)))
    end
    return reverse(v_p)
end
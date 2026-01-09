DEFAULT_TOLERANCE = 1e-5

almost_equal_vectors(v1, v2; tolerance = DEFAULT_TOLERANCE) = maximum((abs.(v1 .- v2))) < tolerance

function evaluate_mrp_result(fmrp::FiniteMarkovRewardProcess, gamma::Float64)
    non_terminal_states = fmrp.fmp.non_terminals
    update(v) = fmrp.reward_function_vec + gamma * (get_transition_matrix(fmrp.fmp) * v)
    v_0 = zeros(length(non_terminal_states))
    v_star = converged(Iter(update, almost_equal_vectors, v_0))
    return Dict(s => v_star[i] for (i, s) in enumerate(non_terminal_states))
end

function extended_vf(v::Dict, s::State)
    non_terminal_vf(st) = v[st.state]
    on_non_terminal(s, non_terminal_vf, 0.0)
end

function greedy_policy_from_vf(fmdp::FiniteMarkovDecisionProcess, vf, gamma::Float64)
    greedy_policy_dict = Dict()
    non_terminal_states = fmdp.non_terminals
    for s in non_terminal_states
       q_values = ((a, expectation(fmdp.mapping[s][a], state_reward -> state_reward[2] + gamma*extended_vf(vf, state_reward[1]))) for a in actions(fmdp, s)) 
       greedy_policy_dict[s.state] = argmax(item -> item[2], q_values)[1]
    end
    return FiniteDeterministicPolicy(greedy_policy_dict)
end

almost_equal_vf_pis(x1::Tuple, x2::Tuple; tolerance = DEFAULT_TOLERANCE) = maximum(abs(x1[1][s] - x2[1][s]) for s in collect(keys(x1[1]))) < tolerance

function policy_iteration_result(fmdp::FiniteMarkovDecisionProcess, gamma::Float64; matrix_method_for_mrp_eval = false)
    function update(vf_policy::Tuple)
        (_, pi) = vf_policy
        fmrp::FiniteMarkovRewardProcess = isa(pi, FinitePolicy) ? apply_finite_policy(fmdp, pi) : apply_finite_policy(fmdp, pi.fp)
        non_terminal_states = fmrp.fmp.non_terminals
        policy_vf = matrix_method_for_mrp_eval ? Dict(non_terminal_states[i] => v for (i,v) in enumerate(get_value_function_vec(fmrp, gamma))) : evaluate_mrp_result(fmrp, gamma)
        improved_pi = greedy_policy_from_vf(fmdp, policy_vf, gamma)
        return (policy_vf, improved_pi)
    end

    v_0 = Dict(s.state => 0.0 for s in fmdp.non_terminals)
    pi_0 = FinitePolicy(Dict(s.state => Choose(actions(fmdp, s)) for s in fmdp.non_terminals))
    converged_values = converged(Iter(update, almost_equal_vf_pis, (v_0, pi_0)))
    return converged_values
end

almost_equal_vfs(v1, v2; tolerance = DEFAULT_TOLERANCE) = maximum(abs(v1[s] - v2[s]) for s in collect(keys(v1))) < tolerance

function value_iteration_result(fmdp::FiniteMarkovDecisionProcess, gamma::Float64)
    function update(v)
        return Dict(s => 
            maximum(expectation(fmdp.mapping[NonTerminal(s)][a], state_reward -> state_reward[2] + gamma * extended_vf(v, state_reward[1])) for a in actions(fmdp, NonTerminal(s))) 
        for s in collect(keys(v)))
    end
    v_0 = Dict(s.state => 0.0 for s in fmdp.non_terminals)
    opt_vf = converged(Iter(update, almost_equal_vfs, v_0))
    opt_policy = greedy_policy_from_vf(fmdp, opt_vf, gamma)
    return (opt_vf, opt_policy)
end

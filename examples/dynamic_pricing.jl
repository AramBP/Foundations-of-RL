include("../src/core.jl")

struct ClearancePricingMDP
    initial_inventory::Int
    time_steps::Int
    price_lambda_pairs
    single_step_mdp::FiniteMarkovDecisionProcess
    mdp::FiniteMarkovDecisionProcess
end

function ClearancePricingMDP(initial_inventory, time_steps, price_lambda_pairs)
    distrs = [Poisson(l) for (_, l) in price_lambda_pairs]
    prices = [p for (p, _) in price_lambda_pairs]
    single_step_mdp = FiniteMarkovDecisionProcess(
        Dict(s => Dict(i => LabeledCategorical(
            Dict((s-k, prices[i]*k) => (k < s ? pdf(distrs[i], k) : 1 - cdf(distrs[i], s-1))
        for k in 0:s))
        for i in 1:length(prices))
    for s in 0:initial_inventory))
    mdp = finite_horizon_MDP(single_step_mdp, time_steps)
    return ClearancePricingMDP(initial_inventory, time_steps, price_lambda_pairs, single_step_mdp, mdp)
end

function get_vf_for_policy(cp_mdp::ClearancePricingMDP, policy::FinitePolicy{WithTime{Int}, Int})
    mrp = apply_finite_policy(cp_mdp.mdp, policy)
    return evaluate(unwrap_finite_horizon_MRP(mrp), 1.0)
end

get_optimal_vf_and_policy(cp_mdp::ClearancePricingMDP) = return optimal_vf_and_policy(unwrap_finite_horizon_MDP(cp_mdp.mdp), 1.0)

initial_inventory = 12
time_steps = 8
price_lambda_pairs = [(1.0, 0.5), (0.7, 1.0), (0.5, 1.5), (0.3, 2.5)]
cp = ClearancePricingMDP(initial_inventory, time_steps, price_lambda_pairs)

function policy_func(x::Int)
    if x < 2
        return 1
    elseif x < 5
        return 2
    elseif x < 8
        return 3
    else
        return 4
    end
end 

stationairy_policy = FiniteDeterministicPolicy(Dict(s => policy_func(s) for s in 0:initial_inventory))
single_step_mrp = apply_finite_policy(cp.single_step_mdp, stationairy_policy.fp)
vf_for_policy = evaluate(unwrap_finite_horizon_MRP(finite_horizon_MRP(single_step_mrp, time_steps)), 1.)

prices = [[price_lambda_pairs[act(policy, NonTerminal(s)).value][1] for s in 0:initial_inventory] for (_, policy) in get_optimal_vf_and_policy(cp)]

plot_matrix = reduce(hcat, prices)
heatmap(plot_matrix,
    title = "Optimal Policy Heatmap",
    xlabel = "Time Steps",
    ylabel = "Inventory",
    c = :viridis,           # Match the Python color palette
    aspect_ratio = 0.5,     # Adjust this number to make it less wide (e.g., 0.5 or 0.7)
    xlims = (0.5, 8.5),     # Tighten the x-axis
    ylims = (0.5, 13.5),    # Tighten the y-axis
    size = (500, 600)       # Explicitly set the window size (width, height)
)
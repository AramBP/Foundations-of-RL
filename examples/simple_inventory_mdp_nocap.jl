using RL.MarkovDecisionProcesses: MarkovDecisionProcess, step, actions, simulate_actions
using RL.DistributionsExt: SampledDistribution, Constant
using Distributions: Poisson
using PrettyPrint

include("simple_inventory_policies.jl")

inventory_position(is::InventoryState) = is.on_hand + is.on_order

struct SimpleInventoryMDPNoCap <: MarkovDecisionProcess{InventoryState, Int}
    poisson_lambda::Float64
    holding_cost::Float64
    stockout_cost::Float64
end

function RL.MarkovDecisionProcesses.step(
    si_mdp::SimpleInventoryMDPNoCap, 
    state::NonTerminal{InventoryState},
    order::Int)

    function sample_next_state_reward(st=state, or=order)
        demand_sample = rand(Poisson(si_mdp.poisson_lambda))
        ip = inventory_position(st.state)
        next_state = InventoryState(max(ip - demand_sample, 0), or)
        reward = si_mdp.holding_cost*st.state.on_hand - si_mdp.stockout_cost*max(demand_sample - ip, 0)
        return (NonTerminal(next_state), reward)
    end
    return SampledDistribution(; sampler = sample_next_state_reward)
end

# Return an infinite generator of non_negative integers to represent the fact that the action space is infinite
RL.MarkovDecisionProcesses.actions(si_mdp::SimpleInventoryMDPNoCap, state::NonTerminal{InventoryState}) = 
    Main.Iterators.countfrom(0, 1)

si_mdp = SimpleInventoryMDPNoCap(1.0, 1.0, 10.0)
si_sp = SimpleInventoryStochasticPolicy(8.0)
si_dp = SimpleInventoryDeterministicPolicy(8.0)

sim1 = [simulate_actions(si_mdp, Constant(NonTerminal(InventoryState(0, 0))), si_sp, 10) for _ in 1:5]
sim2 = [simulate_actions(si_mdp, Constant(NonTerminal(InventoryState(0, 0))), si_dp, 10) for _ in 1:5]
pprintln(sim2[1])
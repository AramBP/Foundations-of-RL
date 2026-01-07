include("../src/core.jl")

struct InventoryState
    on_hand::Int
    on_order::Int 
end

inventory_position(is::InventoryState) = is.on_order + is.on_hand

struct SimpleInventoryMRP <: MarkovRewardProcess{InventoryState}
    capacity::Int
    poisson_lambda::Float64
    holding_cost::Float64
    stockout_cost::Float64
end

function transition_reward(mrp::SimpleInventoryMRP, state::NonTerminal{InventoryState})
    function sample_next_state_reward(st=state)
        demand_sample = rand(Poisson(mrp.poisson_lambda))
        ip = inventory_position(st.state)
        next_state = InventoryState(
            max(ip - demand_sample, 0),
            max(mrp.capacity - ip, 0)
        )
        reward = -mrp.holding_cost*st.state.on_hand - mrp.stockout_cost*max(demand_sample - ip, 0)
        return (NonTerminal(next_state), reward)
    end
    return SampledDistribution(; sampler=sample_next_state_reward)
end

si_mrp = SimpleInventoryMRP(2, 1.0, 1.0, 10.0)
sample_traces = [simulate(si_mrp, Constant(NonTerminal(InventoryState(0,0))), 10)]
pprintln(sample_traces)

include("../src/core.jl")

function unit_sigmoid_func(x; a::Float64)
    x != 0 || (x = 1e-8)
    return 1 / (1 + (1/x - 1)^a)
end

struct SPState
    num_up_moves::Int
    num_down_moves::Int
end

struct StockPrice <: MarkovProcess{SPState}
    alpha3::Float64
end

function up_prob(sp::StockPrice, state::SPState)
    total = state.num_up_moves + state.num_down_moves
    (total == 0) && return 0.5
    return unit_sigmoid_func(state.num_down_moves/total; a=sp.alpha3)
end

function transition(sp::StockPrice, state::NonTerminal{SPState})
    up_p = up_prob(sp, state.state)
    return LabeledCategorical(
        Dict(
            NonTerminal(SPState(state.state.num_up_moves + 1, state.state.num_down_moves)) => up_p,
            NonTerminal(SPState(state.state.num_up_moves, state.state.num_down_moves+1)) => 1-up_p
        )
    )
end

function price_traces(start_price::Int, alpha3::Float64, time_steps::Int, num_trace::Int)
    mp = StockPrice(alpha3)
    start_state_dist = Constant(NonTerminal(SPState(0,0)))
    traces = []
    for _ in 1:num_trace
        simulate_sp = simulate(mp, start_state_dist, time_steps)
        trace = [start_price + s.state.num_up_moves - s.state.num_down_moves for s in simulate_sp]
        push!(traces, trace)
    end
    return traces
end


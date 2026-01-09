struct Iter{X}
    step::Function
    done::Function
    start_state::X
end

Base.iterate(it::Iter) = (it.start_state, it.start_state)

function Base.iterate(it::Iter, prev_state)
    next_state = it.step(prev_state)
    if it.done(prev_state, next_state)
        return nothing
    else
        return (next_state, next_state)
    end
end
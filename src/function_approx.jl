abstract type FunctionApprox end

function update(f::FunctionApprox, xy_vals_seq)
    deriv_func(x_seq, y_seq) = evaluate(f, x_seq) - y_seq
    return update_with_gradient(
        objective_gradient(f, xy_vals_seq, deriv_func)
    )
end

iterate_updates(f::FunctionApprox, xy_seq) = accumulate((fa, xy) -> update(fa, xy), xy_seq, init = f)

function rmse(f::FunctionApprox, xy_vals_seq)
    x_seq = first.(xy_vals_seq)
    y_seq = last.(xy_vals_seq)
    errors = evaluate(f, x_seq) .- y_seq
    return sqrt(mean(errors .* errors))
end

(f::FunctionApprox)(x) = evaluate(f, [x])
argmax(f::FunctionApprox, xs) = (args = collect(xs); args[Base.argmax(evaluate(f, args))])

struct Gradient
    function_approx::FunctionApprox
end

function +(grad::Gradient, x)
    if isa(x, Gradient)
        return Gradient(grad.function_approx + x.function_approx)
    end
    return grad.function_approx + x
end

*(grad::Gradient, x::Float64) = Gradient(grad.function_approx * x)
zero(grad::Gradient) = Gradient(grad.function_approx * 0.0)
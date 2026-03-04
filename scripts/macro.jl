macro brm(x)
    dump(x)
end
# The example BRM model is supposed to work with three dataframes (dBMI, dpmean, dpsd), which IIUC, have to have the same numbers of rows in this example and could have been merged into one.
@brm model(dBMI, dpmean, dpsd) = begin 
    # BMI will be a model parameter
    BMI ~ Normal(dBMI.BMI_measured, 1) # equivalently: BMI ~ Normal(BMI_measured, 1) |> (data=dBMI) 
    # Age_first, Age_second are functions of data, and would be computed/updated exactly once
    Age_first, Age_second = ploynomial_expand(dpmean.Age; order=2) # equivalently: Age_first, Age_second = ploynomial_expand(Age; order=2) |> (data=dpmean) 
    # I'm assuming performance_mean is a function of data and model parameters - I think it kind of has to be
    performance_mean ~ 1 + Age_first * Treatment + Age_second + (1 + Treatment | Subject) + (1 + Age_first | Experimenter) |> (data=dpmean)
    # I'm assuming performance_sd is a function of data and model parameters - I think it kind of has to be
    log(performance_sd) ~ 1 + AGE * BMI + max(Age, BMI) + (1 + Age * BMI | Subject) |> (data=dpsd)
    # Peter didn't specify data here - but I think it would have to be specified? Or would it be added to the model via conditioning syntax?
    Performance ~ Normal(performance_mean, performance_sd) # |> (data=observations_df) as an example
    @defaults begin 
        gr(Subject, by=ClinicalGroup,
            Block1=>[Treatment, Age:BMI],
            Block2=>[Age, BMI]
        )
    end
end

# With a single dataframe, the model definition could look as follows (which would lower to the "obvious" syntax):
model = @brm begin 
    # BMI ~ Normal(BMI_measured, 1)
    Age_first, Age_second = ploynomial_expand(Age; order=2)
    performance_mean ~ 1 + Age_first * Treatment + Age_second + (1 + Treatment | Subject) + (1 + Age_first | Experimenter)
    log(performance_sd) ~ 1 + AGE * BMI + max(Age, BMI) + (1 + Age * BMI | Subject)
    Performance ~ Normal(performance_mean, performance_sd)
    @defaults begin 
        gr(Subject, by=ClinicalGroup,
            Block1=>[Treatment, Age:BMI],
            Block2=>[Age, BMI]
        )
    end
end


macro brm(x)
    esc(_brm(x))
end
begin
function ensurecols end
function maybedists end
# isxcall(x) = Meta.isexpr(x, :call)
isxcall(x, f) = Meta.isexpr(x, :call) && x.args[1] == f
fixcall(x) = x
fixcall(x::Expr) = if Meta.isexpr(x, :call)
    f = x.args[1]
    pargs = []
    args = []
    for arg in fixcall.(x.args[2:end])
        if Meta.isexpr(arg, :parameters)
            append!(pargs, arg.args)
        else
            push!(args, arg)
        end
    end
    if length(pargs) > 0
        Expr(x.head, f, Expr(:parameters, pargs...), args...)
    else
        Expr(x.head, f, args...)
    end
else
    Expr(x.head, fixcall.(x.args)...)
end
xensurecols(x::Expr) = if x.head == :call
    Expr(:call, ensurecols, x.args...) |> fixcall
else 
    dump(x)
    error("Don't know how to handle xensurecols($x)!")
end
    parse!(x::LineNumberNode; info) = x
parse!(x::Expr; info) = if x.head == :block
    Expr(:block, parse!.(x.args; info)...)
elseif x.head == :(=)
    lhs, rhs = x.args
    lhs = parse_assignment_lhs!(lhs; info)
    rhs = parse_assignment_rhs!(rhs; info)
    Expr(:(=), lhs, xensurecols(rhs))
elseif isxcall(x, :~)
    _, lhs, rhs = x.args
    lhs = parse_sampling_lhs!(lhs; info)
    rhs = parse_sampling_rhs!(rhs; info)
    if isxcall(rhs, ensurecols)
        :($lhs = $maybedists(;force=isdata($lhs))($(rhs.args[2:end]...)))
    else
        :($lhs = $maybedists(;force=isdata($lhs))($(rhs)))
    end
else
    dump(x)
    error("Don't know how to handle parse!($x)!")
end
parse_assignment_lhs!(x::Symbol; info) = (get!(info.alllocals, x, :local); x)
parse_assignment_lhs!(x::Expr; info) = begin 
    @assert Meta.isexpr(x, (:tuple, :vect))
    args = parse_assignment_lhs!.(x.args; info)
    Expr(x.head, args...)
end
parse_assignment_rhs!(x::Symbol; info) = (get!(info.alllocals, x, :nonlocal); x)
parse_assignment_rhs!(x::Expr; info) = if x.head == :call
    Expr(:call, x.args[1], parse_assignment_rhs!.(x.args[2:end]; info)...)
elseif Meta.isexpr(x, :parameters)
    x
else
    dump(x)
    error("Don't know how to handle parse_assignment_rhs!($x)!")
end
parse_sampling_lhs!(x::Symbol; info) = (get!(info.alllocals, x, :maybelocal); x)
parse_sampling_lhs!(x::Expr; info) = if x.head == :call
    Expr(:call, x.args[1], parse_sampling_lhs!.(x.args[2:end]; info)...)
else 
    @assert Meta.isexpr(x, (:tuple, :vect))
    args = parse_sampling_lhs!.(x.args; info)
    Expr(x.head, args...)
end
parse_sampling_rhs!(x::Number; info) = x
parse_sampling_rhs!(x::Symbol; info) = (get!(info.alllocals, x, :nonlocal); x)
parse_sampling_rhs!(x::Expr; info) = if x.head == :call
    if x.args[1] in (:+, :*, :|)
        Expr(:call, x.args[1], parse_sampling_rhs!.(x.args[2:end]; info)...)
    else
        xensurecols(parse_assignment_rhs!(x; info))
    end
elseif Meta.isexpr(x, :parameters)
    x
else
    dump(x)
    error("Don't know how to handle parse_sampling_rhs!($x)!")
end
using OrderedCollections
_brm(x::Expr) = begin 
    @assert x.head == :block
    alllocals = OrderedDict{Symbol,Symbol}()
    info = (;alllocals)
    x = parse!(x; info)
    nonlocals = [key for (key, value) in pairs(alllocals) if value == :nonlocal]
    maybelocals = [key for (key, value) in pairs(alllocals) if value == :maybelocal]
    locals = [key for (key, value) in pairs(alllocals) if value == :local]
    init = quote
        (;$(nonlocals...)) = data(__df__)
        (;$(maybelocals...)) = maybedata(__df__)
    end
    finalize = :(BRM(;$(keys(alllocals)...)))
    Expr(:(=), :(model(__df__)), Expr(:block, init, x.args..., finalize))
end
end
@macroexpand @brm begin 
    Age_first, Age_second = ploynomial_expand(Age; order=2)
    performance_mean ~ 1 + Age_first * Treatment + Age_second + (1 + Treatment | Subject) + (1 + Age_first | Experimenter)
    log(performance_sd) ~ 1 + Age * BMI + max(Age, BMI) + (1 + Age * BMI | Subject)
    Performance ~ Normal(performance_mean, performance_sd)
end


# model(__df__) = begin
#     (; Age, Treatment, Subject, Experimenter, BMI) = data(__df__)
#     (; performance_mean, performance_sd, Performance) = maybedata(__df__)
#     (Age_first, Age_second) = (ensurecols)(ploynomial_expand, Age; order=2)
#     performance_mean = ((maybedists)(; force=isdata(performance_mean)))(1 + Age_first * Treatment + Age_second + ((1 + Treatment) | Subject) + ((1 + Age_first) | Experimenter))
#     log(performance_sd) = ((maybedists)(; force=isdata(log(performance_sd))))(1 + Age * BMI + (ensurecols)(max, Age, BMI) + ((1 + Age * BMI) | Subject))
#     Performance = ((maybedists)(; force=isdata(Performance)))(Normal, performance_mean, performance_sd)
#     BRM(; Age_first, Age_second, Age, performance_mean, Treatment, Subject, Experimenter, performance_sd, BMI, Performance)
# end
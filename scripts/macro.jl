using OrderedCollections
macro brm(x)
    esc(_brm(x))
end
macro brm(df, x)
    esc(Expr(:call, _brm(x), df)) 
end
macro n(x)
    esc(_n(x))
end
macro x(x)
    esc(_x(x))
end
macro getproperty(x)
    esc(_getproperty(x))
end
_getproperty(x::Expr) = begin 
    @assert x.head == :(.)
    @assert length(x.args) == 2
    lhs, qrhs = x.args
    :(hasproperty($lhs, $qrhs) ? $x : $(qrhs.value))
end
begin
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
function assign end
function doublepipe end
function gr end
_brm(x::AbstractString; kwargs...) = _brm(Meta.parse("""
begin
    $x
end
"""); kwargs...)
_brm(x::Expr; df=nothing) = begin 
    lhs, x = x.head == :(=) ? x.args : (:($(gensym("model"))(__df__)), x)
    alllocals = OrderedDict{Symbol,Symbol}()
    info = (;alllocals)
    x = parse!(x; info)
    nonlocals = [key for (key, value) in pairs(alllocals) if value == :nonlocal]
    maybelocals = [key for (key, value) in pairs(alllocals) if value == :maybelocal]
    init = quote
        (;$(nonlocals...)) = data(__df__)
        (;$(maybelocals...)) = maybedata(__df__)
    end
    finalize = quote
        $BRMI(;$(keys(alllocals)...))
    end
    if isnothing(df)
        Expr(:(=), lhs, Expr(:block, init, x, finalize))
    else
        Expr(:let, 
            Expr(:block, :(__df__ = $df), :(__ddf__ = $data(__df__)), [:($nonlocal = @getproperty __ddf__.$nonlocal) for nonlocal in nonlocals]...), 
            Expr(:block, :((;$(maybelocals...)) = maybedata(__df__)), x, finalize)
        )
    end
end
brm(df, formula::AbstractString) = eval(_brm(formula; df))
parse!(x; info) = x
parse!(x::Expr; info) = if x.head == :block
    Expr(:block, parse!.(x.args; info)...)
elseif x.head == :(=)
    lhs, rhs = x.args
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:local)
    :(@n $lhs = @x $assign($(xname(lhs)), $rhs))
elseif isxcall(x, :~)
    _, lhs, rhs = x.args
    parselocals!(rhs; info, val=:nonlocal)
    parselocals!(lhs; info, val=:maybelocal)
    :(@n $lhs = @x $x)
else
    dump(x)
    error("Don't know how to handle parse!($x)!")
end
parselocals!(x; kwargs...) = x
parselocals!(x::Symbol; info, val) = get!(info.alllocals, x, val)
parselocals!(x::Expr; info, val) = if Meta.isexpr(x, (:call, :kw))
    parselocals!.(x.args[2:end]; info, val)
else
    parselocals!.(x.args; info, val)
end
_n(x::Expr) = begin 
    @assert x.head == :(=)
    lhs, rhs = x.args
    alhs = xassignable(lhs)
    nlhs = xname(alhs)
    :($alhs = $NamedColumn($nlhs, $rhs))
end
xassignable(x::Symbol) = x
xassignable(x::Expr) = if Meta.isexpr(x, (:tuple, :vect))
    Expr(x.head, xassignable.(x.args)...)
elseif x.head == :call 
    if length(x.args) == 2
        xassignable(x.args[2])
    else
        @warn "Don't know how to handle xassignable($x)!"
        Symbol(x)
    end
else
    dump(x)
    error("Don't know how to handle xassignable($x)!")
end
xname(x::Symbol) = Meta.quot(x)
xname(x::Expr) = if Meta.isexpr(x, (:tuple, :vect))
    Expr(x.head, xname.(x.args)...)
else
    @warn "Don't know how to handle xassignable($x)!"
    Symbol(x)
    # dump(x)
    # error("Don't know how to handle xname($x)!")
end
_x(x) = x
_x(x::Symbol) = x
_x(x::Expr) = if x.head == :call
    Expr(:call, ExprColumn, _x.(x.args)...) |> fixcall
elseif x.head ==  :||
    Expr(:call, ExprColumn, doublepipe, _x.(x.args)...)
else
    Expr(x.head, _x.(x.args)...)
end
struct Data{P}
    parent::P
end
Base.parent(d::Data) = getfield(d, :parent)
Base.hasproperty(d::Data, x::Symbol) = hasproperty(parent(d), x)
Base.getproperty(d::Data, x::Symbol) = NamedColumn(x, DataColumn(getproperty(parent(d), x)))
data(x) = Data(x)
struct MaybeData{P}
    parent::P
end
Base.parent(d::MaybeData) = getfield(d, :parent)
Base.hasproperty(d::MaybeData, x::Symbol) = hasproperty(parent(d), x)
Base.getproperty(d::MaybeData, x::Symbol) = NamedColumn(x, hasproperty(d, x) ? DataColumn(getproperty(parent(d), x)) : MissingColumn())
maybedata(x) = MaybeData(x)
abstract type AbstractColumn end
struct MissingColumn <: AbstractColumn end
struct DataColumn{P} <: AbstractColumn
    parent::P
end
Base.parent(d::DataColumn) = getfield(d, :parent)
struct NamedColumn{N,P} <: AbstractColumn
    name::N
    parent::P
end
name(x::NamedColumn) = getfield(x, :name)
Base.parent(x::NamedColumn) = getfield(x, :parent)

struct ExprColumn{F,A<:Tuple,K<:NamedTuple} <: AbstractColumn
    f::F
    args::A
    kwargs::K
    ExprColumn(f, args...; kwargs...) = new{typeof(f),typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...))
    ExprColumn(f::Type, args...; kwargs...) = new{Type{f},typeof(args),typeof((;kwargs...))}(f,args,(;kwargs...))
end
getf(x::ExprColumn) = getfield(x, :f)
getargs(x::ExprColumn) = getfield(x, :args)
getargs(x::ExprColumn, n) = (rv = getargs(x); @assert length(rv) == n; rv)
getargs(::typeof(+), x::ExprColumn{typeof(+)}) = getargs(x)
getargs(::typeof(+), x::ExprColumn) = (x,)
getargs(::typeof(+), x) = (x,)
getkwargs(x::ExprColumn) = getfield(x, :kwargs)
getop(x) = getf(x)
getop(::ExprColumn{typeof(doublepipe)}) = :||
getop(::ExprColumn{typeof(assign)}) = :(=)

struct LikelihoodColumn{P,R} <: AbstractColumn
    parent::P
    rhs::R
end
Base.parent(d::LikelihoodColumn) = getfield(d, :parent)
rhs(d::LikelihoodColumn) = getfield(d, :rhs)
maybedists(lhs::AbstractColumn, x::AbstractColumn) = LikelihoodColumn(lhs, x)

struct BRMI{O<:NamedTuple}
    operations::O
end
BRMI(;kwargs...) = BRMI((;kwargs...))
Base.show(io::IO, (;operations)::BRMI) = begin 
    print(io, "BRMI:\n")
    for (key, value::NamedColumn) in pairs(operations)
        print(io, "  ", key, ": ", parent(value), "\n")
    end
end
Base.show(io::IO, d::DataColumn) = begin
    print(io, "data (eltype=", eltype(parent(d)), ")")
end
Base.show(io::IO, x::ExprColumn{<:Union{typeof.((~,*,+,|,doublepipe,assign))...}}) = begin
    print(io, "(", )
    join(io, getargs(x), " $(getop(x)) ")
    print(io, ")")
end
nonemptyjoin(io::IO, iterator, args...; first) = if length(iterator) > 0
    print(io, first)
    join(io, iterator, args...)
end
Base.show(io::IO, x::ExprColumn) = begin
    print(io, getf(x), "(", )
    join(io, getargs(x), ", ")
    nonemptyjoin(io, ["$key=$value" for (key, value) in pairs(getkwargs(x))], ", "; first="; ")
    print(io, ")")
end
Base.show(io::IO, x::NamedColumn) = print(io, name(x))

end
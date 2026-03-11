using LogExpFunctions, InverseFunctions, Distributions, ElasticArrays, LogDensityProblems, LinearAlgebra

struct VBRMI{P<:BRMI,M<:NamedTuple}
    parent::P
    meta::M
end
VBRMI(p::BRMI) = VBRMI(p, finalize(foldl(vmeta, p.operations; init=(;materialized=(;), blocks=(;)))))
finalize(x) = merge(x, (;block_data=map(x.blocks) do values 
    m, n = size(values)
    (;L=zeros(n, n))
end))
rmerge(x::NamedTuple, y::NamedTuple) = begin
    xykeys = (intersect(keys(x), keys(y))...,)
    merge(x, y, map(rmerge, NamedTuple{xykeys}(x), NamedTuple{xykeys}(y)))
end
rmerge(::AbstractDict, ::AbstractDict) = error("Can only rmerge NamedTuples for now!")
rmerge(x, y) = y
vmeta(meta, x::NamedColumn) = begin
    meta, m = vmeta(meta, parent(x))::Tuple
    rmerge(meta, (;materialized=(;name(x)=>m)))
end
vmeta(meta, x::DataColumn) = meta, x
vmeta(meta, x::ExprColumn{typeof(assign)}) = vmeta_assignment(meta, getargs(x)...; getkwargs(x)...)
vmeta(meta, x::ExprColumn{typeof(~)}) = vmeta_sampling(meta, getargs(x)...; getkwargs(x)...)

vmeta_assignment(meta, ::Symbol, x) = meta, vmaterialize(vbroadcasted(x; meta))
vbroadcasted(;kwargs...) = (args...)->vbroadcasted(args...; kwargs...)
vbroadcasted(x::NamedColumn{<:Any,<:DataColumn}; meta) = parent(meta.materialized[name(x)])
vbroadcasted(x::NamedColumn; meta) = meta.materialized[name(x)]
vbroadcasted(x::ExprColumn; meta) = Base.broadcasted(getf(x), map(vbroadcasted(;meta), getargs(x))...)
getinverse(x::ExprColumn{<:Any,<:Tuple{<:Any}}) = inverse(getf(x))
getinverse(x::ExprColumn{<:Any,<:Tuple{<:ExprColumn}}) = inverse(getf(x)) ∘ getinverse(getargs(x, 1)[1]) 
vmeta_sampling(meta, lhs::ExprColumn, rhs) = begin
    meta, o = vmeta_sampling_rhs(meta, rhs; group=:__population__)
    meta, vmaterialize(Base.broadcasted(getinverse(lhs), o))
end
vmeta_sampling(meta, ::NamedColumn{<:Any,MissingColumn}, rhs) = begin
    meta, o = vmeta_sampling_rhs(meta, rhs; group=:__population__)
    meta, vmaterialize(o)
end
vmeta_sampling(meta, lhs::NamedColumn{<:Any,<:DataColumn}, rhs) = begin
    meta, o = vmeta_sampling_rhs(meta, rhs; group=:__population__)
    meta, LikelihoodColumn(parent(parent(lhs)), o)
end
vmeta_sampling_rhs(;kwargs...) = (args...)->vmeta_sampling_rhs(args...; kwargs...)
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(+)}; kwargs...) = begin 
    meta, args = foldl(getargs(x); init=(meta, ())) do (_meta, _args), _arg
        _meta, _arg = vmeta_sampling_rhs(_meta, _arg; kwargs...)
        _meta, (_args..., _arg)
    end
    meta, Base.broadcasted(+, args...)
end
vmeta_sampling_rhs(meta, x::ExprColumn; kwargs...) = vmeta_sampling_rhs(meta, vbroadcasted(x; meta); kwargs...)
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(*)}; kwargs...) = error("NOT IMPLEMENTED")
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(&)}; kwargs...) = error("NOT IMPLEMENTED")
vmeta_sampling_rhs(meta, x::ExprColumn{typeof(|)}; kwargs...) = begin
    lhs, rhs = getargs(x, 2)
    vmeta_sampling_rhs(meta, lhs; group=rhs)
end
vmeta_sampling_rhs(meta, ::Int; group) = begin
    meta, p = growblock!!(meta, group, 1)
    meta, p
end 
vmeta_sampling_rhs(meta, x::NamedColumn; kwargs...) = vmeta_sampling_rhs(meta, meta.materialized[name(x)]; kwargs...)
vmeta_sampling_rhs(meta, x::DataColumn; kwargs...) = vmeta_sampling_rhs(meta, parent(x); kwargs...)
vmeta_sampling_rhs(meta, x::AbstractVector{<:AbstractFloat}; group) = begin
    meta, p = growblock!!(meta, group, 1)
    meta, Base.broadcasted(*, x, p)
end 
FBroadcasted{F,Style<:Union{Nothing, Base.Broadcast.BroadcastStyle},Axes} = Base.Broadcast.Broadcasted{Style,Axes,F}
vmeta_sampling_rhs(meta, x::FBroadcasted; group) = begin
    meta, p = growblock!!(meta, group, 1)
    meta, Base.broadcasted(*, x, p)
end 
vmeta_sampling_rhs(meta, x::AbstractVector{<:Integer}; group) = begin
    meta, p = growblock!!(meta, group, 1)
    meta, Base.broadcasted(*, x, p)
    # meta, p = growblock!!(meta, group, length(unique(parent(x)))-1)
    # Base.broadcasted(*, x, p)
end
vmeta_sampling_rhs(meta, x::FBroadcasted{<:Type{<:Distribution}}; group) = meta, x
vmaterialize(x) = MaterializedColumn(Base.materialize(x), x)
struct MaterializedColumn{P,B} <: AbstractColumn
    parent::P
    broadcast::B
end
Base.parent(x::MaterializedColumn) = getfield(x, :parent)
getbroadcast(x::MaterializedColumn) = getfield(x, :broadcast)
Base.broadcastable(x::MaterializedColumn) = Base.broadcastable(parent(x))

n_levels(group::NamedColumn) = length(unique(parent(parent(group))))
growblock!!(meta, group::Symbol, n) = growblock!!(meta, group, 1, n)
growblock!!(meta, group::Symbol, m, n) = begin
    g = get(meta.blocks, group) do 
        ElasticMatrix(zeros(m, 0))
    end
    idxs = (size(g, 2)+1):(size(g, 2)+n)
    append!(g, zeros(m, n))
    rmerge(meta, (;blocks=(;group=>g))), view(g, :, 1)#idxs)
end
growblock!!(meta, group::NamedColumn, n) = growblock!!(meta, name(group), n_levels(group), n)

Base.show(io::IO, (;parent, broadcast)::MaterializedColumn) = print(io, eltype(parent), "[...] .= ", broadcast)
Base.show(io::IO, (;parent, rhs)::LikelihoodColumn) = print(io, eltype(parent), "[...] .~ ", rhs)
Base.show(io::IO, vbrm::VBRMI) = begin 
    (;parent, meta) = vbrm
    print(io, parent)
    print(io, "dim: ", LogDensityProblems.dimension(vbrm), "\n")
    print(io, "materialized:\n")
    for (key, value) in pairs(meta.materialized)
        print(io, "  ", key, ": ", value, "\n")
    end
    print(io, "blocks (n_levels, n_params):\n")
    for (key, value) in pairs(meta.blocks)
        print(io, "  ", key, ": ", size(value), "\n")
    end
end
LogDensityProblems.dimension(vbrm::VBRMI) = hyperdim(vbrm) + directdim(vbrm)
hyperdim(vbrm::VBRMI) = sum(pairs(vbrm.meta.blocks)) do (k, v)
    n = size(v, 2)
    k == :__population__ ? 0 : n * (n+1) ÷ 2
end
directdim(vbrm::VBRMI) = sum(length, vbrm.meta.blocks)
advance!!(x, pos) = x[pos+1], pos+1
advance!!(x, pos, n) = view(x, pos+1:pos+n), pos+n
lprior!((;meta)::VBRMI, x::AbstractVector; init=(0., 0)) = foldl(pairs(meta.blocks); init) do (lprior, pos), (key, values)
    m, n = size(values)
    if key == :__population__
        xi, pos = advance!!(x, pos, n)
        values[1, :] .= xi
        lprior += sum(Base.Fix1(logpdf, Normal()), xi)
    else
        C = LinearAlgebra.Cholesky(meta.block_data[key].L, :L, 0)
        lprior, pos = lprior!(C, x; init=(lprior, pos))
        for vi in eachrow(values)
            xi, pos = advance!!(x, pos, n)
            mul!(vi, C.L, xi)
            lprior += sum(Base.Fix1(logpdf, Normal()), xi)
        end
    end
    lprior, pos
end |> first
log_abs_tanh(x) = begin 
    z = -2*abs(x)
    (log1mexp(z) - log1pexp(z))
end
log_square_tanh(x) = 2 * log_abs_tanh(x)
"Either wrong or better LKJCholesky unconstraining + prior"
lprior!((;L)::Cholesky, x; init, eta=1.) = begin 
    lprior, pos = init
    n = LinearAlgebra.checksquare(L)
    log_scale, pos = advance!!(x, pos)
    lprior += logpdf(Normal(), log_scale)
    L[1, 1] = exp(log_scale)
    for i in 2:n
        log_scale, pos = advance!!(x, pos)
        lprior += logpdf(Normal(), log_scale)
        xi, pos = advance!!(x, pos)
        tmp = log_abs_tanh(xi / sqrt(n-1))
        L[i, 1] = sign(xi) * exp(log_scale + tmp)
        log_sos = 2 * tmp
        lprior += log1mexp(log_sos)
        for j in 2:i-1
            xi, pos = advance!!(x, pos)
            tmp1 = .5 * log1mexp(log_sos)
            lprior += tmp1
            tmp2 = log_abs_tanh(xi / sqrt(n-j))
            lprior += log1mexp(2*tmp2)
            tmp = tmp1 + tmp2
            L[i, j] = sign(xi) * exp(log_scale + tmp)
            log_sos = logaddexp(log_sos, 2*tmp)
        end
        L[i, i] = exp(log_scale + .5 * log1mexp(log_sos))
        lprior += (n - i + 2*eta-2) * .5 * log1mexp(log_sos)
    end
    lprior, pos
end
llikelihood!((;meta)::VBRMI) = foldl(meta.materialized; init=0.) do llikelihood, m
    llikelihood + llikelihood!(m)
end
llikelihood!(::DataColumn) = 0.
llikelihood!(x::MaterializedColumn) = (Base.materialize!(parent(x), getbroadcast(x)); 0.)
# llikelihood!(x::LikelihoodColumn) = sum(Base.broadcasted(logpdf, rhs(x), parent(x)); init=0.)
# The below is faster for some reason?
llikelihood!(x::LikelihoodColumn) = ssum(Base.broadcasted(logpdf, rhs(x), parent(x)); init=0.)
llikelihood!(x) = error(typeof(x))

ssum(args...; kwargs...) = sum(args...; kwargs...)
ssum(x::Base.Broadcast.Broadcasted; init) = begin 
    rv = init
    for xi in x
        rv += xi
    end
    rv
end

Distributions.logpdf(vbrmi::VBRMI, x::AbstractVector) = lprior!(vbrmi, x) + llikelihood!(vbrmi)
LogDensityProblems.logdensity(vbrmi::VBRMI, x::AbstractVector) = lprior!(vbrmi, x) + llikelihood!(vbrmi)
# Gemini claims you can do this!
using Pkg
Pkg.activate(@__DIR__)
insert!(LOAD_PATH, 2, joinpath(@__DIR__, ".."))

include("../macro.jl")
include("../vimpl.jl")
include("../examples/database.jl")

using Chairmarks, Random

db = Database()
df = db.dataset(:bambi, :escs)
fdf = map(Vector{Float64}, (;df.drugs, df.o, df.c, df.e, df.a, df.n))
# Using fdf is considerably faster than using df
tmp = @brm fdf """
    loc ~ o + c + e + a + n
    log(err_scale) ~ 1
    drugs ~ Normal(loc, err_scale)
"""
display(@brm """
    loc ~ o + c + e + a + n
    log(err_scale) ~ 1
    drugs ~ Normal(loc, err_scale)
""")
vtmp = VBRMI(tmp)
display(vtmp)
display(@be randn(LogDensityProblems.dimension(vtmp)) LogDensityProblems.logdensity($vtmp, _))

# import Reactant

# rx = Reactant.to_rarray(randn(LogDensityProblems.dimension(vtmp)))
# errors: NoFieldMatchError(...)
# _rtmp = Reactant.to_rarray(vtmp)
# errors: MethodError: no method matching _copyto!(::SubArray{…}, ::Base.Broadcast.Broadcasted{…})
# rtmp = Reactant.@compile LogDensityProblems.logdensity(vtmp, rx)
# display(@be randn(LogDensityProblems.dimension(vtmp)) LogDensityProblems.logdensity($rtmp, _))
# error()

using LogDensityProblemsAD, Mooncake, Enzyme, DifferentiationInterface
struct ADLogDensity{F, B, E}
    f::F
    backend::B
    extras::E
end

ADLogDensity(f, backend) = ADLogDensity(
    f, 
    backend, 
    DifferentiationInterface.prepare_gradient(
        Base.Fix1(LogDensityProblems.logdensity, f), 
        backend, 
        zeros(LogDensityProblems.dimension(f))
    )
)

LogDensityProblems.capabilities(::ADLogDensity) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::ADLogDensity) = LogDensityProblems.dimension(p.f)
LogDensityProblems.logdensity(p::ADLogDensity, x) = LogDensityProblems.logdensity(p.f, x)
LogDensityProblems.logdensity_and_gradient(
    p::ADLogDensity, x
) = DifferentiationInterface.value_and_gradient(
    Base.Fix1(LogDensityProblems.logdensity, p.f), p.extras, p.backend, x
)



mvtmp1 = ADgradient(AutoMooncake(), vtmp)
mvtmp2 = ADLogDensity(vtmp, AutoMooncake())
evtmp1 = ADgradient(AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse)), vtmp)
evtmp2 = ADLogDensity(vtmp, AutoEnzyme(; 
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Duplicated
))

x = randn(LogDensityProblems.dimension(vtmp))
display(mapreduce(hcat, (mvtmp1, mvtmp2, evtmp1, evtmp2)) do b 
    LogDensityProblems.logdensity_and_gradient(b, copy(x))[2]
end)

@info "Primal"
display(@be randn(LogDensityProblems.dimension(vtmp)) LogDensityProblems.logdensity($vtmp, _))
@info "Inefficient Mooncake skipped..."
# display(@be randn(LogDensityProblems.dimension(vtmp)) LogDensityProblems.logdensity_and_gradient($mvtmp1, _))
@info "Mooncake"
display(@be randn(LogDensityProblems.dimension(vtmp)) LogDensityProblems.logdensity_and_gradient($mvtmp2, _))
@warn "wrong Enzyme skipped..."
# display(@be randn(LogDensityProblems.dimension(vtmp)) LogDensityProblems.logdensity_and_gradient($evtmp1, _))
@info "correct Enzyme"
display(@be randn(LogDensityProblems.dimension(vtmp)) LogDensityProblems.logdensity_and_gradient($evtmp2, _))
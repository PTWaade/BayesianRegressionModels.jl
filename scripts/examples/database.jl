include("all.jl")
using DynamicObjects

@dynamicstruct "" serial struct Database
    mod(m::Symbol) = _SOURCE_MODULES[m]
    @cached dataset(m::Symbol, name::Symbol) = getproperty(mod(m), :load)(Val(name)) 
    # formula(m::Symbol, name::Symbol) = getproperty(mod(m), :examples)(Val(name))[1]
    # example(m::Symbol, name::Symbol) = begin
    #     _formula, datakey = getproperty(mod(m), :examples)(Val(name))
    #     brm(dataset(m, datakey), _formula)
    # end
end 
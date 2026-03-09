using Bijectors: VectorBijectors as VB

# struct RegressionPrior{Tfixed<:MultivariateDistribution,Tsds<:MultivariateDistribution,Tcorrs<:AbstractVector,Tspecs<:RegressionSpecifications} <: Distributions.Distribution{RegressionVariate,Distributions.Continuous}
#     #Multivariate distribution flattened from vector (R regressions) of multivariate priors (across P fixed effect terms)
#     fixed_effects::Tfixed
#     #Multivariate distribution flattened from vector (R regressions) of vectors (F Random effect factors) of vectors (G random effect groups) of vectors (Q random effect terms)
#     random_effect_sds::Tsds
#     #Vector (F random effect factors) of vectors (G random effect groups) of vectors (B random effect blocks) of LKJCholesky correlation priors
#     random_effect_correlations_cholesky::Tcorrs
#     #Model specifications
#     specifications::Tspecs
# end

struct RegressionPriorToVec{B1,B2}
    fixed_effects_vec_fn::B1
    random_effect_sds_vec_fn::B2
end

function VB.from_linked_vec(d::BRM.RegressionPrior)
    return RegressionPriorToVec(
        # These are already just bog standard product distributions, so we can
        # rely on existing functionality inside Bijectors.jl.
        VB.from_linked_vec(d.fixed_effects),
        VB.from_linked_vec(d.random_effect_sds)
        # The Cholesky one is more complicated
    )
end

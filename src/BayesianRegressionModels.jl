module BayesianRegressionModels

## For the regression distribution ##
using Distributions
using LinearAlgebra
using Random
using PDMats #This gives efficient computation for positive definite matrices, used for the LKJCholesky stuff
using DimensionalData #This allows for named dimensions in arrays
using DimensionalData: @dim

include("1_RegressionSpecifications.jl")

include("2_RegressionCoefficients.jl")

include("3_RegressionPrior.jl")

include("4_RegressionPredictors.jl")

include("5_linear_combination.jl")

include("7_basis_expansions.jl")

include("8_interaction_operators.jl")

include("9_regression_submodels.jl")

include("10_formula.jl")

end

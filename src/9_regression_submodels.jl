"""
    abstract type AbstractPredictorUpdate
    
Abstract type for specifying how to update the RegressionPredictors after an operation.

Subtypes of this include ...
"""
abstract type AbstractPredictorUpdate end

## 2. Concrete type and dispatch for not updating the RegressionPredictors ##
struct NoPredictorUpdate <: AbstractPredictorUpdate end

function update_predictors!(
    predictors::Tpredictors,
    generated_values::Tgenerated_values,
    update_info::NoPredictorUpdate
) where {Tpredictors<:AbstractVector{<:RegressionPredictors},Tgenerated_values<:AbstractVector}

    return nothing

end

## 3. Concrete type and dispatch for updating the RegressionPredictors ##
struct UpdatePredictors <: AbstractPredictorUpdate
    term_label::Symbol
    regression_labels::Vector{Symbol}
end
function update_predictors!(
    predictors::Tpredictors,
    generated_values::Tgenerated_values,
    update_info::UpdatePredictors
) where {Tpredictors<:AbstractVector{<:RegressionPredictors},Tgenerated_values<:AbstractVector}

    update_predictors!(predictors, update_info.term_label, generated_values, update_info.regression_labels)

end

## 1. Abstract types for different operations in the Turing model ###
abstract type AbstractRegressionOperation end

## 2. Concrete dispatch type for perforning a linear combination of predictors and coefficients ##
struct LinearCombination <: AbstractRegressionOperation end

## 3. Abstract type for using Turing submodels ###
abstract type AbstractRegressionSubmodel <: AbstractRegressionOperation end

## 4. Container struct for the operation and update information
struct RegressionOperation{Toperation<:AbstractRegressionOperation,Tupdate<:AbstractPredictorUpdate}
    operation_info::Toperation
    predictor_update::Tupdate
    store_outcome::Bool

    function RegressionOperation(
        operation::Toperation,
        predictor_update::Tupdate=NoPredictorUpdate();
        store_outcome::Bool=false
    ) where {Toperation<:AbstractRegressionOperation,Tupdate<:AbstractPredictorUpdate}
        return new{Toperation,Tupdate}(operation, predictor_update, store_outcome)
    end
end


######################################
### SIMPLE DISTRIBUTION LIKELIHOOD ###
######################################
## 1. Concrete type for carrying information about the single distribution likelihood ##
Base.@kwdef struct DistributionLikelihood{T<:Distribution,Targs<:NamedTuple,Tkwargs<:NamedTuple,Tobservations} <: AbstractRegressionSubmodel
    #The distribution type to be used
    dist::Type{T}

    #The positional arguments to be passed to the distribution constructor
    dist_args::Targs = (;)

    #The keyword arguments to be passed to the distribution constructor
    dist_kwargs::Tkwargs = (;)

    #The observations which the observation is evaluated against
    observations::Tobservations
end

# Extended in TuringExt
function regression_model end

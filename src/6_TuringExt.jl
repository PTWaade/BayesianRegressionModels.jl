##############################
### PREDICTOR UPDATE TYPES ###
##############################
## 1. Abstract type for specifying how to update the RegressionPredictors after an operation ##
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


#######################
### OPERATION types ###
#######################
## 1. Abstract types for different operations in the Turing model ###
abstract type AbstractRegressionOperation end

## 2. Concrete dispatch type for perforning a linear combination of predictors and coefficients ##
struct LinearCombination <: AbstractRegressionOperation end

## 3. Abstract type for using Turing submodels ###
abstract type AbstractRegressionSubmodel <: AbstractRegressionOperation end 
@model function regression_submodel(
    submodel_info::Tsubmodel,
    outcomes::Toutcomes,
    predictors::Tpredictors,
) where {Tsubmodel<:AbstractRegressionSubmodel, Toutcomes<:NamedTuple, Tpredictors<:AbstractVector{<:RegressionPredictors}}

    @error "A Turing submodel has not been implemented for the type $Tsubmodel"

end
## 4. Container struct for the operation and update information
struct RegressionOperation{Toperation<:AbstractRegressionOperation, Tupdate<:AbstractPredictorUpdate}
    operation_info::Toperation
    predictor_update::Tupdate
    store_outcome::Bool

    function RegressionOperation(
        operation::Toperation, 
        predictor_update::Tupdate = NoPredictorUpdate(); 
        store_outcome::Bool = false 
        ) where {Toperation<:AbstractRegressionOperation, Tupdate<:AbstractPredictorUpdate}
        return new{Toperation, Tupdate}(operation, predictor_update, store_outcome)
    end
end


##############################
### TOP-LEVEL TURING MODEL ###
##############################
## 1. Full multi-step regression model ##
@model function regression_model(
    predictors::Tpredictors,
    prior::Tprior,
    operations::Toperations,
) where {Tpredictors<:AbstractVector{<:RegressionPredictors},Tprior<:RegressionPrior,Toperations<:NamedTuple}

    # 0. Prepare outcome container
    outcomes = (;)

    # 1. Sample coefficients
    coefficients ~ prior

    ## 2. Extract information ##
    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = get_fixed_effects(coefficients)
    #Vector (R regressions) of vectors (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = get_random_effects(coefficients)
    #Collection of labels
    labels = coefficients.specifications.labels

    #For each regression operation
    for (operation_label, val) in pairs(operations)

        #Unpack the operation
        operation = val.operation_info
        store_outcome = val.store_outcome
        predictor_update = val.predictor_update

        ## 3. For calculating linear combinations ##
        if operation isa LinearCombination

            #Apply the linear combination
            generated_values = linear_combination(
                fixed_effects,
                random_effects,
                predictors,
                labels,
                operation_label,
            )

        ## 4. For applying a Turing submodel as likelihood or to generate values ##
        elseif operation isa AbstractRegressionSubmodel

            #Apply the Turing submodel (likelihood and/or generation values)
            generated_values ~ to_submodel(prefix(regression_submodel(operation, outcomes, predictors), operation_label), false)

        end

        ## 5. Store the output in the predictions object ##
        if store_outcome
            outcomes = merge(outcomes, NamedTuple{(operation_label,)}((generated_values,)))
        end

        ## 6. Update the regression predictors ##
        update_predictors!(predictors, generated_values, predictor_update)

    end

    #Return the generated values and the coefficients for postprocessing
    return (coefficients = coefficients, predictors = predictors, outcomes = outcomes)
end
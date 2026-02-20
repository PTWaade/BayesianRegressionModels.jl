module BRMTuringExt

import BayesianRegressionModels as BRM
using Turing
using FlexiChains

## 2. Submodel for evaluating the likelihood of observations, parameterising a distribution with predictions, a prior, or pre-specified values ##
@model function regression_submodel(
    likelihood_info::Tlikelihood,
    predictions::Tpredictions,
    predictors::Tpredictors,
) where {Tlikelihood<:BRM.DistributionLikelihood,Tpredictions<:NamedTuple,Tpredictors<:AbstractVector{<:BRM.RegressionPredictors}}

    ## 1. Extract observations ##
    #If the observations are a pre-specified array
    if likelihood_info.observations isa AbstractArray

        #Use them as they are
        observations = likelihood_info.observations

        #If they are to be extracted from the predictors
    elseif likelihood_info.observations isa BRM.ExtractPredictors

        #Extract the corresponding, transformed, predictors
        observations = BRM.get_predictor_values(predictors, likelihood_info.observations)

        #If the observations come from the linear predictions
    elseif likelihood_info.observations isa BRM.ExtractOutcome

        #Extract the corresponding, transformed, predictions
        observations = likelihood_info.observations.inv_link.(getproperty(likelihood_info, likelihood_info.observations.regression_label))

        #If the observations are to be sampled from a distribution
    elseif likelihood_info.observations isa Distribution

        #Sample the parameter using :observations as prefix
        observations ~ to_submodel(prefix(sample_dist(likelihood_info.observations), :observations), false)

    end

    # 2. Find the number of observations in the observations
    n_observations = size(observations)[end]

    # 3. Go through every positional argument and corresponding key for the distribution
    dist_args = NamedTuple(
        begin

            #If the argument values are in the linear predictions
            if val isa BRM.ExtractOutcome

                #Extract and transform them
                arg = val.inv_link.(getproperty(predictions, val.regression_label))

                #If the argument values are to be extracted from the predictors
            elseif val isa BRM.ExtractPredictors

                #Extract the corresponding, transformed, predictors
                arg = BRM.get_predictor_values(predictors, val)

                #If a prior distribution has been specified
            elseif val isa Distribution

                #Sample the parameter using the key as a prefix
                param ~ to_submodel(prefix(sample_dist(val), key), false)

                #Use it for every observation
                arg = fill(param..., n_observations)

                #If a vector/scalar of values has been specified
            elseif val isa AbstractArray

                #Just use it as it is
                arg = val

                #If the argument is a single scalar
            elseif val isa Real

                #Use the same value for all observations
                arg = fill(val, n_observations)

            else

                @error "A distribution argument of type $(typeof(val)) was passed to the DistributionLikelihood. This is not supported."

            end

            #For matrices with only one dimension
            if size(arg, 2) == 1
                #Make them vectors
                arg = vec(arg)
            else
                #Make other matrices into vectors of vectors
                arg = collect(eachrow(arg))
            end

            #Store the positional argument
            key => arg

        end

        for (key, val) in pairs(likelihood_info.dist_args)
    )

    # 4. Go through every keyword argument and corresponding key for the distribution
    dist_kwargs = NamedTuple(
        begin

            #If the argument values are in the linear predictions
            if val isa BRM.ExtractOutcome

                #Extract and transform them
                arg = val.inv_link.(getproperty(predictions, val.regression_label))

                #If the argument values are to be extracted from the predictors
            elseif val isa BRM.ExtractPredictors

                #Extract the corresponding, transformed, predictors
                arg = BRM.get_predictor_values(predictors, val)

                #If a prior distribution has been specified
            elseif val isa Distribution

                #Sample the parameter using the key as a prefix
                param ~ to_submodel(prefix(sample_dist(val), key), false)

                #Use it for every observation
                arg = fill(param..., n_observations)

                #If a vector/scalar of values has been specified
            elseif val isa AbstractArray

                #Just use it as it is
                arg = val

                #If the argument is a single scalar
            elseif val isa Real

                #Use the same value for all observations
                arg = fill(val, n_observations)

            else

                @error "A distribution argument of type $(typeof(val)) was passed to the DistributionLikelihood. This is not supported."

            end

            #For matrices with only one dimension
            if size(arg, 2) == 1
                #Make them vectors
                arg = vec(arg)
            else
                #Make other matrices into vectors of vectors
                arg = collect(eachrow(arg))
            end

            #Store the keyword argument
            key => arg

        end

        for (key, val) in pairs(likelihood_info.dist_kwargs)
    )

    # 5. Evaluate corresponding likelihood distributions for each outcome
    imputed_observations ~ to_submodel(evaluate_observations(
            observations,
            product_distribution(likelihood_info.dist.(dist_args...; dist_kwargs...))
        ), false)

    # 6. Return (potentially imputed) observations
    return imputed_observations

end

## 3. Convenience submodel to allow sampling inside loops ##
@model function sample_dist(dist)

    return val ~ dist

end

## 4. Convenience submodel to allow evaluating the observations (they have to be a model argument)
@model function evaluate_observations(observations::Tobservations, dist::Tdist) where {Tobservations<:AbstractArray,Tdist<:Distribution}

    observations ~ dist

    return observations

end

@model function regression_submodel(
    submodel_info::Tsubmodel,
    outcomes::Toutcomes,
    predictors::Tpredictors,
) where {Tsubmodel<:BRM.AbstractRegressionSubmodel,Toutcomes<:NamedTuple,Tpredictors<:AbstractVector{<:BRM.RegressionPredictors}}

    @error "A Turing submodel has not been implemented for the type $Tsubmodel"

end

##############################
### TOP-LEVEL TURING MODEL ###
##############################
## 1. Full multi-step regression model ##
@model function BRM.regression_model(
    predictors::Tpredictors,
    prior::Tprior,
    operations::Toperations,
) where {Tpredictors<:AbstractVector{<:BRM.RegressionPredictors},Tprior<:BRM.RegressionPrior,Toperations<:NamedTuple}

    # 0. Prepare outcome container
    outcomes = (;)

    # 1. Sample coefficients
    coefficients ~ prior

    ## 2. Extract information ##
    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = BRM.get_fixed_effects(coefficients)
    #Vector (R regressions) of vectors (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = BRM.get_random_effects(coefficients)
    #Collection of labels
    labels = coefficients.specifications.labels

    #For each regression operation
    for (operation_label, val) in pairs(operations)

        #Unpack the operation
        operation = val.operation_info
        store_outcome = val.store_outcome
        predictor_update = val.predictor_update

        ## 3. For calculating linear combinations ##
        if operation isa BRM.LinearCombination

            #Apply the linear combination
            generated_values = BRM.linear_combination(
                fixed_effects,
                random_effects,
                predictors,
                labels,
                operation_label,
            )

            ## 4. For applying a Turing submodel as likelihood or to generate values ##
        elseif operation isa BRM.AbstractRegressionSubmodel

            #Apply the Turing submodel (likelihood and/or generation values)
            generated_values ~ to_submodel(prefix(regression_submodel(operation, outcomes, predictors), operation_label), false)

        end

        ## 5. Store the output in the predictions object ##
        if store_outcome
            outcomes = merge(outcomes, NamedTuple{(operation_label,)}((generated_values,)))
        end

        ## 6. Update the regression predictors ##
        BRM.update_predictors!(predictors, generated_values, predictor_update)

    end

    #Return the generated values and the coefficients for postprocessing
    return (coefficients=coefficients, predictors=predictors, outcomes=outcomes)
end

end

######################################
### SIMPLE DISTRIBUTION LIKELIHOOD ###
######################################
## 1. Concrete type for carrying information about the single distribution likelihood ##
Base.@kwdef struct DistributionLikelihood{T<:Distribution, Targs<:NamedTuple, Tkwargs<:NamedTuple, Tobservations} <: AbstractRegressionSubmodel
    #The distribution type to be used
    dist::Type{T}

    #The positional arguments to be passed to the distribution constructor
    dist_args::Targs = (;)

    #The keyword arguments to be passed to the distribution constructor
    dist_kwargs::Tkwargs = (;)

    #The observations which the observation is evaluated against
    observations::Tobservations
end

## 2. Submodel for evaluating the likelihood of observations, parameterising a distribution with predictions, a prior, or pre-specified values ##
@model function regression_submodel(
    likelihood_info::Tlikelihood,
    predictions::Tpredictions,
    predictors::Tpredictors,
) where {Tlikelihood<:DistributionLikelihood, Tpredictions<:NamedTuple, Tpredictors<:AbstractVector{<:RegressionPredictors}}

    ## 1. Extract observations ##
    #If the observations are a pre-specified array
    if likelihood_info.observations isa AbstractArray

        #Use them as they are
        observations = likelihood_info.observations

    #If they are to be extracted from the predictors
    elseif likelihood_info.observations isa ExtractPredictors

        #Extract the corresponding, transformed, predictors
        observations = get_predictor_values(predictors, likelihood_info.observations)

    #If the observations come from the linear predictions
    elseif likelihood_info.observations isa ExtractOutcome

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
            if val isa ExtractOutcome

                #Extract and transform them
                arg = val.inv_link.(getproperty(predictions, val.regression_label))

            #If the argument values are to be extracted from the predictors
            elseif val isa ExtractPredictors

                #Extract the corresponding, transformed, predictors
                arg = get_predictor_values(predictors, val)

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
            if val isa ExtractOutcome

                #Extract and transform them
                arg = val.inv_link.(getproperty(predictions, val.regression_label))

            #If the argument values are to be extracted from the predictors
            elseif val isa ExtractPredictors

                #Extract the corresponding, transformed, predictors
                arg = get_predictor_values(predictors, val)

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
    observations ~ product_distribution(likelihood_info.dist.(dist_args...; dist_kwargs...))

    @show observations

    return observations

end

## 3. Convenience submodel to allow sampling inside loops ##
@model function sample_dist(dist)

    return val ~ dist

end
######################################
### SIMPLE DISTRIBUTION LIKELIHOOD ###
######################################

## 1. Concrete type for carrying information about the single distribution likelihood ##
struct DistributionLikelihood{T<:Distribution,Targs<:NamedTuple} <: AbstractRegressionLikelihood
    #The distribution type to be used
    dist::Type{T}
    #The arguments to be passed to the distribution constructor
    dist_args::Targs
end

## 2. Extension of top-level likelihood model ##
@model function regression_likelihood(
    outcomes::Toutcomes,
    predictions::Tpredictions,
    likelihood_info::Tlikelihood,
) where {Toutcomes<:NamedTuple,Tlikelihood<:Tuple{Vararg{<:DistributionLikelihood}},Tpredictions<:DimVector}

    # For each set of outcomes and corresponding likelihood
    for ((key_l, outcomes_l), likelihood) in zip(pairs(outcomes), likelihood_info)
        # Apply the distribution likelihood model
        outcomes_l ~ to_submodel(prefix(distribution_likelihood(outcomes_l, predictions, likelihood), key_l), false)
    end

end

## 3. Submodel for evaluating the likelihood of data, parameterising a distribution with predictions, a prior, or pre-specified values ##
@model function distribution_likelihood(
    outcomes::Tdata,
    predictions::Tpredictions,
    likelihood_info::Tlikelihood,
) where {Tdata<:AbstractVector,Tlikelihood<:DistributionLikelihood,Tpredictions<:DimVector}

    # Find the number of observations in the data
    n_observations = size(outcomes)[end]

    # Go through every argument and corresponding key for the distribution
    dist_args = Tuple(
        begin

            if arg isa Tuple{Symbol, Function} # If a symbol has been specified

                # Unpack the regression label and the inverse link function
                (predictions_label, inv_link) = arg

                # Get the prediction associated with that symbol, and transform them using the inverse link function
                inv_link.(predictions[At(predictions_label)])

            
            elseif arg isa Distribution # If a prior distribution has been specified

                # Sample the parameter using the key as a prefix
                param ~ to_submodel(prefix(sample_dist(arg), key), false)

                # Use it for every observation
                fill(param, n_observations)

                
            elseif arg isa AbstractVector # If a vector/scalar of values has been specified
                arg
            end
        end

        for (key, arg) in pairs(likelihood_info.dist_args)
    )

    # Evaluate the distribution for each observation
    outcomes ~ product_distribution([likelihood_info.dist(dist_args_o...) for dist_args_o in zip(dist_args...)])

end

## 4. Convenience submodel to allow sampling inside loops ##
@model function sample_dist(dist)

    return val ~ dist

end
######################################
### SIMPLE DISTRIBUTION LIKELIHOOD ###
######################################

## 1. Concrete type for carrying information about the single distribution likelihood ##
struct DistributionLikelihood{T<:Distribution,Targs<:NamedTuple} <: AbstractRegressionLikelihood
    dist::Type{T}
    dist_args::Targs
end

## 2. Extension of top-level likelihood model ##
@model function regression_likelihood(
    outcomes::Tdata,
    predictions::Tpredictions,
    likelihood_info::Tlikelihood,
) where {Tdata<:NamedTuple,Tlikelihood<:Tuple{Vararg{<:DistributionLikelihood}},Tpredictions<:DimVector}

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
            # If a symbol has been specified
            if arg isa Symbol

                # Get the prediction associated with that symbol
                predictions[At(arg)]

                # If a prior distribution has been specified
            elseif arg isa Distribution

                # Sample the parameter using the key as a prefix
                param ~ to_submodel(prefix(sample_dist(arg), key), false)

                # Use it for every observation
                fill(param, n_observations)

                # If a vector/scalar of values has been specified
            else
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

    return out ~ dist

end
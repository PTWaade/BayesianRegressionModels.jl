

#####################
### TURING MODELS ###
#####################

struct SimpleRegressionLikelihood end

###### SIMPLE TURING MODEL ######
@model function simple_regression(
    predictors::RegressionPredictors, 
    outcomes::T, 
    priors::RegressionPrior, 
    likelihood::SimpleRegressionLikelihood
    ) where {T<:AbstractArray}

    # 1. Sample coefficients
    coefficients ~ priors

    # 2. Calculate predictions
    predictions = linear_combination(predictors, coefficients)

    # 3. Implement likelihood function
    outcomes ~ to_submodel(likelihood.model(predictions, outcomes, likelihood))

    return outcomes
end


###### MULTISTEP TURING MODEL ######

## 1. Types for different operations ##
abstract type RegressionOperation end

## 1.1 Type for perforning a linear combination of predictors and coefficients ##
struct LinearPrediction{Tupdate<:RegressionUpdateInfo} <: RegressionOperation 
    regression_label::Symbol
    update_info::Tupdate
end

## 1.2 Type for using a Turing submodel as a likelihood, combining outcomes and predictors ##
struct LikelihoodSubmodel{Tkwargs <: NamedTuple, Tupdate<:RegressionUpdateInfo} <: RegressionOperation 
    model::Function
    kwargs::Tkwargs
    regression_label::Symbol
    predictor_labels::Vector{Symbol}
    outcome_labels::Vector{Symbol}
    submodel_prefix::Symbol
    update_info::Tupdate
end

## 1.3 Type for using a Turing submodel to generate new values that can later be used as predictors ##
struct GenerationSubmodel{Tkwargs<:NamedTuple, Tupdate<:RegressionUpdateInfo} <: RegressionOperation 
    model::Function
    kwargs::Tkwargs
    regression_label::Symbol
    predictor_labels::Vector{Symbol}
    submodel_prefix::Symbol
    update_info::Tupdate
end

## 1.4 Type for using a normal function to generate new values with a function ##
struct GenerationFunction{Tkwargs <: NamedTuple, Tupdate<:RegressionUpdateInfo} <: RegressionOperation
    func::Function
    kwargs::Tkwargs
    regression_label::Symbol
    predictor_labels::Vector{Symbol}
    update_info::Tupdate
end


## 2. Turing model for carrying out the whole regression ##
@model function multistep_regression(predictors::RegressionPredictors, priors::RegressionPrior, operations::Tuple{Vararg{Toperations}}) where {Toperations<:RegressionOperation}

    ## 1. Sample coefficients ##
    coefficients ~ priors

    ## 2. Extract information ##
    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = get_fixed_effects(coefficients)
    #Vector (R regressions) of vectors (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = get_random_effects(coefficients)
    #Collection of labels
    labels = coefficients.specifications.labels

    ## 3. Apply operations one at a time ##
    #Initialise vector of generated values
    generated_values = Vector(undef, length(operations)) #CAN WE MAKE THIS TYPE-STABLE PLEASE

    #For each operation
    for (i, operation) in enumerate(operations)

        ## 3.1 For linear predictions ##
        if operation isa LinearPrediction

            #Apply the linear prediction
            generated_values[i] = linear_combination(
                fixed_effects, 
                random_effects, 
                predictors, 
                labels, 
                operation.regression_label, 
            )
        
        ## 3.2 For applying a likelihood with a Turing submodel ##
        elseif operation isa LikelihoodSubmodel

            #Extract the needed predictors
            predictors_op = get_predictors(predictors, operation.predictor_labels, operation.regression_label)

            #Extract the needed outcomes
            outcomes_op = get_outcomes(outcomes, operation.outcome_labels, operation.regression_label)

            #Pass both to the likelihood model
            generated_values[i] ~ to_submodel(prefix(operation.model(predictors_op, outcomes_op; operation.kwargs...), operation.submodel_prefix), false) 

        
        ## 3.3 For generating new variables with a Turing submodel ##
        elseif operation isa GenerationSubmodel

            #Extract the needed predictors
            predictors_op = get_predictors(predictors, operation.predictor_labels, operation.regression_label)

            #Pass them to the submodel to generate new values
            generated_values[i] ~ to_submodel(prefix(operation.model(predictors_op; operation.kwargs...), operation.submodel_prefix), false) 


        ## 3.4 For applying a transformation function ##
        elseif operation isa GenerationFunction

            #Extract the needed predictors
            predictors_op = get_predictors(predictors, operation.predictor_labels, operation.regression_label)

            #Apply the transformation
            generated_values[i] = operation.func(predictors_op; operation.kwargs...)

        end

        ## 4. Update predictors for the next loop ##
        update!(predictors, generated_values[i], opertaion.update_info)

    end

    return generated_values
end
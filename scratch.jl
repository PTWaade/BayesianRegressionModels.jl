









## 2. Model using generated predictors ##
@model function m2(predictors::RegressionPredictors, priors::RegressionPrior)

    ## 0. Extract model information ##
    labels = priors.specifications.labels

    ## 1. Sample coefficients ##
    coefficients ~ priors

    ## 2. Extract information ##
    # Extract coefficients
    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = get_fixed_effects(coefficients)
    #Vector (R regressions) of vectors (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = get_random_effects(coefficients)

    ## 3. Calculate the outcomes for the first regression ##
    outcomes_1 = linear_combination(
            fixed_effects, 
            random_effects, 
            predictors, 
            labels, 
            :Regression1, 
        )

    ## 4. Overwrite Term2 with the outcomes ##
    new_predictors = update_predictor(predictors, outcomes_1, :Term2)

    ## 5. DO the other regression using the new predictor ##
    outcomes_2 = linear_combination(
            fixed_effects, 
            random_effects, 
            new_predictors, 
            labels, 
            :Regression2, 
        )

    # 3. Here the likelihood would come

end

model = m2(predictors, priors)

chain = sample(model, Prior(), 1000, chain_type=VNChain)





## 3. Model using generated predictors, and updating them by mutation ##
@model function m3(predictors::RegressionPredictors, priors::RegressionPrior)

    ## 0. Extract model information ##
    labels = priors.specifications.labels

    ## 1. Sample coefficients ##
    coefficients ~ priors

    ## 2. Extract information ##
    # Extract coefficients
    #Vector (R regressions) of vectors (P fixed effect terms)
    fixed_effects = get_fixed_effects(coefficients)
    #Vector (R regressions) of vectors (F factors) of matrices (J random effect levels, Q_total random effect terms)
    random_effects = get_random_effects(coefficients)

    ## 3. Calculate the outcomes for the first regression ##
    outcomes_1 = linear_combination(
            fixed_effects, 
            random_effects, 
            predictors, 
            labels, 
            :Regression1, 
        )

    ## 4. Overwrite Term2 with the outcomes ##
    update_predictor!(predictors, outcomes_1, :Term2)

    ## 5. DO the other regression using the new predictor ##
    outcomes_2 = linear_combination(
            fixed_effects, 
            random_effects, 
            predictors, 
            labels, 
            :Regression2, 
        )

    # 3. Here the likelihood would come

end

model = m3(deepcopy(predictors), priors)

chain = sample(model, Prior(), 1000, chain_type=VNChain)





## 4. Model using generated level assignments, and updating them by mutation ##
@model function m4(predictors::RegressionPredictors, priors::RegressionPrior)

    ## 1. Sample coefficients ##
    coefficients ~ priors

    ## 2. Sample the item factor
    item_assignments ~ filldist(Categorical(.3, .3, .4), 12)

    ## 3. Update the item factor ##
    updated_predictors = update_level_assignments(predictors, item_assignments, :ItemFactor)

    ## 4. Calculate the outcomes ##
    outcomes = linear_combination(updated_predictors, coefficients)

    # 5. Here the likelihood would come

end

model = m4(deepcopy(predictors), priors)

chain = sample(model, Prior(), 1000, chain_type=VNChain)




## 5. Model using generated level assignments, and updating them by mutation ##
@model function m5(predictors::RegressionPredictors, priors::RegressionPrior)

    ## 1. Sample coefficients ##
    coefficients ~ priors

    ## 2. Sample the item factor
    item_assignments ~ filldist(Categorical(.3, .3, .4), 12)

    ## 3. Update the item factor ##
    update_level_assignments!(predictors, item_assignments, :ItemFactor)

    ## 4. Calculate the outcomes ##
    outcomes = linear_combination(predictors, coefficients)

    # 5. Here the likelihood would come

end

model = m5(deepcopy(predictors), priors)

chain = sample(model, Prior(), 1000, chain_type=VNChain)







using Turing

@model function m(x)

    a ~ Normal(0, 1)

    b ~ Normal(0, 1)

    σ ~ Exponential(1)

    x ~ Normal(a + b, σ)

end

posterior_sampled = sample(m(10), NUTS(), 1000)
posterior_variational = vi(m, full_rank_gaussian(m), 1000)
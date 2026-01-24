
include(joinpath("..", "src", "BayesianRegressionModels.jl"))



#########################################
### TEST REGRESSIONPRIOR DISTRIBUTION ###
#########################################

## Labels ##
#Names for the different regressions
regression_labels = RegressionDim([:Regression1, :Regression2])

#Names for the fixed effect terms in each regression
fixed_effect_term_labels = DimArray([
        #Regression 1
        FixedEffectTermDim([:Term1, :Term2, :Term3]),
        #Regression 2
        FixedEffectTermDim([:Term1, :Term2, :Term3, :Term4, :Term5])
    ], regression_labels)

#Names for the random effect factors
random_effect_factor_labels = RandomEffectFactorDim([:SubjectFactor, :ItemFactor])

#Names and structure of random effect terms in each regression and factor
random_effect_term_labels = DimArray([
        #Regression 1
        DimArray([
                #Factor 1
                RandomEffectTermDim([:Term1, :Term2]),
                #Factor 2
                RandomEffectTermDim([:Term1])
            ], random_effect_factor_labels),
        #Regression 2
        DimArray([
                #Factor 1
                RandomEffectTermDim([:Term1, :Term2, :Term3]),
                #Factor 2
                RandomEffectTermDim([])
            ], random_effect_factor_labels)
    ], regression_labels)

#Names and structure of random effect groups in each factor
random_effect_group_labels = DimArray([
        #Factor 1
        RandomEffectGroupDim([:f1_HealthyGroup, :f1_SickGroup]),
        #Factor 2
        RandomEffectGroupDim([:f2_AllItemsGroup])
    ], random_effect_factor_labels)

#Names and structure of random effect correlation blocks in each factor
random_effect_block_labels = DimArray([
        #Factor 1
        RandomEffectBlockDim([:f1_Block1, :f1_Block2]),
        #Factor 2
        RandomEffectBlockDim([:f2_Block1])
    ], random_effect_factor_labels)

#Names for individual random effect levels in each factor
random_effect_level_labels = DimArray([
        #Factor 1
        RandomEffectLevelDim([:Subj1, :Subj2, :Subj3, :Subj4]),
        #Factor 2
        RandomEffectLevelDim([:Item1, :Item2, :Item3])
    ], random_effect_factor_labels)

## Regression 1 ##
# 3 fixed effect terms P
# 2 Random effect factors F
#    - Factor 1: 2 groups G, 2 terms Q
#    - Factor 2: 1 group G, 1 term Q

#Fixed effect priors
r1_fixed = DimArray(
    [Normal(0, 1), Normal(0, 1), Normal(0, 1)], 
    fixed_effect_term_labels[At(regression_labels[1])]
)

#Random effect SD priors
r1_random_effect_sds = DimArray([

    #Factor 1, with 2 terms
    DimArray([
        # Group 1
        DimArray([Gamma(2, 0.1), Gamma(2, 0.1)], random_effect_term_labels[At(regression_labels[1])][At(random_effect_factor_labels[1])]),
        # Group 2
        DimArray([Gamma(2, 0.5), Gamma(2, 0.5)], random_effect_term_labels[At(regression_labels[1])][At(random_effect_factor_labels[1])]),

    ], random_effect_group_labels[At(random_effect_factor_labels[1])]),
    #Factor 2, with 1 term
    DimArray([

        # Group 1
        DimArray([Gamma(2, 0.1)], random_effect_term_labels[At(regression_labels[1])][At(random_effect_factor_labels[2])]),

    ], random_effect_group_labels[At(random_effect_factor_labels[2])]),
], random_effect_factor_labels)

## Regression 2 ##
# 5 fixed effect terms P
# 2 Random effect factors F
#    - Factor 1: 2 groups G, 3 terms Q
#    - Factor 2: 1 groups G, 0 terms Q

#Fixed effect priors
r2_fixed = DimArray(
    [Normal(0, 1), Normal(0, 1), Normal(0, 1), Normal(0, 1), Normal(0, 1)], 
    fixed_effect_term_labels[At(regression_labels[2])]
)

#Random effect SD priors
r2_random_effect_sds = DimArray([

    #Factor 1, with 3 terms
    DimArray([
        
        # Group 1
        DimArray([Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1)], random_effect_term_labels[At(regression_labels[2])][At(random_effect_factor_labels[1])]),
        # Group 2
        DimArray([Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.1)], random_effect_term_labels[At(regression_labels[2])][At(random_effect_factor_labels[1])]),

    ], random_effect_group_labels[At(random_effect_factor_labels[1])]),
    #Factor 2, with 0 terms (not used in the regression)
    DimArray([

        # Group 1
        DimArray(Gamma{Float64}[], random_effect_term_labels[At(regression_labels[2])][At(random_effect_factor_labels[2])]),

    ], random_effect_group_labels[At(random_effect_factor_labels[2])]),

], random_effect_factor_labels)


## Random effect correlation priors ##
#Factor 1: 2 groups, 5 total terms across regressions (Reg 1: 2, Reg 2: 3). Block 1 has terms 1-2, block 2 has terms 3-5.
#Factor 2: 1 group, 1 total term across regressions (Reg 1: 1, Reg 2: 0). Block 1 has has term 1.

f1_cor_priors = DimArray([
    #Group 1
    DimArray([
        #Block 1
        LKJCholesky(2, 1.0),
        #Block 2
        LKJCholesky(3, 1.0)
    ], random_effect_block_labels[At(random_effect_factor_labels[1])]),

    #Group 2
    DimArray([
        #Block 1
        nothing, #Random effects in this block assumed to be independent
        #Block 2
        LKJCholesky(3, 1.0)
    ], random_effect_block_labels[At(random_effect_factor_labels[1])]),
], random_effect_group_labels[At(random_effect_factor_labels[1])])


f2_cor_priors = DimArray([
    #Group 1
    DimArray([
        #Block 1
        LKJCholesky(1, 1.0),
    ], random_effect_block_labels[At(random_effect_factor_labels[2])])
], random_effect_group_labels[At(random_effect_factor_labels[2])])

## Random effect group assignments ##
# Factor 1: 4 subjects (2 Healthy, 2 Sick)
# Factor 2: 3 items (All 1 group)
group_assignments = DimArray([
    #Factor 1
    DimArray([:f1_HealthyGroup, :f1_HealthyGroup, :f1_SickGroup, :f1_SickGroup], random_effect_level_labels[At(random_effect_factor_labels[1])]),
    #Factor 2
    DimArray([:f2_AllItemsGroup, :f2_AllItemsGroup, :f2_AllItemsGroup], random_effect_level_labels[At(random_effect_factor_labels[2])])
], random_effect_factor_labels)

## Random effect term block assignments ##
# Regression 1
#  - Factor 1: Terms 1 and 2 in Block 1
#  - Factor 2: Term 1 in Block 1
# Regression 2
#  - Factor 1: Terms 1, 2, and 3 in Block 2
#  - Factor 2: Not used
block_assignments = DimArray([
    # Regression 1
    DimArray([
        # Factor 1: Subject. Term 1 and 2 in Block 1
        DimArray([:f1_Block1, :f1_Block1], random_effect_term_labels[At(:Regression1)][At(:SubjectFactor)]),
        # Factor 2: Item. Term 1 in Block 1
        DimArray([:f2_Block1], random_effect_term_labels[At(:Regression1)][At(:ItemFactor)])
    ], random_effect_factor_labels),

    # Regression 2
    DimArray([
        # Factor 1: Subject. Term 1, 2, 3 in Block 2
        DimArray([:f1_Block2, :f1_Block2, :f1_Block2], random_effect_term_labels[At(:Regression2)][At(:SubjectFactor)]),
        # Factor 2: Item. Not used
        DimArray(Symbol[], random_effect_term_labels[At(:Regression2)][At(:ItemFactor)])
    ], random_effect_factor_labels)
], regression_labels)

## Random effect parameterisations ##
# Factor 1: Non-centered
# Factor 2: Centered
random_effect_parameterisations = DimArray([
    #Factor 1
    NonCentered,
    #Factor 2
    Centered
], random_effect_factor_labels)

#Final structuring of information
fixed_effect_priors = DimArray([r1_fixed, r2_fixed], regression_labels)
random_effect_sd_priors = DimArray([r1_random_effect_sds, r2_random_effect_sds], regression_labels)
random_effect_correlations_cholesky_priors = DimArray([f1_cor_priors, f2_cor_priors], random_effect_factor_labels)



## Collect individual priors into multivariate priors ##
#Fixed effects
fixed_effect_priors_gathered = product_distribution([prior_p for priors_r in fixed_effect_priors for prior_p in priors_r])
#Random effect SDs
random_effect_sd_priors_gathered = product_distribution([prior_q for priors_r in random_effect_sd_priors for priors_f in priors_r for priors_g in priors_f for prior_q in priors_g])


## Initialise labels ##
labels = RegressionLabels(
    regression_labels,
    fixed_effect_term_labels,
    random_effect_factor_labels,
    random_effect_term_labels,
    random_effect_group_labels,
    random_effect_block_labels,
    random_effect_level_labels
)

## Generate indices for mapping from flattened vector to structured coefficients
(fixed_effect_indices, random_effect_sds_indices, random_effect_sds_block_indices) = generate_indices(labels, block_assignments)

## Initialise specifications and priors ##
regression_specifications = RegressionSpecifications(group_assignments, block_assignments, random_effect_parameterisations, fixed_effect_indices, random_effect_sds_indices, random_effect_sds_block_indices, labels)
priors = RegressionPrior(fixed_effect_priors_gathered, random_effect_sd_priors_gathered, random_effect_correlations_cholesky_priors, regression_specifications)


## Execute ##
coeffs = rand(priors)
total_logprob = logpdf(priors, coeffs)
fixed_effects = get_fixed_effects(coeffs)
random_effects = get_random_effects(coeffs)






################################
### TEST REGRESSION FUNCTION ###
################################
using Random
Random.seed!(123)

# 1. Setup Observation IDs
N = 12

observation_labels = DimArray([
    ObservationDim(1:N),
    ObservationDim(1:N)
], regression_labels)

# Mapping observations to Subject (1-4) and Item (1-3)


subj_idx = repeat([1, 2, 3, 4], inner=3)  # [1,1,1, 2,2,2, 3,3,3, 4,4,4]
item_idx = repeat([1, 2, 3], outer=4)          # [1,2,3, 1,2,3, 1,2,3, 1,2,3]



# 2. Regression 1 Data (3 Fixed Effects)
# X1: Intercept + 2 Predictors
X1 = DimArray(hcat(ones(N), randn(N, 2)), (observation_labels[At(regression_labels[1])], fixed_effect_term_labels[At(regression_labels[1])]))

# Random Effect Design Matrices for Regression 1
# Factor 1 (Subject) uses 2 terms (e.g., Intercept and first Predictor)
Z1_f1 = X1[:, 1:2]
# Factor 2 (Item) uses 1 term (e.g., Intercept)
Z1_f2 = X1[:, 1:1]

# 3. Regression 2 Data (5 Fixed Effects)
# X2: Intercept + 4 Predictors
X2 = DimArray(hcat(ones(N), randn(N, 4)), (observation_labels[At(regression_labels[2])], fixed_effect_term_labels[At(regression_labels[2])]))

# Random Effect Design Matrices for Regression 2
# Factor 1 (Subject) uses 3 terms (e.g., Intercept and first two Predictors)
Z2_f1 = X2[:, 1:3]
# Factor 2 (Item) uses 0 terms (As specified in your evaluation: empty vector)
Z2_f2 = X2[:, []]

# Summarize for the model
fixed_effect_design_matrices = DimArray([X1, X2], regression_labels)
random_effect_design_matrices = DimArray([
    DimArray([Z1_f1, Z1_f2], random_effect_factor_labels), # Reg 1: Factor 1, Factor 2
    DimArray([Z2_f1, Z2_f2], random_effect_factor_labels)  # Reg 2: Factor 1, Factor 2
], regression_labels)

level_assignments = DimArray([
    #Regression 1
    DimArray([subj_idx item_idx], (observation_labels[At(regression_labels[1])], random_effect_factor_labels)),
    #Regression 2
    DimArray([subj_idx item_idx], (observation_labels[At(regression_labels[2])], random_effect_factor_labels))
    ], regression_labels)

predictors = RegressionPredictors(fixed_effect_design_matrices, random_effect_design_matrices, level_assignments)

coefficients = rand(priors)

outcomes = linear_combination(predictors, coefficients)

new_predictors = update_predictor(predictors, zeros(N), :Term3, :Regression1)
new_predictors = update_predictor(new_predictors, zeros(N), :Term2)

update_predictor!(new_predictors, ones(N), :Term2, :Regression2)
update_predictor!(new_predictors, ones(N), :Term4)

new_predictors = update_level_assignments(new_predictors, ones(N), :ItemFactor, :Regression1)
new_predictors = update_level_assignments(new_predictors, ones(N) .+ 1, :SubjectFactor)

update_level_assignments!(new_predictors, ones(N) .+ 2, :ItemFactor, :Regression2)
update_level_assignments!(new_predictors, ones(N) .+ 3, :SubjectFactor)












##########################
### TEST TURING MODELS ###
##########################

## 1. Simple regression ##
model = simple_regression(predictors, priors)

chain = sample(model, Prior(), 1000, chain_type=VNChain)




## 2. Multistep regression ##

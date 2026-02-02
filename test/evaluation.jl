
include(joinpath("..", "src", "BayesianRegressionModels.jl"))


##########################
### INPUT FROM FORMULA ###
##########################

# @formula Performance ~ 1 + Age_first * Treatment + Age_second + (1 + Treatment | Subject) + (1 + Age_first | Experimenter), data = df1
    # @expansion PolynomialExpansion() Age [Age_first, Age_second]       <- expand the Age variable into Age_first and Age_second using polynomial expansion (only for this regression)

# @formula Accuracy ~ 1 + Age * BMI + 
#                     @interaction Age : BMI MaxOperator() label= :max_age_BMI +         <- Add a custom interaction operator (here one that selects the maximum), and give it a custom label
#                     (1 + Age * BMI | Subject), data = df2

# @group Subject by = ClinicalGroup
# @block Subject Block1 Performance = [Treatment] Accuracy = [Age:BMI]
# @block Subject Block2 Accuracy = [Age, BMI]

#NOTES:
#- If an expansion is not specified for a given variable, it just uses the IdentityExpansion(), unless it is a categoricla variable, in which case it uses DummyCodeExpansion()
#- If an expansion is specified, but the original variable is used in the formula, then all expanded terms are included (e.g., if Age is used in the first regressions, it will include both Age_first and Age_second - the same for categorical variables)
#- Expansions can be added for a specific regression, or globally for all regressions (not shown here)

########################
### 1 SPECIFY LABELS ###
########################

# 1. Labels for the different regressions
regression_labels = RegressionDim([:Performance, :Accuracy])


# 2. Labels for the levels of the categorical variables
categorical_level_labels = DimArray([
        
    #Variable 1: Treatment
    CategoricalLevelDim([:Low, :Medium, :High]),

    #Variable 2: Subject
    CategoricalLevelDim([:Subj1, :Subj2, :Subj3, :Subj4]),

    #Variable 3: Experimenter
    CategoricalLevelDim([:Exp1, :Exp2, :Exp3]),

    #Variable 4: ClinicalGroup
    CategoricalLevelDim([:ControlGroup, :TargetGroup])

], CategoricalVariableDim([:Treatment, :Subject, :Experimenter, :ClinicalGroup]))


# 3. Labels of basis terms (no interactions)
basis_term_labels = DimArray([

    #Regression 1: Performance
    BasisTermDim([
        :Intercept, :Age_first, :Age_second, :Treatment_Medium, :Treatment_High
    ]),

    #Regression 2: Accuracy
    BasisTermDim([
        :Intercept, :Age, :BMI
    ]),

], regression_labels)


# 4. Labels for the fixed effect terms in the design matrices
fixed_effect_term_labels = DimArray([
    
    # Regression 1: Performance
    FixedEffectTermDim([
        :Intercept, :Age_first, :Age_second, :Treatment_Medium, :Treatment_High, :Age_x_Treatment_Medium, :Age_x_Treatment_High
    ]),
    
    # Regression 2: Accuracy
    FixedEffectTermDim([
        :Intercept, :Age, :BMI, :Age_x_BMI, :max_age_BMI,
    ])

], regression_labels)


# 5. Labels for the random effect factors
random_effect_factor_labels = RandomEffectFactorDim([:Subject, :Experimenter])


# 6. Labels for the random effect terms in the design matrices
random_effect_term_labels = DimArray([
    
    #Regression 1: Performance
    DimArray([

        #Factor 1: Subject
        RandomEffectTermDim([:Intercept, :Treatment_Medium, :Treatment_High]),
        #Factor 2: Experimenter
        RandomEffectTermDim([:Intercept, :Age_first])

    ], random_effect_factor_labels),
    
    #Regression 2: Accuracy
    DimArray([

        #Factor 1: Subject
        RandomEffectTermDim([:Intercept, :Age, :BMI, :Age_x_BMI]),
        #Factor 2: Experimenter (not present in the formula)
        RandomEffectTermDim(Symbol[]) 

    ], random_effect_factor_labels)
    
], regression_labels)


# 7. Labels and structure of random effect groups in each factor
random_effect_group_labels = DimArray([

        #Factor 1: Subject (get the levels from the categorical levels container)
        RandomEffectGroupDim(parent(categorical_level_labels[At(:ClinicalGroup)])),

        #Factor 2: Experimenter (no group specified, so use a single group)
        RandomEffectGroupDim([:SingleGroup])

    ], random_effect_factor_labels)


# 8. Labels and structure of random effect correlation blocks in each factor
random_effect_block_labels = DimArray([

        #Factor 1: Subject (ResidualBlock has all unnasigned terms in the factor)
        RandomEffectBlockDim([:Block1, :Block2, :ResidualBlock]),

        #Factor 2: Experimenter (no blocks specified, so use a single block)
        RandomEffectBlockDim([:SingleBlock])

    ], random_effect_factor_labels)


# 9. Labels for observations in each regression
observation_labels = DimArray([

    #Regression 1: Performance (df1 has 12 rows)
    ObservationDim(1:12),
    #Regression 2: Accuracy (df2 has 16 rows)
    ObservationDim(1:16)

], regression_labels)


# 10. Create labels object #
labels = RegressionLabels(
    regression_labels,
    categorical_level_labels,
    basis_term_labels,
    fixed_effect_term_labels,
    random_effect_factor_labels,
    random_effect_term_labels,
    random_effect_group_labels,
    random_effect_block_labels,
    observation_labels
)



###############################
### 2 CREATE SPECIFICATIONS ###
###############################

## 1. Random effect group assignments ##
group_assignments = DimArray([

    #Factor 1: Subject
    DimArray([:ControlGroup, :ControlGroup, :TargetGroup, :TargetGroup], categorical_level_labels[At(:Subject)]),

    #Factor 2: Experimenter
    DimArray([:SingleGroup, :SingleGroup, :SingleGroup], categorical_level_labels[At(:Experimenter)])

], random_effect_factor_labels)

## 2. Random effect term block assignments ##
block_assignments = DimArray([
    # Regression 1: Performance
    DimArray([
        
        # Factor 1: Subject. Intercept is Residual, Treatment_Medium and Treatment_High are Block1
        DimArray([:ResidualBlock, :Block1, :Block1], random_effect_term_labels[At(:Performance)][At(:Subject)]),
        
        # Factor 2: Experimenter. Both Intercept and Age_first is in a single block
        DimArray([:SingleBlock, :SingleBlock], random_effect_term_labels[At(:Performance)][At(:Experimenter)])

    ], random_effect_factor_labels),

    # Regression 2: Accuracy
    DimArray([
        
        # Factor 1: Subject. Intercept is Residual, Age and BMI main effects are in Block2, Age:BMI is in Block1
        DimArray([:ResidualBlock, :Block2, :Block2, :Block1], random_effect_term_labels[At(:Accuracy)][At(:Subject)]),
        
        # Factor 2: Experimenter. Not used (Empty)
        DimArray(Symbol[], random_effect_term_labels[At(:Accuracy)][At(:Experimenter)])

    ], random_effect_factor_labels)
], regression_labels)


## 3. Random effect geometries ##
random_effect_geometries = DimArray([
    
    #Factor 1: Subject
    NonCentered,

    #Factor 2: Experimenter
    Centered

], random_effect_factor_labels)


## 4. Generate indices for mapping from flattened vector to structured coefficients ##
(fixed_effect_indices, random_effect_sds_indices, random_effect_sds_block_indices) = generate_indices(labels, block_assignments)


## 5. Create specifications object ##
specifications = RegressionSpecifications(group_assignments, block_assignments, random_effect_geometries, fixed_effect_indices, random_effect_sds_indices, random_effect_sds_block_indices, labels)



########################
### 3 SPECIFY PRIORS ###
########################

## 1. Fixed effect priors ##
fixed_effect_priors = DimArray([

    # Regression 1: Performance (7 terms: Intercept, Age_first, Age_second, 2 Treatment levels, 2 Age_first:Treatment interactions)
    DimArray(
        fill(Normal(0, 1), 7), 
        fixed_effect_term_labels[At(:Performance)]),
    
    # Regression 2: Accuracy (5 terms: Intercept, Age, BMI, Age:BMI product, Max(Age, BMI))
    DimArray(
        fill(Normal(0, 1), 5),
        fixed_effect_term_labels[At(:Accuracy)])

], regression_labels)


## 2. Random effect standard deviation priors ##
random_effect_sd_priors = DimArray([

    # Regression 1: Performance
    DimArray([

        # Factor 1: Subject (3 terms: Intercept, 2 Treatment)
        DimArray([

            # Group 1: ControlGroup:
            DimArray(
                [Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1)], 
                random_effect_term_labels[At(:Performance)][At(:Subject)]),

            # Group 2: TargetGroup
            DimArray(
                [Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.5)],
                random_effect_term_labels[At(:Performance)][At(:Subject)])

        ], random_effect_group_labels[At(:Subject)]),

        # Factor 2: Experimenter (2 terms: Intercept, Age_first)
        DimArray([
            
            #Group 1: SingleGroup
            DimArray(
                [Gamma(2, 0.1), Gamma(2, 0.1)], 
                random_effect_term_labels[At(:Performance)][At(:Experimenter)])

        ], random_effect_group_labels[At(:Experimenter)])

    ], random_effect_factor_labels),

    # Regression 2: Accuracy
    DimArray([

        # Factor 1: Subject (4 terms: Intercept, Age, BMI, Age_x_BMI)
        DimArray([

            # Group 1: ControlGroup
            DimArray(
                [Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1)], 
                random_effect_term_labels[At(:Accuracy)][At(:Subject)]),

            # Group 2: TargetGroup
            DimArray(
                [Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.5)], 
                random_effect_term_labels[At(:Accuracy)][At(:Subject)])

        ], random_effect_group_labels[At(:Subject)]),

        # Factor 2: Experimenter (0 terms: not used)
        DimArray([

            # Group 1: SingleGroup
            DimArray(Gamma{Float64}[], 
            random_effect_term_labels[At(:Accuracy)][At(:Experimenter)])

        ], random_effect_group_labels[At(:Experimenter)])
        
    ], random_effect_factor_labels)

], regression_labels)


## 3. Random effect correlation LKJCholesky priors ##
random_effect_correlation_LKJcholesky_priors = DimArray([

    # Factor 1: Subject
    DimArray([
        
        # Group 1: ControlGroup
        DimArray([

            # Block 1: Block1 (3 terms: Interaction/Slopes correlation)
            LKJCholesky(3, 1.0), 

            # Block 2: Block2 (2 terms: main effects correlation)
            LKJCholesky(2, 1.0),

            # Block 3: ResidualBlock (2 terms: the intercepts)
            nothing #no correlation

        ], random_effect_block_labels[At(:Subject)]),

        # Group 2: TargetGroup
        DimArray([

            # Block 1: Block1 (3 terms: Interaction/Slopes correlation)
            LKJCholesky(3, 1.0),
            
            # Block 2: Block2 (2 terms: main effects correlation)
            LKJCholesky(2, 1.0),
            
            # Block 3: ResidualBlock (2 terms: the intercepts)
            nothing #No correlation

        ], random_effect_block_labels[At(:Subject)])

    ], random_effect_group_labels[At(:Subject)]),

    # Factor 2: Experimenter
    DimArray([
        
        # Group 1: SingleGroup
        DimArray([

            # Block 1: SingleBlock (2 terms: Intercept and Age_first)
            LKJCholesky(2, 1.0)

        ], random_effect_block_labels[At(:Experimenter)])

    ], random_effect_group_labels[At(:Experimenter)])

], random_effect_factor_labels)


## 4. Collect individual priors into multivariate priors ##
#Fixed effects
fixed_effect_priors_gathered = product_distribution([prior_p for priors_r in fixed_effect_priors for prior_p in priors_r])
#Random effect SDs
random_effect_sd_priors_gathered = product_distribution([prior_q for priors_r in random_effect_sd_priors for priors_f in priors_r for priors_g in priors_f for prior_q in priors_g])

## 5. Make final prior object ##
priors = RegressionPrior(fixed_effect_priors_gathered, random_effect_sd_priors_gathered, random_effect_correlation_LKJcholesky_priors, specifications)




###########################
### 4 CREATE PREDICTORS ###
###########################

## Set seed ##
using Random
Random.seed!(123)

## - For regression 1 - ##
## 1. Get number of observations ##
n_observations_r1 = length(labels.observations[At(:Performance)])


## 2. Create basis matrix ##
#Generate treatment cateogrical data (3 levels)
treatment_r1 = rand(1:3, n_observations_r1)
age_r1 = randn(n_observations_r1)

#Create basis matrix
basis_matrix_r1 = DimArray(hcat(
    
    # Variable 1: Intercept
    ones(n_observations_r1), 

    # Variable 2: Age_first
    age_r1,    

    # Variable 3: Age_second
    age_r1.^2,    

    # Variable 4: Treatment_Medium  
    treatment_r1 .== 2,  

    # Variable 5: Treatment_High
    treatment_r1 .== 3   

), (observation_labels[At(:Performance)], basis_term_labels[At(:Performance)]))


## 3. Create fixed effects design matrix ##
fixed_effects_design_matrix_r1 = DimArray(hcat(

    #The basis terms are all there
    parent(basis_matrix_r1), 

    #The Age_first:Treatment_Medium interaction
    basis_matrix_r1[:, At(:Age_first)] .* basis_matrix_r1[:, At(:Treatment_Medium)], 
    
    #The Age_first:Treatment_High interaction
    basis_matrix_r1[:, At(:Age_first)] .* basis_matrix_r1[:, At(:Treatment_High)]
    
), (observation_labels[At(:Performance)], fixed_effect_term_labels[At(:Performance)]))


## 4. Create random effect design matrices ##
random_effect_design_matrices_r1 = DimArray([
    
    # Factor 1: Subject (3 terms)
    basis_matrix_r1[:, At([:Intercept, :Treatment_Medium, :Treatment_High])],
    
    # Factor 2: Experimenter (2 terms)
    basis_matrix_r1[:, At([:Intercept, :Age_first])]

], random_effect_factor_labels)


## 5. Create level assignments for random effects ##
random_effect_level_assignments_r1 = DimArray(hcat( 

    #Variable 2: Subject (4 levels)
    repeat([1, 2, 3, 4], inner=3),  # [1,1,1, 2,2,2, 3,3,3, 4,4,4]
    
    #Variable 3: Experimenter (3 levels)
    repeat([1, 2, 3], outer=4),     # [1,2,3, 1,2,3, 1,2,3, 1,2,3]
    
    ), (observation_labels[At(:Performance)], CategoricalVariableDim([:Subject, :Experimenter])))


## 6. Create interaction recipes ##
fixed_effects_interaction_recipes_r1 = [
        nothing, nothing, nothing, nothing, nothing, # Terms 1-5 are basis terms
        InteractionRecipe([2, 4], MultiplicationOperator()),       # Term 6: Age_first (2) * Treatment_Medium (4)
        InteractionRecipe([2, 5], MultiplicationOperator())        # Term 7: Age_first (2) * Treatment_High (5)
    ]

random_effects_interaction_recipes_r1 = [
        # Factor 1: Subject (3 terms, no interactions)
        Union{Nothing, InteractionRecipe{MultiplicationOperator}}[nothing for _ in 1:3],                   

        # Factor 2: Experimenter (2 terms, no interactions)
        Union{Nothing, InteractionRecipe{MultiplicationOperator}}[nothing for _ in 1:2]       
    ]


## 7. Create info for each term ##
terms_info_r1 = (

    Intercept = TermInfo(
        basis_expansion_type = IdentityExpansion(),    
        basis_matrix_indices = [1], 
        fixed_effects_indices = [1],
        random_effects_indices = [(1, [1]), (2, [1])],
        level_assignments_idx = 0,
        dependent_interaction_indices = DependentInteractionIndices(Int[], Tuple{Int, Int}[])),
    
    Age = TermInfo(
        basis_expansion_type = PolynomialExpansion(),
        basis_matrix_indices = [2, 3], 
        fixed_effects_indices = [2, 3],
        random_effects_indices = [(2, [2, 0])],
        level_assignments_idx = 0,
        dependent_interaction_indices = DependentInteractionIndices([6, 7], Tuple{Int, Int}[]) 
    ),
    
    Treatment = TermInfo(
        basis_expansion_type = DummyCodeExpansion(),
        basis_matrix_indices = [4, 5], 
        fixed_effects_indices = [4, 5],
        random_effects_indices = [(1, [2, 3])],
        level_assignments_idx = 0, 
        dependent_interaction_indices = DependentInteractionIndices([6, 7], Tuple{Int, Int}[])
    ),

    # Subject info
    Subject = TermInfo(
        basis_expansion_type = DummyCodeExpansion(),
        basis_matrix_indices = Int[], 
        fixed_effects_indices = Int[],
        random_effects_indices =Tuple{Int, Vector{Int}}[],
        level_assignments_idx = 1,  
        dependent_interaction_indices = DependentInteractionIndices(Int[], Tuple{Int, Int}[])
    ),

    # Experimenter info
    Experimenter = TermInfo(
        basis_expansion_type = DummyCodeExpansion(),
        basis_matrix_indices = Int[], 
        fixed_effects_indices = Int[],
        random_effects_indices =Tuple{Int, Vector{Int}}[],
        level_assignments_idx = 2,  
        dependent_interaction_indices = DependentInteractionIndices(Int[], Tuple{Int, Int}[])
    )
)

## 8. Create container for marking interaction effects for being updated ##
interaction_udpate_markers_r1 = InteractionUpdateMarkers(BitSet(), [BitSet() for _ in 1:length(random_effect_design_matrices_r1)])

## 9. Instantiate predictors ##
predictors_r1 = RegressionPredictors(
    basis_matrix_r1,
    fixed_effects_design_matrix_r1,
    random_effect_design_matrices_r1,
    random_effect_level_assignments_r1,
    terms_info_r1,
    fixed_effects_interaction_recipes_r1,
    random_effects_interaction_recipes_r1,
    interaction_udpate_markers_r1
)

## - For regression 2 - ##
## 1. Get number of observations ##
n_observations_r2 = length(labels.observations[At(:Accuracy)])

## 2. Create basis matrix ##
basis_matrix_r2 = DimArray(hcat(
    
    # Variable 1: Intercept
    ones(n_observations_r2), 

    # Variable 2: Age
    randn(n_observations_r2),    

    # Variable 3: BMI
    randn(n_observations_r2)   

), (observation_labels[At(:Accuracy)], basis_term_labels[At(:Accuracy)]))


## 3. Create fixed effects design matrix ##
fixed_effects_design_matrix_r2 = DimArray(hcat(

    # Terms 1-3: Intercept, Age, BMI
    parent(basis_matrix_r2), 

    # Term 4: Age:BMI interaction
    basis_matrix_r2[:, At(:Age)] .* basis_matrix_r2[:, At(:BMI)],

    # Term 5: max(Age, BMI)
    max.(basis_matrix_r2[:, At(:Age)], basis_matrix_r2[:, At(:BMI)])
    
), (observation_labels[At(:Accuracy)], fixed_effect_term_labels[At(:Accuracy)]))


## 4. Create random effect design matrices ##
random_effect_design_matrices_r2 = DimArray([
    
    # Factor 1: Subject (4 terms: Intercept, Age, BMI, Age_x_BMI)
    fixed_effects_design_matrix_r2[:, At([:Intercept, :Age, :BMI, :Age_x_BMI])],
    
    # Factor 2: Experimenter (No terms)
    DimArray(Matrix{Float64}(undef, n_observations_r2, 0), (observation_labels[At(:Accuracy)], RandomEffectTermDim(Symbol[])))

], random_effect_factor_labels)


## 5. Create level assignments for random effects ##
random_effect_level_assignments_r2 = DimArray(hcat( 

    #Variable 2: Subject (4 levels)
    repeat([1, 2, 3, 4], inner=4),  # [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,44,]
    
    ), (observation_labels[At(:Accuracy)], CategoricalVariableDim([:Subject])))



## 6. Create interaction recipes ##
fixed_effects_interaction_recipes_r2 = [nothing, nothing, nothing, 
    InteractionRecipe([2, 3], MultiplicationOperator()), # 4: Age (2) * BMI (3)
    InteractionRecipe([2, 3], MaxOperator())             # 5: max(Age (2), BMI (3))
]  

random_effects_interaction_recipes_r2 = [
    # Factor 1: Subject (4 terms, 3 main effects and 1 interaction)
    [nothing, nothing, nothing, 
    InteractionRecipe([2, 3], MultiplicationOperator()), # 4: Age (2) * BMI (3)
    ], 

    # Factor 2: Experimenter (No terms)
    Vector{Union{Nothing, InteractionRecipe{MultiplicationOperator}}}()
]

## 7. Create info for each term ##
terms_info_r2 = (

    Intercept = TermInfo(
        basis_expansion_type = IdentityExpansion(),
        basis_matrix_indices = [1], 
        fixed_effects_indices = [1],
        random_effects_indices = [(1, [1])],
        level_assignments_idx = 0,
        dependent_interaction_indices = DependentInteractionIndices(Int[], Tuple{Int, Int}[])
    ),
    
    Age = TermInfo(
        basis_expansion_type = IdentityExpansion(),
        basis_matrix_indices = [2], 
        fixed_effects_indices = [2],
        random_effects_indices = [(1, [2])],
        level_assignments_idx = 0,
        # Age is part of the interaction in the fixed effects design matrix (col 4) and in the Subject random effects design matrix (Factor 1, col 4)
        dependent_interaction_indices = DependentInteractionIndices([4, 5], [(1, 4)]) 
    ),
    
    BMI = TermInfo(
        basis_expansion_type = IdentityExpansion(),
        basis_matrix_indices = [3], 
        fixed_effects_indices = [3],
        random_effects_indices = [(1, [3])],
        level_assignments_idx = 0,
        # BMI is part of the interaction in Fixed (col 4) and Subject Random (Factor 1, col 4)
        dependent_interaction_indices = DependentInteractionIndices([4,5], [(1, 4)])
    ),

    Subject = TermInfo(
        basis_expansion_type = DummyCodeExpansion(),
        basis_matrix_indices = Int[], 
        fixed_effects_indices = Int[],
        random_effects_indices = Tuple{Int, Vector{Int}}[],
        level_assignments_idx = 1,  
        dependent_interaction_indices = DependentInteractionIndices(Int[], Tuple{Int, Int}[])
    ),
)

## 8. Create container for marking interaction effects for being updated ##
interaction_udpate_markers_r2 = InteractionUpdateMarkers(BitSet(), [BitSet() for _ in 1:length(random_effect_design_matrices_r2)])

## 9. Instantiate predictors ##
predictors_r2 = RegressionPredictors(
    basis_matrix_r2,
    fixed_effects_design_matrix_r2,
    random_effect_design_matrices_r2,
    random_effect_level_assignments_r2,
    terms_info_r2,
    fixed_effects_interaction_recipes_r2,
    random_effects_interaction_recipes_r2,
    interaction_udpate_markers_r2
)

## - Final predictors object - ##
predictors = DimArray([predictors_r1, predictors_r2], regression_labels)


#################
### 5 TESTING ###
#################

## 1. Test rand() ##
coefficients = rand(priors)

## 2. Test logprob() ##
total_logprob = logpdf(priors, coefficients)

## 3. Test extraction of coefficients ##
fixed_effects = get_fixed_effects(coefficients)
random_effects = get_random_effects(coefficients)

## 4. Test updating predictors ##
update_variables!(predictors, (:Age, :Treatment), (randn(12), rand(1:3, 12)), :Performance)

## 5. Test linear_combination ##
outcomes = linear_combination(predictors, coefficients)

## 6. Test Turing models ##

# 6.1. Simple regression #
model = simple_regression(predictors, [nothing], priors, nothing)
chain = sample(model, Prior(), 1000, chain_type=VNChain)

# 6.2. Multistep regression #

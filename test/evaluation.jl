include(joinpath("..", "src", "BayesianRegressionModels.jl"))

##########################
### INPUT FROM FORMULA ###
##########################

## NOTE: THIS IS AN INITIAL SUGGESTION FOR A FORMULA SYNTAX. THERE CANBEMUCH BETTER WAYS TO DO IT. THE SUGGESTION IS THAT IT DECOMPOSES INTO OPERATIONS - SEE BELOW

# @formula BMI = 1, data = dBMI, measurement_error(sd = 1, observations = BMI_measured)  <- this is to generate a latent BMI variable, where the BMI_measured is a noisy measurement of it.

# @formula performance_mean = 1 + Age_first * Treatment + Age_second + (1 + Treatment | Subject) + (1 + Age_first | Experimenter), data = dpmean,
#       @expansion PolynomialExpansion(2) Age [Age_first, Age_second]       <- expand the Age variable into Age_first and Age_second using polynomial expansion (only for this regression)

# @formula performance_sd = 1 + Age * BMI + 
#                     @interaction Age : BMI MaxOperator() label= :max_age_BMI +         <- Add a custom interaction operator (here one that selects the maximum), and give it a custom label
#                     (1 + Age * BMI | Subject), data = dpsd

# @likelihood Performance ~ Normal(performance_mean, performance_sd)

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
regression_labels = RegressionDim([:BMI, :performance_mean, :performance_sd])


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

        #Regression 1: BMI
        BasisTermDim([
            :Intercept    
        ]),

        #Regression 2: performance_mean
        BasisTermDim([
            :Intercept, :Age_first, :Age_second, :Treatment_Medium, :Treatment_High
        ]),

        #Regression 3: performance_sd
        BasisTermDim([
            :Intercept, :Age, :BMI
        ]),
    
    ], regression_labels)


# 4. Labels for the fixed effect terms in the design matrices
fixed_effect_term_labels = DimArray([

        # Regression 1: BMI  
        FixedEffectTermDim([
            :Intercept
        ]),    

        # Regression 2: performance_mean
        FixedEffectTermDim([
            :Intercept, :Age_first, :Age_second, :Treatment_Medium, :Treatment_High, :Age_x_Treatment_Medium, :Age_x_Treatment_High
        ]),

        # Regression 3: performance_sd
        FixedEffectTermDim([
            :Intercept, :Age, :BMI, :Age_x_BMI, :max_age_BMI,
        ])
        
    ], regression_labels)


# 5. Labels for the random effect factors
random_effect_factor_labels = RandomEffectFactorDim([:Subject, :Experimenter])


# 6. Labels for the random effect terms in the design matrices
random_effect_term_labels = DimArray([

        #Regression 1: BMI
        DimArray([
                
            # Factor 1: Subject
            RandomEffectTermDim(Symbol[]),
            # Factor 2: Experimenter
            RandomEffectTermDim(Symbol[])
        
        ], random_effect_factor_labels),

        #Regression 1: performance_mean
        DimArray([

            #Factor 1: Subject
            RandomEffectTermDim([:Intercept, :Treatment_Medium, :Treatment_High]),
            #Factor 2: Experimenter
            RandomEffectTermDim([:Intercept, :Age_first])
                
        ], random_effect_factor_labels),

        #Regression 2: performance_sd
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


# 9. Labels for outcomes in each regression
outcome_labels = DimArray([

        #Regression 1: BMI
        OutcomeDim(1:12),

        #Regression 2: performance_mean
        OutcomeDim(1:12),

        #Regression 2: performance_sd
        OutcomeDim(1:12)
        
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
    outcome_labels
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
        
        # Regression 1: BMI
        DimArray([

                # Factor 1: Subject. 
                DimArray(Symbol[], random_effect_term_labels[At(:BMI)][At(:Subject)]),

                # Factor 2: Experimenter.
                DimArray(Symbol[], random_effect_term_labels[At(:BMI)][At(:Experimenter)])

        ], random_effect_factor_labels),

        # Regression 2: performance_mean
        DimArray([

                # Factor 1: Subject. Intercept is Residual, Treatment_Medium and Treatment_High are Block1
                DimArray([:ResidualBlock, :Block1, :Block1], random_effect_term_labels[At(:performance_mean)][At(:Subject)]),

                # Factor 2: Experimenter. Both Intercept and Age_first is in a single block
                DimArray([:SingleBlock, :SingleBlock], random_effect_term_labels[At(:performance_mean)][At(:Experimenter)])

        ], random_effect_factor_labels),

        # Regression 3: performance_sd
        DimArray([

                # Factor 1: Subject. Intercept is Residual, Age and BMI main effects are in Block2, Age:BMI is in Block1
                DimArray([:ResidualBlock, :Block2, :Block2, :Block1], random_effect_term_labels[At(:performance_sd)][At(:Subject)]),

                # Factor 2: Experimenter. Not used (Empty)
                DimArray(Symbol[], random_effect_term_labels[At(:performance_sd)][At(:Experimenter)])
                
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

        #Regression 1: BMI (1 term: intercept)
        DimArray(
            fill(Normal(0, 1), 1),
            fixed_effect_term_labels[At(:BMI)]),

        # Regression 2: performance_mean (7 terms: Intercept, Age_first, Age_second, 2 Treatment levels, 2 Age_first:Treatment interactions)
        DimArray(
            fill(Normal(0, 1), 7),
            fixed_effect_term_labels[At(:performance_mean)]),

        # Regression 3: performance_sd (5 terms: Intercept, Age, BMI, Age:BMI product, Max(Age, BMI))
        DimArray(
            fill(Normal(0, 1), 5),
            fixed_effect_term_labels[At(:performance_sd)])
    
    ], regression_labels)


## 2. Random effect standard deviation priors ##
random_effect_sd_priors = DimArray([

    # Regression 1: BMI
    DimArray([

        # Factor 1: Subject (0 terms)
        DimArray([

            # Group 1: ControlGroup
            DimArray(
                Gamma{Float64}[],
                random_effect_term_labels[At(:BMI)][At(:Subject)]),

            # Group 2: TargetGroup
            DimArray(
                Gamma{Float64}[],
                random_effect_term_labels[At(:BMI)][At(:Subject)])
        
        ], random_effect_group_labels[At(:Subject)]),

        # Factor 2: Experimenter (2 terms: Intercept, Age_first)
        DimArray([

            #Group 1: SingleGroup
            DimArray(
                Gamma{Float64}[],
                random_effect_term_labels[At(:BMI)][At(:Experimenter)])
                    
        ], random_effect_group_labels[At(:Experimenter)])
                        
    ], random_effect_factor_labels),


    # Regression 2: performance_mean
    DimArray([

        # Factor 1: Subject (3 terms: Intercept, 2 Treatment)
        DimArray([

            # Group 1: ControlGroup:
            DimArray(
                [Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1)],
                random_effect_term_labels[At(:performance_mean)][At(:Subject)]),

            # Group 2: TargetGroup
            DimArray(
                [Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.5)],
                random_effect_term_labels[At(:performance_mean)][At(:Subject)])
                
        ], random_effect_group_labels[At(:Subject)]),

        # Factor 2: Experimenter (2 terms: Intercept, Age_first)
        DimArray([

            #Group 1: SingleGroup
            DimArray(
                [Gamma(2, 0.1), Gamma(2, 0.1)],
                random_effect_term_labels[At(:performance_mean)][At(:Experimenter)])
                    
        ], random_effect_group_labels[At(:Experimenter)])
                        
    ], random_effect_factor_labels),

    # Regression 3: performance_sd
    DimArray([

        # Factor 1: Subject (4 terms: Intercept, Age, BMI, Age_x_BMI)
        DimArray([

            # Group 1: ControlGroup
            DimArray(
                [Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1), Gamma(2, 0.1)],
                random_effect_term_labels[At(:performance_sd)][At(:Subject)]),

            # Group 2: TargetGroup
            DimArray(
                [Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.5), Gamma(2, 0.5)],
                random_effect_term_labels[At(:performance_sd)][At(:Subject)])
                    
        ], random_effect_group_labels[At(:Subject)]),

        # Factor 2: Experimenter (0 terms: not used)
        DimArray([

            # Group 1: SingleGroup
            DimArray(
                Gamma{Float64}[],
                random_effect_term_labels[At(:performance_sd)][At(:Experimenter)])
                    
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
## 1. Get number of outcomes ##
n_outcomes_r1 = length(labels.outcomes[At(:BMI)])

## 2. Create basis matrix ##
#Create basis matrix
basis_matrix_r1 = DimArray(hcat(

        # Variable 1: Intercept
        ones(n_outcomes_r1),

       ), (outcome_labels[At(:BMI)], basis_term_labels[At(:BMI)]))


## 3. Create fixed effects design matrix ##
fixed_effects_design_matrix_r1 = DimArray(hcat(

    #The basis terms are all there
    parent(basis_matrix_r1),

    ), 

(outcome_labels[At(:BMI)], fixed_effect_term_labels[At(:BMI)]))

## 4. Create random effect design matrices ##
random_effect_design_matrices_r1 = DimArray([

    # Factor 1: Subject (0 terms)
    DimArray(Matrix{Float64}(undef, n_outcomes_r1, 0), (outcome_labels[At(:BMI)], RandomEffectTermDim(Symbol[]))),

    # Factor 2: Experimenter (0 terms)
    DimArray(Matrix{Float64}(undef, n_outcomes_r1, 0), (outcome_labels[At(:BMI)], RandomEffectTermDim(Symbol[])))
        
], random_effect_factor_labels)

## 5. Create level assignments for random effects ##
random_effect_level_assignments_r1 = DimArray(

    #No random effect factors
    Matrix{Int}(undef, n_outcomes_r1, 0), 

    (outcome_labels[At(:BMI)], CategoricalVariableDim(Symbol[]))
)

## 6. Create interaction recipes ##
fixed_effects_interaction_recipes_r1 = [
    nothing, #No interactions 
]

random_effects_interaction_recipes_r1 = DimArray([
        # Factor 1: Subject (0 terms)
        Vector{Union{Nothing,InteractionRecipe{MultiplicationOperator}}}(),

        # Factor 2: Experimenter (0 terms)
        Vector{Union{Nothing,InteractionRecipe{MultiplicationOperator}}}()
    ], random_effect_factor_labels)

## 7. Create info for each term ##
terms_info_r1 = (
    
    #Intercept info
    Intercept=TermInfo(
        basis_expansion_type=IdentityExpansion(),
        basis_matrix_indices=[1],
        fixed_effects_indices=[1],
        random_effects_indices=Tuple{Int,Vector{Int}}[],
        level_assignments_idx=0,
        dependent_interaction_indices=DependentInteractionIndices(Int[], Tuple{Int,Int}[])), 

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
n_outcomes_r2 = length(labels.outcomes[At(:performance_mean)])

## 2. Create basis matrix ##
#Generate treatment cateogrical data (3 levels)
treatment_r2 = rand(1:3, n_outcomes_r2)
age_r2 = randn(n_outcomes_r2)

#Create basis matrix
basis_matrix_r2 = DimArray(hcat(

        # Variable 1: Intercept
        ones(n_outcomes_r2),

        # Variable 2: Age_first
        age_r2,

        # Variable 3: Age_second
        age_r2 .^ 2,

        # Variable 4: Treatment_Medium  
        treatment_r2 .== 2,

        # Variable 5: Treatment_High
        treatment_r2 .== 3), (outcome_labels[At(:performance_mean)], basis_term_labels[At(:performance_mean)]))


## 3. Create fixed effects design matrix ##
fixed_effects_design_matrix_r2 = DimArray(hcat(

        #The basis terms are all there
        parent(basis_matrix_r2),

        #The Age_first:Treatment_Medium interaction
        basis_matrix_r2[:, At(:Age_first)] .* basis_matrix_r2[:, At(:Treatment_Medium)],

        #The Age_first:Treatment_High interaction
        basis_matrix_r2[:, At(:Age_first)] .* basis_matrix_r2[:, At(:Treatment_High)]), (outcome_labels[At(:performance_mean)], fixed_effect_term_labels[At(:performance_mean)]))


## 4. Create random effect design matrices ##
random_effect_design_matrices_r2 = DimArray([

        # Factor 1: Subject (3 terms)
        basis_matrix_r2[:, At([:Intercept, :Treatment_Medium, :Treatment_High])],

        # Factor 2: Experimenter (2 terms)
        basis_matrix_r2[:, At([:Intercept, :Age_first])]], random_effect_factor_labels)


## 5. Create level assignments for random effects ##
random_effect_level_assignments_r2 = DimArray(hcat(

        #Variable 2: Subject (4 levels)
        repeat([1, 2, 3, 4], inner=3),  # [1,1,1, 2,2,2, 3,3,3, 4,4,4]

        #Variable 3: Experimenter (3 levels)
        repeat([1, 2, 3], outer=4),     # [1,2,3, 1,2,3, 1,2,3, 1,2,3]
    ), (outcome_labels[At(:performance_mean)], CategoricalVariableDim([:Subject, :Experimenter])))


## 6. Create interaction recipes ##
fixed_effects_interaction_recipes_r2 = [
    nothing, nothing, nothing, nothing, nothing, # Terms 1-5 are basis terms
    InteractionRecipe([2, 4], MultiplicationOperator()),       # Term 6: Age_first (2) * Treatment_Medium (4)
    InteractionRecipe([2, 5], MultiplicationOperator())        # Term 7: Age_first (2) * Treatment_High (5)
]

random_effects_interaction_recipes_r2 = DimArray([
        # Factor 1: Subject (3 terms, no interactions)
        Union{Nothing,InteractionRecipe{MultiplicationOperator}}[nothing for _ in 1:3],

        # Factor 2: Experimenter (2 terms, no interactions)
        Union{Nothing,InteractionRecipe{MultiplicationOperator}}[nothing for _ in 1:2]
    ], random_effect_factor_labels)

## 7. Create info for each term ##
terms_info_r2 = (
    
    #Intercept info
    Intercept=TermInfo(
        basis_expansion_type=IdentityExpansion(),
        basis_matrix_indices=[1],
        fixed_effects_indices=[1],
        random_effects_indices=[(1, [1]), (2, [1])],
        level_assignments_idx=0,
        dependent_interaction_indices=DependentInteractionIndices(Int[], Tuple{Int,Int}[])), 
        
    #Age info    
    Age=TermInfo(
        basis_expansion_type=PolynomialExpansion(2),
        basis_matrix_indices=[2, 3],
        fixed_effects_indices=[2, 3],
        random_effects_indices=[(2, [2, 0])],
        level_assignments_idx=0,
        dependent_interaction_indices=DependentInteractionIndices([6, 7], Tuple{Int,Int}[])
    ), 
    
    #Treatment info
    Treatment=TermInfo(
        basis_expansion_type=DummyCodeExpansion(),
        basis_matrix_indices=[4, 5],
        fixed_effects_indices=[4, 5],
        random_effects_indices=[(1, [2, 3])],
        level_assignments_idx=0,
        dependent_interaction_indices=DependentInteractionIndices([6, 7], Tuple{Int,Int}[])
    ),

    # Subject info
    Subject=TermInfo(
        basis_expansion_type=DummyCodeExpansion(),
        basis_matrix_indices=Int[],
        fixed_effects_indices=Int[],
        random_effects_indices=Tuple{Int,Vector{Int}}[],
        level_assignments_idx=1,
        dependent_interaction_indices=DependentInteractionIndices(Int[], Tuple{Int,Int}[])
    ),

    # Experimenter info
    Experimenter=TermInfo(
        basis_expansion_type=DummyCodeExpansion(),
        basis_matrix_indices=Int[],
        fixed_effects_indices=Int[],
        random_effects_indices=Tuple{Int,Vector{Int}}[],
        level_assignments_idx=2,
        dependent_interaction_indices=DependentInteractionIndices(Int[], Tuple{Int,Int}[])
    )
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

## - For regression 2 - ##
## 1. Get number of outcomes ##
n_outcomes_r3 = length(labels.outcomes[At(:performance_sd)])

## 2. Create basis matrix ##
basis_matrix_r3 = DimArray(hcat(

        # Variable 1: Intercept
        ones(n_outcomes_r3),

        # Variable 2: Age
        randn(n_outcomes_r3),

        # Variable 3: BMI
        randn(n_outcomes_r3)), (outcome_labels[At(:performance_sd)], basis_term_labels[At(:performance_sd)]))


## 3. Create fixed effects design matrix ##
fixed_effects_design_matrix_r3 = DimArray(hcat(

        # Terms 1-3: Intercept, Age, BMI
        parent(basis_matrix_r3),

        # Term 4: Age:BMI interaction
        basis_matrix_r3[:, At(:Age)] .* basis_matrix_r3[:, At(:BMI)],

        # Term 5: max(Age, BMI)
        max.(basis_matrix_r3[:, At(:Age)], basis_matrix_r3[:, At(:BMI)])), (outcome_labels[At(:performance_sd)], fixed_effect_term_labels[At(:performance_sd)]))


## 4. Create random effect design matrices ##
random_effect_design_matrices_r3 = DimArray([

        # Factor 1: Subject (4 terms: Intercept, Age, BMI, Age_x_BMI)
        fixed_effects_design_matrix_r3[:, At([:Intercept, :Age, :BMI, :Age_x_BMI])],

        # Factor 2: Experimenter (No terms)
        DimArray(Matrix{Float64}(undef, n_outcomes_r3, 0), (outcome_labels[At(:performance_sd)], RandomEffectTermDim(Symbol[])))
        
], random_effect_factor_labels)


## 5. Create level assignments for random effects ##
random_effect_level_assignments_r3 = DimArray(hcat(

        #Variable 2: Subject (4 levels)
        repeat([1, 2, 3, 4], inner=3),  # [1,1,1, 2,2,2, 3,3,3, 4,4,4]
    ), (outcome_labels[At(:performance_sd)], CategoricalVariableDim([:Subject])))



## 6. Create interaction recipes ##
fixed_effects_interaction_recipes_r3 = [nothing, nothing, nothing,
    InteractionRecipe([2, 3], MultiplicationOperator()), # 4: Age (2) * BMI (3)
    InteractionRecipe([2, 3], MaxOperator())             # 5: max(Age (2), BMI (3))
]

random_effects_interaction_recipes_r3 = DimArray([
    # Factor 1: Subject (4 terms, 3 main effects and 1 interaction)
    [nothing, nothing, nothing,
        InteractionRecipe([2, 3], MultiplicationOperator()), # 4: Age (2) * BMI (3)
    ],

    # Factor 2: Experimenter (No terms)
    Vector{Union{Nothing,InteractionRecipe{MultiplicationOperator}}}()
], random_effect_factor_labels)

## 7. Create info for each term ##
terms_info_r3 = (
    
    #Intercept info
    Intercept=TermInfo(
        basis_expansion_type=IdentityExpansion(),
        basis_matrix_indices=[1],
        fixed_effects_indices=[1],
        random_effects_indices=[(1, [1])],
        level_assignments_idx=0,
        dependent_interaction_indices=DependentInteractionIndices(Int[], Tuple{Int,Int}[])
    ), 
    
    #Age info
    Age=TermInfo(
        basis_expansion_type=IdentityExpansion(),
        basis_matrix_indices=[2],
        fixed_effects_indices=[2],
        random_effects_indices=[(1, [2])],
        level_assignments_idx=0,
        # Age is part of the interaction in the fixed effects design matrix (col 4) and in the Subject random effects design matrix (Factor 1, col 4)
        dependent_interaction_indices=DependentInteractionIndices([4, 5], [(1, 4)])
    ), 
    
    #BMI info
    BMI=TermInfo(
        basis_expansion_type=IdentityExpansion(),
        basis_matrix_indices=[3],
        fixed_effects_indices=[3],
        random_effects_indices=[(1, [3])],
        level_assignments_idx=0,
        # BMI is part of the interaction in Fixed (col 4) and Subject Random (Factor 1, col 4)
        dependent_interaction_indices=DependentInteractionIndices([4, 5], [(1, 4)])
    ), 
    
    #Subject info
    Subject=TermInfo(
        basis_expansion_type=DummyCodeExpansion(),
        basis_matrix_indices=Int[],
        fixed_effects_indices=Int[],
        random_effects_indices=Tuple{Int,Vector{Int}}[],
        level_assignments_idx=1,
        dependent_interaction_indices=DependentInteractionIndices(Int[], Tuple{Int,Int}[])
    ),
)

## 8. Create container for marking interaction effects for being updated ##
interaction_udpate_markers_r3 = InteractionUpdateMarkers(BitSet(), [BitSet() for _ in 1:length(random_effect_design_matrices_r3)])

## 9. Instantiate predictors ##
predictors_r3 = RegressionPredictors(
    basis_matrix_r3,
    fixed_effects_design_matrix_r3,
    random_effect_design_matrices_r3,
    random_effect_level_assignments_r3,
    terms_info_r3,
    fixed_effects_interaction_recipes_r3,
    random_effects_interaction_recipes_r3,
    interaction_udpate_markers_r3
)

## - Final predictors object - ##
predictors = DimArray([predictors_r1, predictors_r2, predictors_r3], regression_labels)

####################################
### 5 MAKE REGRESSION OPERATIONS ###
####################################
## All regession operations ##
operations = (
    
    #Operation 1: generate the underlying BMI with a simple linear combination
    BMI = RegressionOperation(
        LinearCombination(),
        UpdatePredictors(:BMI,[:performance_sd])
        ),

    #Operation 2: get the likelihood of the measured BMI given the latent BMI
    BMI_measurement_error_likelihood = RegressionOperation(
        DistributionLikelihood(
            dist = Normal,
            dist_args = (
                μ = ExtractPredictors(:performance_sd, [:BMI]),
                σ = 1,
                ),
            observations = rand(Normal(), n_outcomes_r1) #These are the measured BMI_measured
        ),
    ),

    #Operation 3: Get the predicted mean for the performance with a linear combination
    performance_mean = RegressionOperation(
        LinearCombination(),
        store_outcome = true
    ),

    #Operation 4: Get the predicted SD for the performance with a linear combination
    performance_sd = RegressionOperation(
        LinearCombination(),
        store_outcome = true
    ),

    #Operation 5: evaluate the measured performance against a Gaussian likelihood
    performance_likelihood = RegressionOperation(
        DistributionLikelihood(
            dist = Normal,
            dist_args = (
                μ = ExtractOutcome(:performance_mean),
                σ = ExtractOutcome(:performance_sd, exp)
                ),
            observations = ones(n_outcomes_r2) #These are the measured performances
        ),
    )
)


#################
### 7 TESTING ###
#################

## 1. Test rand() ##
coefficients = rand(priors)

## 2. Test logprob() ##
total_logprob = logpdf(priors, coefficients)

## 3. Test extraction of coefficients ##
fixed_effects = get_fixed_effects(coefficients)
random_effects = get_random_effects(coefficients)

## 4. Test updating predictors ##
update_predictors!(predictors, (:Age, :Treatment), (randn(12), rand(1:3, 12)), :performance_mean)

## 5. Test getter function for predictor basis term values ##
get_predictor_values(predictors, ExtractPredictors(:performance_mean, [:Age_first, :Age_second], exp))

## 6. Test linear_combination ##
outcomes = linear_combination(predictors, coefficients)

## 7. Test Turing regression model ##
model = regression_model(predictors, priors, operations)
chain = sample(model, Prior(), 1000, chain_type=VNChain)

chain[Symbol("performance_likelihood.observations")]
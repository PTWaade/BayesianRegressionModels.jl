import BayesianRegressionModels as BRM
using DimensionalData: DimArray, At
using Distributions: Normal, Gamma, product_distribution, LKJCholesky, logpdf
using Turing: sample, Prior, @varname
using FlexiChains: VNChain

##########################
### INPUT FROM FORMULA ###
##########################

## NOTE: THIS IS AN INITIAL SUGGESTION FOR A FORMULA SYNTAX. THERE CAN BE MUCH BETTER WAYS TO DO IT. THE SUGGESTION IS THAT IT DECOMPOSES INTO OPERATIONS - SEE BELOW

# @formula BMI = 1, data = dBMI, measurement_error(sd = 1, observations = BMI_measured)  <- this is to generate a latent BMI variable, where the BMI_measured is a noisy measurement of it.

# @formula performance_mean = 1 + Age_first * Treatment + Age_second + (1 + Treatment | Subject) + (1 + Age_first | Experimenter), data = dpmean,
#       @expansion PolynomialExpansion(2) Age [Age_first, Age_second]       <- expand the Age variable into Age_first and Age_second using polynomial expansion (only for this regression)

# @formula performance_sd = 1 + Age * BMI + 
#                     @interaction Age : BMI MaxOperator() label= :max_age_BMI +         <- Add a custom interaction operator (here one that selects the maximum), and give it a custom label
#                     (1 + Age * BMI | Subject), data = dpsd

# @likelihood Performance ~ Normal(performance_mean, performance_sd)

# @group Subject by = ClinicalGroup prior = nothing
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
regression_labels = BRM.RegressionDim([:BMI, :performance_mean, :performance_sd])


# 2. Labels for the levels of the categorical variables
categorical_level_labels = DimArray([

        #Variable 1: Treatment
        BRM.CategoricalLevelDim([:Low, :Medium, :High]),

        #Variable 2: Subject
        BRM.CategoricalLevelDim([:Subj1, :Subj2, :Subj3, :Subj4]),

        #Variable 3: Experimenter
        BRM.CategoricalLevelDim([:Exp1, :Exp2, :Exp3]),

        #Variable 4: ClinicalGroup
        BRM.CategoricalLevelDim([:ControlGroup, :TargetGroup])
        
    ], BRM.CategoricalVariableDim([:Treatment, :Subject, :Experimenter, :ClinicalGroup]))


# 3. Labels of basis terms (no interactions)
basis_term_labels = DimArray([

        #Regression 1: BMI
        BRM.BasisTermDim([
            :Intercept    
        ]),

        #Regression 2: performance_mean
        BRM.BasisTermDim([
            :Intercept, :Age_first, :Age_second, :Treatment_Medium, :Treatment_High
        ]),

        #Regression 3: performance_sd
        BRM.BasisTermDim([
            :Intercept, :Age, :BMI
        ]),
    
    ], regression_labels)


# 4. Labels for the fixed effect terms in the design matrices
fixed_effect_term_labels = DimArray([

        # Regression 1: BMI  
        BRM.FixedEffectTermDim([
            :Intercept
        ]),    

        # Regression 2: performance_mean
        BRM.FixedEffectTermDim([
            :Intercept, :Age_first, :Age_second, :Treatment_Medium, :Treatment_High, :Age_x_Treatment_Medium, :Age_x_Treatment_High
        ]),

        # Regression 3: performance_sd
        BRM.FixedEffectTermDim([
            :Intercept, :Age, :BMI, :Age_x_BMI, :max_age_BMI,
        ])
        
    ], regression_labels)


# 5. Labels for the random effect factors
random_effect_factor_labels = BRM.RandomEffectFactorDim([:Subject, :Experimenter])


# 6. Labels for the random effect terms in the design matrices
random_effect_term_labels = DimArray([

        #Regression 1: BMI
        DimArray([
                
            # Factor 1: Subject
            BRM.RandomEffectTermDim(Symbol[]),
            # Factor 2: Experimenter
            BRM.RandomEffectTermDim(Symbol[])
        
        ], random_effect_factor_labels),

        #Regression 1: performance_mean
        DimArray([

            #Factor 1: Subject
            BRM.RandomEffectTermDim([:Intercept, :Treatment_Medium, :Treatment_High]),
            #Factor 2: Experimenter
            BRM.RandomEffectTermDim([:Intercept, :Age_first])
                
        ], random_effect_factor_labels),

        #Regression 2: performance_sd
        DimArray([

            #Factor 1: Subject
            BRM.RandomEffectTermDim([:Intercept, :Age, :BMI, :Age_x_BMI]),
            #Factor 2: Experimenter (not present in the formula)
            BRM.RandomEffectTermDim(Symbol[])

        ], random_effect_factor_labels)
    
    ], regression_labels)


# 7. Labels and structure of random effect groups in each factor
random_effect_group_labels = DimArray([

        #Factor 1: Subject (get the levels from the categorical levels container)
        BRM.RandomEffectGroupDim(parent(categorical_level_labels[At(:ClinicalGroup)])),

        #Factor 2: Experimenter (no group specified, so use a single group)
        BRM.RandomEffectGroupDim([:SingleGroup])

    ], random_effect_factor_labels)


# 8. Labels and structure of random effect correlation blocks in each factor
random_effect_block_labels = DimArray([

        #Factor 1: Subject (ResidualBlock has all unnasigned terms in the factor)
        BRM.RandomEffectBlockDim([:Block1, :Block2, :ResidualBlock]),

        #Factor 2: Experimenter (no blocks specified, so use a single block)
        BRM.RandomEffectBlockDim([:SingleBlock])
        
    ], random_effect_factor_labels)


# 9. Labels for outcomes in each regression
outcome_labels = DimArray([

        #Regression 1: BMI
        BRM.OutcomeDim(1:12),

        #Regression 2: performance_mean
        BRM.OutcomeDim(1:12),

        #Regression 2: performance_sd
        BRM.OutcomeDim(1:12)
        
    ], regression_labels)


# 10. Create labels object #
labels = BRM.RegressionLabels(
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
        BRM.NonCentered,

        #Factor 2: Experimenter
        BRM.Centered

    ], random_effect_factor_labels)


## 4. Generate indices for mapping from flattened vector to structured coefficients ##
(fixed_effect_indices, random_effect_sds_indices, random_effect_sds_block_indices) = BRM.generate_indices(labels, block_assignments)


## 5. Create specifications object ##
specifications = BRM.RegressionSpecifications(group_assignments, block_assignments, random_effect_geometries, fixed_effect_indices, random_effect_sds_indices, random_effect_sds_block_indices, labels)



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
priors = BRM.RegressionPrior(fixed_effect_priors_gathered, random_effect_sd_priors_gathered, random_effect_correlation_LKJcholesky_priors, specifications)

x = rand(priors)

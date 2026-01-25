################
### PREAMBLE ###
################

#TODO:
# - A. Prioritised changes
#   - 1. Overload for gradient compat: to_vec(), from_vec_transform(), to_linked_vec_transform(), from_linked_vec_transform()
#   - 2. [!] Upgrade RegressionPredictors and update!:
            # Have a cateorical data field which holds the Integer version of categorical predictors and level assignments
            # Have a root matrix which has one copy of all predictors (and categorical predictors in dummy coding format)
            # Have design matrices as a view on the root matrix
            # Have level assignments as a view on the categorical data field
            # Have an InteractionDependencies object which stores for each interaction column in the design matrices which columns it depends on (and which function to use to combine them, default is multiplication)
            # Have a TermMetaData object which stores whether a given term is categorical, and whether it is a predictor (or a level assignment), and which columns in the root matrix it corresponds to, and which columns in the design matrices are interactions that depends on it
            # Make update! first change the categorical predictor, and then update the root matrix
            # Make update! change the root matrix, and then update relevant interaction effect columns in the design matrices
            # Make update! allow for receiving multiple terms to update at the same time. Only update interactions after all predictors have been updated (store all IDs that must be updated). Use a Set around the to_update flags to only do it once
            # Make update! take a NamedTuple as input, where keys are term names and values are vector with values to update them with
            # Make update! use different types internally for updating continuous predictors, categorical predictors, and level assignments
# - B. Optimisation
#   - 1. [!] Minimise use of DimensionalData where not needed
#   - 2. Ensure type stability
#   - 3. Pre-allocate random effect block assignments
# - C. Core Utilities
#   - 1. Make constructor function for RegressionPrior
#      A. Which construct multivariate distributions from individual priors_f
#      B. Which creates default priors or extrapolates single priors to full structure
#      C. Which checks that the inputs are all properly structured and matching
#      D. Which allows for using symbols to define the levels assignments (these are then transformed into integers)
#   - 2. Make custom summary functionalities (FlexiChains & MCMCChains)
#   - 3. Make custom plotting functions (FlexiChains & MCMCChains)
# - D. Fixes
#   - 1. Add comments with canonical mathematical notation
#   - 2. Set full, concrete type requirements everywhere possible
#   - 3. Ensure that DualNumbers can be used throughout
#   - 4. Make getter functions for random effect hyperparameters
#   - 5. Organise repository
#   - 6. Make RegressionPrior modular, so that differnet components can be sample one at a time
# - E. Functionality
#   - 1. unit tests
#   - 2. documentation
# - F. Usage
#   - 1. Fit the example Turing model with FlexiChains
#   - 2. Make example with Horseshoe priors.
#   - 3. Make example with Spike-and-slab priors.
#   - 4. Make example with a latent mixture model.
#   - 5. Make example with multi-step Turing model
#   - 6. Make example with categorical predictors
# - G. Near future features
#   - 1. Make preconstructed spike-and-slab prior distribtution 
#   - 2. Make preconstructed horseshoe prior distribution 
#   - 3. Make preconstructed Variance Component Analysis prior distribtution (letting random effect sd priors come from a multivariate distribution which can weigh between them).
#   - 3. Allow for sharing parameters across regressions (e.g., fixed effects being identical in multiple regressions)
#   - 4. Make constructor for combining multivariate distributions so that they sample vectors
#   - 5. Add labels for categorical predictors
#   - 6. Make example with splines / polynomial terms
#   - 7. Make example with completely custom functions
# - H. Extra
#   - 1. Make Turing submodel alternative to rand and logpdf (and benchmark)
# - I. Long-future and difficult features
#   - 1. Structured random effects across levels (e.g., gaussian process, AR1, etc.)
#   - 2. Non-parametric, infinite mixture, Dirichlet process models etc (i.e., where not just the level assignments, but also the number of levels, is inside the Turing model)
#   - 3. Allow for estimating group memberships of random effect levels inside the Turing model
# - X. Decisions to make
#   - 1. What should be the value in matrices with un-generated values? 0, undef or missing?
#   - 2. What do we do with missing values in the predictors? Set them to 0, drop them, or return an error? How about NaN?
#   - 3. Should fixed effects and random effect sds be stored as flat vectors or as structured vectors internally?

### TERMINOLOGY ###
# - Regression (r): A single regression model. Multiple can be connected.
# - Fixed effect terms (p): The coefficients associated with the predictors (e.g., Intercept, Age, Gender).
# - Random effect factor (f): The identifier variable (e.g., SubjectID, ItemID).
# - Random effect levels (l): The unique instances within a factor (e.g., Subject 1, Subject 2).
# - Random effect groups (g): Sub-partitioning levels of a factor (Strata) for independent random effect variances and random effects (e.g., healthy vs. clinical). brms syntax: (1 | gr(subjID, by = diagnosis))
# - Random effect terms (q): The variables that vary across a factor (e.g., Intercept, Age, Gender). Is often collapsed across regressions to q_total.
# - Random effect blocks (b): Sets of random effect terms that are internally correlated (e.g., Intercept, Age, Gender). Blocks can be across regressions. brms syntax: (1 + age |p| subjID)
# - Observations (n): Each row in the data frame

### FUNCTIONALITY ###
# - multiple fixed effect terms (intercept and multiple predictors)
# - multiple random effect factors (e.g., subjects and items) with multiple random effect terms (intercept and multiple predictors)
# - grouped random effects (e.g., healthy vs. clinical subjects)
# - multiple regressions (e.g., multivariate outcomes)
# - random effect correlations within groups and between terms and regressions
# - multiple random effect correlation blocks within a factor
# - can use centered and non-centered parameterisations for random effects, on a per-factor basis
# - can update the predictors during the Turing model, so that predictors can be inferred
# - can update random effect level assignments during the Turing model, so that level assignments can be inferred

### CONSTRAINTS ###
# - Random effect groups g must be applied across all regressions r 
# - Random effect levels l can only belong to a single group g within a factor f
# - There must be entries for each group in each factor for each regression. If a regression does not use a given factor, pass an empty vector instead of a vector with labels/priors for each term.
# - Random effect correlations must be specified for all terms across regressions but within a group and within a factor
# - I've made a hard assumption that the covariance matrices use a LKJCholesky prior, and not a LKJ for example.
# - We do not allow random effect blocks to be different within different random effect groups. Implementationally this would get difficult.
# - The priors over the fixed effects and random effect sds are multivariate distributions. If there is only a single fixed effect, this must still be a multivariate distribution, such as a one-dimensional MvNormal.
# - Even though predictors and random effect level assignments can be inferred, the number of levels and their labels must be known ahead of time (for random effect levels and for categorical predictors)

### POINTS OF UNCERTAINTY ###
# - adding the jitter to the covariance matrices when reconstructing them for logpdf calculations. Is this done right? How large should it be?

### DIFFERENCES TO brms ###
# - instead of the gr() syntax for grouping random effects, the grouping should perhaps be passed as a separate argument.
#   The reason is that a single factor must have the same grouping across all regressions.
#   We could also allow users to specify it in the brms way, but then we would need to check consistency across regressions.






## For the regression distribution ##
using Distributions
using LinearAlgebra
using Random
using PDMats #This gives efficient computation for psitive definite matrices, used for the LKJCholesky stuff
using DimensionalData #This allows for named dimensions in arrays
using DimensionalData: @dim

## For the Turing model ##
using Turing
using FlexiChains



include("1_regression_prior.jl")

include("2_linear_combination.jl")

include("3_TuringExt.jl")

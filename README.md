# BayesianRegressionModels.jl
Temporary repository for Julia package for doing Bayesian regression models, using Turing.jl as backend. Inspired by brms and bambi.

### TERMINOLOGY ###
- Regression (r): A single regression model. Multiple can be connected.
- Fixed effect terms (p): The coefficients associated with the predictors (e.g., Intercept, Age, Gender).
- Random effect factor (f): The identifier variable (e.g., SubjectID, ItemID).
- Random effect levels (l): The unique instances within a factor (e.g., Subject 1, Subject 2).
- Random effect groups (g): Sub-partitioning levels of a factor (Strata) for independent random effect variances and random effects (e.g., healthy vs. clinical). brms syntax: (1 | gr(subjID, by = diagnosis))
- Random effect terms (q): The variables that vary across a factor (e.g., Intercept, Age, Gender). Is often collapsed across regressions to q_total.
- Random effect blocks (b): Sets of random effect terms that are internally correlated (e.g., Intercept, Age, Gender). Blocks can be across regressions. brms syntax: (1 + age |p| subjID)
- Observations (n): Each row in the data frame

### FUNCTIONALITY ###
- multiple fixed effect terms (intercept and multiple predictors)
- multiple random effect factors (e.g., subjects and items) with multiple random effect terms (intercept and multiple predictors)
- grouped random effects (e.g., healthy vs. clinical subjects)
- multiple regressions (e.g., multivariate outcomes)
- random effect correlations within groups and between terms and regressions
- multiple random effect correlation blocks within a factor
- can use centered and non-centered parameterisations for random effects, on a per-factor basis
- can update the predictors during the Turing model, so that predictors can be inferred
- can update random effect level assignments during the Turing model, so that level assignments can be inferred
- can use custom expansions, such as polynomial or spline expansions of predictors, even if they created during the model
- can use custom interaction operators beyond multiplication, even if their basis is created during the TUring model
- can include measurement error with an appropriate regression submodel
- can impute missing data byt having missings in data vectors used for distribution likelihoods

### CONSTRAINTS ###
- Random effect groups g must be applied across all regressions r 
- Random effect levels l can only belong to a single group g within a factor f
- There must be entries for each group in each factor for each regression. If a regression does not use a given factor, pass an empty vector instead of a vector with labels/priors for each term.
- Random effect correlations must be specified for all terms across regressions but within a group and within a factor
- I've made a hard assumption that the covariance matrices use a LKJCholesky prior, and not a LKJ for example.
- We do not allow random effect blocks to be different within different random effect groups. Implementationally this would get difficult.
- The priors over the fixed effects and random effect sds are multivariate distributions. If there is only a single fixed effect, this must still be a multivariate distribution, such as a one-dimensional MvNormal.
- Even though predictors and random effect level assignments can be inferred, the number of levels and their labels must be known ahead of time (for random effect levels and for categorical predictors)

### POINTS OF UNCERTAINTY ###
- adding the jitter to the covariance matrices when reconstructing them for logpdf calculations. Is this done right? How large should it be?
- Is there a better way than growing the NamedTuple in the model ? A recursive function?

### DIFFERENCES TO brms ###
- instead of the gr() syntax for grouping random effects, the grouping should perhaps be passed as a separate argument.
   The reason is that a single factor must have the same grouping across all regressions.
   We could also allow users to specify it in the brms way, but then we would need to check consistency across regressions.



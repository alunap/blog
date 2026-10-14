using DataFramesMeta, Distributions, StatsBase, StatsPlots, RCall, MCMCDiagnosticTools

"""We would like the exact same random variables as in Chapter 4 of the book
so let's make use of RCall to reproduce it exactly."""


R"""
set.seed(4)
y10 <- rnorm(n = 10, mean = 600, sd = 30) # Sample of 10 birds
y1000 <- rnorm(n = 1000, mean = 600, sd = 30) 
"""
@rget y10 y1000

truth = (mean=600, sd=30)

p1 = histogram(y10,
    color=:grey,
    xlims=(500, 700),
    bins=5,
    xlabel="y10",
    ylabel="Frequency",
    legend=false
)

p2 = histogram(y1000,
    color=:grey,
    bins=10,
    xlabel="y1000",
    legend=false
)

# 3. Combine in a 1x2 layout
plot(p1, p2, layout=(1, 2), size=(800, 400))

## Using canned functions
"""In Julia we use GLM.jl for the equivalent to R code. It takes an object that implements the Tables.jl inteface, not
 a vector, so we just wrap it in a DataFrame."""

using GLM

y10lm = lm(@formula(y10 ~ 1), DataFrame(y10=y10))
y1000lm = lm(@formula(y1000 ~ 1), DataFrame(y1000=y1000))

using Turing

@model function model_4_4(mass)
    # 1. Priors
    # Note: BUGS dnorm(0, 0.0001) uses precision τ = 0.0001, 
    # which corresponds to variance 10,000 or SD = 100 in Turing.
    pop_mean ~ Uniform(0, 5000)
    pop_sd ~ Uniform(1, 100)

    # Add derived quantity for variance if required
    pop_var = pop_sd^2
    # 3. Likelihood (Vectorized)
    mass .~ Normal(pop_mean, pop_sd)

end

# Sampling
model = model_4_4(y1000)
chains = sample(model, NUTS(), 10000; chain_type=Chains)
describe(chains)
plot(chains)
# add pop_var if we wish
pop_var_samples = chains[:pop_sd] .^ 2

# We can also look at individual diagnostics afterwards
bfmi(chains) # updated from gelmandiag
ess(chains) # effective sample size for each parameter
rhat(chains) # potential scale reduction factor for each parameter
mcse(chains) # Monte Carlo standard error for each parameter


#--------------------------#
# None of these versions work
#--------------------------#


# JAGS version
# No luck getting JuliaBUGS to work so far
using JuliaBUGS
using JuliaBUGS: Gibbs
using AbstractMCMC, AdvancedHMC, ADTypes, Mooncake, FlexiChains

jagsmodel = @bugs("""
model {
        # priors
        pop.mean ~ dunif(0, 5000)
        pop.sd ~ dunif(0, 100)
        pop.var <- pop.sd * pop.sd
        precision <- 1 / pop.var

        # likelihood
        for (i in 1:N) {
            mass[i] ~ dnorm(pop.mean, precision)
        }
}
""")

dataList = (N=length(y1000), mass=y1000,)

# Inits can be simply a NamedTuple. note that the @bugs macro automatically converted dots in variable names to underscores
# We use the underscore variant, after translation.
inits() = (
    pop_mean=rand(Normal(600, 1)),
    pop_sd=rand(Uniform(1, 30))
)

# Or for multiple chains, a vector of named tuples
n_chains = 3

jags_inits = [
    (
        pop_mean=rand(Normal(600, 1)),
        pop_sd=rand(Uniform(1, 30))
    ) for _ in 1:n_chains
]

# In R we have to specify which params to monitor. In JuliaBUGS everything is saved, we just extract what we 
# are interested in at the end.
# Filter the resulting chain to only include the desired parameters

model = jagsmodel(data)
ex = JuliaBUGS.BUGSExamples.VOLUME_1
rats = ex[:rats]
rats


model = JuliaBUGS.set_evaluation_mode(
    jagsmodel, JuliaBUGS.UseGeneratedLogDensityFunction()
)
model = JuliaBUGS.BUGSModelWithGradient(model, AutoMooncake(; config=nothing))

# 5. Sample (e.g., using Gibbs or MH sampler)
chain = sample(model, Gibbs(), 10_000)

# 6. Subset parameters (equivalent to R's params)
chain_params = chain[[:pop_mean, :pop_sd, :pop_var]]
display(chain_params)


# -----------------------------------------------
# This is now deprecated, and turing_model does not work as per docs
using TuringGLM
fm = @formula(y1000 ~ 1)
model = TuringGLM.turing_model(fm, DataFrame(y1000=y1000))
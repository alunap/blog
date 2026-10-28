using Test
using Random: MersenneTwister, randn
using Turing
using FlexiChains: FlexiChain, Parameter, VarName, @varname

include(joinpath(@__DIR__, "..", "files", "model_comparison.jl"))
using .ModelComparison

const N_ITERS = 100
const N_CHAINS = 2
const MOCK_CHAIN = FlexiChain{VarName}(
    N_ITERS,
    N_CHAINS,
    Dict(
        Parameter(@varname(α)) => randn(MersenneTwister(1), N_ITERS, N_CHAINS),
        Parameter(@varname(β)) => randn(MersenneTwister(2), N_ITERS, N_CHAINS),
        Parameter(@varname(σ)) => randn(MersenneTwister(3), N_ITERS, N_CHAINS),
    ),
)

struct MockModel end

const MockThreadsSampler = typeof(MCMCThreads())

Turing.sample(::MockModel, ::Turing.NUTS, ::Integer; progress=false) = MOCK_CHAIN
Turing.sample(
    ::MockModel,
    ::Turing.NUTS,
    ::MockThreadsSampler,
    ::Integer,
    ::Integer;
    progress=false,
) = MOCK_CHAIN

@testset "FlexiChains model comparison" begin
    result = compare_model_efficiency(MockModel(), N_CHAINS, ["α", "β", "σ"])

    @test result.sampler == ["Turing"]
    @test isfinite(result.min_ess[1])
    @test isfinite(result.mean_ess[1])
end

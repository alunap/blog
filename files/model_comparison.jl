
module ModelComparison

using Turing, FlexiChains, MCMCDiagnosticTools, DataFrames, Statistics, CSV

export compare_model_efficiency

function compare_model_efficiency(model, n_chains, params_to_track, jags_file)
    # ⚠️ JIT warm-up — CRITICAL. Run a tiny throwaway chain first, otherwise
    # compilation time destroys Turing's numbers.
    sample(model, NUTS(), 10; progress=false)

    # Time the real run (NUTS warmup/adaptation included, matching the JAGS timing)
    elapsed = @elapsed chn = sample(model, NUTS(), MCMCThreads(), 1000, n_chains; progress=false)

    # Per-parameter ESS, restricted to the SAME parameters monitored in JAGS
    common = params_to_track
    ess_summary = ess(chn[Symbol.(common)])
    ess_vals = [ess_summary[Symbol(param)] for param in common]

    turing_metrics = DataFrame(sampler="Turing",
        min_ess=minimum(ess_vals),
        mean_ess=mean(ess_vals),
        seconds=elapsed)
    turing_metrics.min_ess_s = turing_metrics.min_ess / elapsed
    turing_metrics.mean_ess_s = turing_metrics.mean_ess / elapsed
    jags_metrics = CSV.read(jags_file, DataFrame)

    combined_metrics = vcat(turing_metrics, jags_metrics)

    return combined_metrics
end

end # module ModelComparison
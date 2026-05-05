using HDF5
using Statistics
using Printf

function read_rj(rjsamples_file::String)
    h5open(rjsamples_file, "r") do hdf
        mode = haskey(HDF5.attributes(hdf), "mode") ?
               read(HDF5.attributes(hdf)["mode"]) : ""
        mode == "RJMCMC" ||
            throw(ArgumentError("Not an RJMCMC output file (mode = \"$mode\")"))
        model_index   = read(hdf, "model_index")    
        params_shared = read(hdf, "params_shared")  
        param4        = read(hdf, "param4")          
        ha_samples    = read(hdf, "ha_samples")      
        active_models = haskey(hdf, "active_models") ?
                        Int.(read(hdf, "active_models")) : collect(1:N_MODELS)
        return model_index, params_shared, param4, ha_samples, active_models
    end
end

function model_posterior_probs(rjsamples_file::String)::Matrix{Float64}
    midx, _, _, _, _ = read_rj(rjsamples_file)
    n_hex, n_post = size(midx)
    probs = zeros(n_hex, N_MODELS)
    for i in 1:n_hex, s in 1:n_post
        probs[i, midx[i, s]] += 1.0
    end
    probs ./= n_post
    return probs
end

function map_model_index(rjsamples_file::String)::Vector{Int}
    probs = model_posterior_probs(rjsamples_file)
    return [argmax(probs[i, :]) for i in 1:size(probs, 1)]
end

function model_posterior_entropy(rjsamples_file::String)::Vector{Float64}
    probs = model_posterior_probs(rjsamples_file)
    n_hex = size(probs, 1)
    ent   = zeros(n_hex)
    for i in 1:n_hex, m in 1:N_MODELS
        p = probs[i, m]
        p > 0 && (ent[i] -= p * log(p))
    end
    return ent
end

function marginal_param_samples(rjsamples_file::String;
    model::Union{Nothing,Int} = nothing)::Array{Float64,3}

    midx, pshared, _, _, _ = read_rj(rjsamples_file)
    model === nothing && return pshared

    n_hex, n_post = size(midx)
    in_model = [findall(==(model), midx[i, :]) for i in 1:n_hex]
    n_filt   = minimum(length.(in_model))
    if n_filt == 0
        @warn "model=$(MODEL_NAMES[model]) was never visited in at least one bin; " *
              "conditional samples unavailable"
        return Array{Float64,3}(undef, n_hex, 0, 3)
    end
    out = Array{Float64,3}(undef, n_hex, n_filt, 3)
    for i in 1:n_hex
        out[i, :, :] = pshared[i, in_model[i][1:n_filt], :]
    end
    return out
end

function conditional_param4_samples(rjsamples_file::String, target::Int)::Matrix{Float64}
    midx, _, p4, _, _ = read_rj(rjsamples_file)
    n_hex, n_post  = size(midx)
    in_model = [findall(==(target), midx[i, :]) for i in 1:n_hex]
    n_filt   = minimum(length.(in_model))
    if n_filt == 0
        @warn "model=$(MODEL_NAMES[target]) never visited in at least one bin"
        return Matrix{Float64}(undef, n_hex, 0)
    end
    out = Matrix{Float64}(undef, n_hex, n_filt)
    for i in 1:n_hex
        out[i, :] = p4[i, in_model[i][1:n_filt]]
    end
    return out
end

function rj_quality_mask(rjsamples_file::String;
    min_map_model_prob::Float64   = 0.5,
    alpha_width_threshold::Float64 = 0.5,
    tfrac_width_threshold::Float64 = 0.4)::BitVector

    probs                    = model_posterior_probs(rjsamples_file)
    midx, pshared, _, _, _ = read_rj(rjsamples_file)
    n_hex                    = size(probs, 1)
    mask                     = trues(n_hex)

    for i in 1:n_hex
        maximum(probs[i, :]) < min_map_model_prob && (mask[i] = false; continue)

        α_fin = filter(isfinite, pshared[i, :, 3])
        if !isempty(α_fin)
            q = quantile(α_fin, [0.16, 0.84])
            q[2] - q[1] > alpha_width_threshold && (mask[i] = false; continue)
        end

        A_samp   = pshared[i, :, 1]
        Bnt_samp = pshared[i, :, 2]
        frac     = A_samp ./ (A_samp .+ Bnt_samp)
        frac_fin = filter(isfinite, frac)
        if !isempty(frac_fin)
            q = quantile(frac_fin, [0.16, 0.84])
            q[2] - q[1] > tfrac_width_threshold && (mask[i] = false)
        end
    end

    n_masked = count(!, mask)
    println("RJ quality mask: $(n_hex - n_masked)/$n_hex bins pass ($n_masked masked)")
    return mask
end

function rj_diagnostics_summary(rjsamples_file::String)
    visit_fracs, within_rates, between_rates, active_models =
        h5open(rjsamples_file, "r") do hdf
            read(hdf, "model_visit_fractions"),
            read(hdf, "within_acceptance_rates"),
            read(hdf, "between_acceptance_rates"),
            haskey(hdf, "active_models") ?
                Int.(read(hdf, "active_models")) : collect(1:N_MODELS)
        end

    n_hex = size(visit_fracs, 1)
    println("\n── RJMCMC diagnostics summary ($n_hex bins) ─────────────────")
    @printf("  %-18s  %8s  %8s  %12s  %13s\n",
            "Model", "Visit %", "Never%", "Within acc.", "Between acc.")
    for m in active_models
        vf        = mean(visit_fracs[:, m]) * 100
        never_pct = count(==(0.0), visit_fracs[:, m]) / n_hex * 100
        wr        = mean(filter(isfinite, within_rates[:, m]))
        br        = mean(filter(isfinite, between_rates[:, m]))
        @printf("  %-18s  %7.1f%%  %7.1f%%  %11.3f  %12.3f\n",
                MODEL_NAMES[m], vf, never_pct, wr, br)
    end
    println("────────────────────────────────────────────────────────────\n")
end

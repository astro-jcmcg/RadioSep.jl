using Distributions
using LinearAlgebra
using HDF5
using Statistics
using ProgressMeter
using Printf

const PL_IDX  = 1
const CPL_IDX = 2
const FFA_IDX = 3
const BPL_IDX = 4
const N_MODELS    = 4
const MODEL_DIM   = (3, 4, 4, 4)
const MODEL_NAMES = ("PowerLaw", "CurvedPowerLaw", "ffa", "BrokenPowerLaw")
const LOG_BPL_RANGE = log(1e12) - log(1e7)   # ≈ 11.51

model_from_idx(i::Int) = (PowerLaw(), CurvedPowerLaw(), ffa(), BrokenPowerLaw())[i]

function model_to_idx(m)::Int
    m isa PowerLaw       && return PL_IDX
    m isa CurvedPowerLaw && return CPL_IDX
    m isa ffa            && return FFA_IDX
    m isa BrokenPowerLaw && return BPL_IDX
    throw(ArgumentError("Unrecognised model type $(typeof(m)). " *
                        "Must be PowerLaw, CurvedPowerLaw, ffa, or BrokenPowerLaw."))
end

struct RJConfig
    n_samples::Int
    n_burnin::Int
    n_within::Int
    n_pilot::Int                           
    active_models::Vector{Int}             
    log_model_prior::NTuple{4,Float64}     
    initial_scale::Float64
    min_adapt_count::Int
    adapt_regularisation::Float64
end

function RJConfig(;
    n_samples::Int                = 2000,
    n_burnin::Int                 = 400,
    n_within::Int                 = 10,
    n_pilot::Int                  = 300,
    model_prior                   = [0.25, 0.25, 0.25, 0.25],
    active_models                 = nothing,
    initial_scale::Float64        = 0.3,
    min_adapt_count::Int          = 20,
    adapt_regularisation::Float64 = 1e-6)

    active_idx = if active_models === nothing
        collect(1:N_MODELS)
    else
        sort(unique(Int[model_to_idx(m) for m in active_models]))
    end
    length(active_idx) >= 2 || throw(ArgumentError("active_models must include at least 2 models"))

    mp = collect(Float64, model_prior)
    length(mp) == 4   || throw(ArgumentError("model_prior must have length 4"))
    all(≥(0.0), mp)   || throw(ArgumentError("model_prior values must be non-negative"))

    mp_active = zeros(4)
    for m in active_idx; mp_active[m] = mp[m]; end
    sum(mp_active) > 0 || throw(ArgumentError("active model_prior values must not all be zero"))
    lmp = Tuple(map(p -> p > 0 ? log(p / sum(mp_active)) : -Inf, mp_active))

    RJConfig(n_samples, n_burnin, n_within, n_pilot, active_idx, lmp,
             initial_scale, min_adapt_count, adapt_regularisation)
end

mutable struct AdaptState
    n::Int
    mean::Vector{Float64}
    M2::Matrix{Float64}
end

AdaptState(d::Int) = AdaptState(0, zeros(d), zeros(d, d))

function update_adapt!(ap::AdaptState, x::AbstractVector)
    ap.n += 1
    δ  = x .- ap.mean
    ap.mean .+= δ ./ ap.n
    δ2 = x .- ap.mean
    for j in eachindex(x), i in eachindex(x)
        ap.M2[i, j] += δ[i] * δ2[j]
    end
end

function proposal_chol(ap::AdaptState, d::Int, reg::Float64)
    ap.n < 2d + 1 && return nothing
    Σ_emp  = ap.M2 ./ (ap.n - 1)
    Σ_prop = (2.38^2 / d) .* (Σ_emp .+ reg .* I(d))
    try
        return cholesky(Symmetric(Σ_prop)).L
    catch
        return nothing
    end
end

struct RJState
    model::Int
    params::Vector{Float64}
    lp::Float64
end

function lnpost_rj(m::Int, params, ha_meas::Float64, ha_uncert::Float64,
                    freq, data, data_err)::Float64
    m == PL_IDX && return lnpost(params, ha_meas, ha_uncert, freq, data, data_err)
    return lnpost(params, ha_meas, ha_uncert, freq, data, data_err, model_from_idx(m))
end

function lnlike_rj(m::Int, params, freq, data, data_err)::Float64
    m == PL_IDX && return lnlike(params, freq, data, data_err)
    return lnlike(params, freq, data, data_err, model_from_idx(m))
end

struct Param4Proposal
    μ::Float64    
    σ::Float64   
end

function draw_param4(m::Int, p::Param4Proposal)::Float64
    if m == CPL_IDX
        return p.μ + p.σ * randn()
    elseif m == FFA_IDX
        return rand(truncated(Normal(p.μ, p.σ), 0.0, Inf))
    else  
        lo, hi = log(1e7), log(1e12)
        return exp(rand(truncated(Normal(p.μ, p.σ), lo, hi)))
    end
end

function log_q(u::Float64, m::Int, p::Param4Proposal)::Float64
    if m == CPL_IDX
        return logpdf(Normal(p.μ, p.σ), u)
    elseif m == FFA_IDX
        return logpdf(truncated(Normal(p.μ, p.σ), 0.0, Inf), u)
    else  
        lo, hi = log(1e7), log(1e12)
        return logpdf(truncated(Normal(p.μ, p.σ), lo, hi), log(u)) - log(u)
    end
end

function log_prior_param4(u::Float64, m::Int)::Float64
    m == CPL_IDX && return logpdf(Normal(0.0, 0.5), u)
    m == FFA_IDX && return logpdf(Exponential(0.1), u)
    return -log(u) - LOG_BPL_RANGE
end

function pilot_param4(m::Int, θ_shared::AbstractVector,
                       ha_meas::Float64, ha_uncert::Float64,
                       freq, data, data_err,
                       n_pilot::Int)::Param4Proposal

    u_can0, step, σ_floor = if m == CPL_IDX
        0.0,  0.3,  0.02
    elseif m == FFA_IDX
        0.1,  0.15, 0.02
    else  
        0.5*(log(1e7)+log(1e12)), 1.0, 0.2
    end

    θ_full   = Vector{Float64}(undef, 4)
    θ_full[1:3] = θ_shared
    u_can    = u_can0
    u_nat    = m == BPL_IDX ? exp(u_can) : u_can
    θ_full[4] = u_nat
    lp_cur   = lnpost_rj(m, θ_full, ha_meas, ha_uncert, freq, data, data_err)

    samples_can = Vector{Float64}(undef, n_pilot)

    for k in 1:n_pilot
        u_prop = u_can + step * randn()
        if m == FFA_IDX
            u_prop = max(u_prop, 1e-10)
        elseif m == BPL_IDX
            u_prop = clamp(u_prop, log(1e7), log(1e12))
        end

        u_nat_prop   = m == BPL_IDX ? exp(u_prop) : u_prop
        θ_full[4]    = u_nat_prop
        lp_prop      = lnpost_rj(m, θ_full, ha_meas, ha_uncert, freq, data, data_err)

        if log(rand()) < lp_prop - lp_cur
            u_can  = u_prop
            lp_cur = lp_prop
        end
        samples_can[k] = u_can
    end

    s = @view samples_can[(n_pilot÷2+1):end]
    μ = mean(s)
    σ = max(std(s), σ_floor)
    return Param4Proposal(μ, σ)
end

function within_step(state::RJState,
                       ha_meas::Float64, ha_uncert::Float64,
                       freq, data, data_err,
                       adapt::AdaptState,
                       config::RJConfig)::Tuple{RJState,Bool}

    d    = MODEL_DIM[state.model]
    chol = proposal_chol(adapt, d, config.adapt_regularisation)

    θ_new = if chol !== nothing
        state.params .+ chol * randn(d)
    else
        state.params .+ config.initial_scale .* randn(d)
    end

    lp_new   = lnpost_rj(state.model, θ_new, ha_meas, ha_uncert, freq, data, data_err)
    accepted = log(rand()) < lp_new - state.lp

    if accepted
        update_adapt!(adapt, θ_new)
        return RJState(state.model, θ_new, lp_new), true
    else
        update_adapt!(adapt, state.params)
        return state, false
    end
end

function between_step_to(state::RJState, j::Int,
                           ha_meas::Float64, ha_uncert::Float64,
                           freq, data, data_err,
                           config::RJConfig,
                           proposals::NTuple{4,Param4Proposal})::Tuple{RJState,Bool}

    lmp    = config.log_model_prior
    ll_old = lnlike_rj(state.model, state.params, freq, data, data_err)
    i      = state.model

    θ_new, ll_new, corr = if i == PL_IDX
        u    = draw_param4(j, proposals[j])
        θ′   = vcat(state.params, u)
        ll   = lnlike_rj(j, θ′, freq, data, data_err)
        corr = log_prior_param4(u, j) - log_q(u, j, proposals[j])
        θ′, ll, corr
    elseif j == PL_IDX
        θ′   = state.params[1:3]
        ll   = lnlike_rj(PL_IDX, θ′, freq, data, data_err)
        u    = state.params[4]
        corr = log_q(u, i, proposals[i]) - log_prior_param4(u, i)
        θ′, ll, corr
    else
        v    = draw_param4(j, proposals[j])
        θ′   = vcat(state.params[1:3], v)
        ll   = lnlike_rj(j, θ′, freq, data, data_err)
        u    = state.params[4]
        corr = (log_prior_param4(v, j) - log_q(v, j, proposals[j])) +
               (log_q(u, i, proposals[i]) - log_prior_param4(u, i))
        θ′, ll, corr
    end

    log_α    = (lmp[j] - lmp[i]) + (ll_new - ll_old) + corr
    accepted = log(rand()) < log_α

    if accepted
        lp_new = lnpost_rj(j, θ_new, ha_meas, ha_uncert, freq, data, data_err)
        return RJState(j, θ_new, lp_new), true
    else
        return state, false
    end
end

struct RJDiagnostics
    model_visit_counts::Vector{Int}
    within_accepts::Vector{Int}
    within_proposals::Vector{Int}
    between_accepts::Matrix{Int}
    between_proposals::Matrix{Int}
end

function run_rj_chain(data::AbstractVector, data_err::AbstractVector,
                        freq::AbstractVector,
                        ha_meas::Float64, ha_uncert::Float64,
                        config::RJConfig)

    n_total      = config.n_burnin + config.n_samples
    n_post       = config.n_samples
    pilot_update = config.n_burnin ÷ 2  

    mean_flux   = mean(filter(isfinite, data))
    therm_init  = max(ha_meas, mean_flux * 0.1, 1e-9)
    ntherm_init = max(mean_flux - therm_init, therm_init * 0.1, 1e-9)
    θ_shared0   = [therm_init, ntherm_init, -0.8]

    init_m = config.active_models[1]
    θ_init = init_m == PL_IDX ? θ_shared0 :
             vcat(θ_shared0, init_m == CPL_IDX ? 0.0 :
                              init_m == FFA_IDX ? 0.1 :
                              exp(0.5*(log(1e7)+log(1e12))))
    lp_init = lnpost_rj(init_m, θ_init, ha_meas, ha_uncert, freq, data, data_err)
    if !isfinite(lp_init)
        θ_shared0 = [max(ha_meas, 1e-9), max(mean_flux * 0.9, 1e-9), -0.8]
        θ_init    = init_m == PL_IDX ? θ_shared0 : vcat(θ_shared0, θ_init[4])
        lp_init   = lnpost_rj(init_m, θ_init, ha_meas, ha_uncert, freq, data, data_err)
    end
    state = RJState(init_m, θ_init, lp_init)

    prior_proposals = ntuple(Val(N_MODELS)) do m
        m == CPL_IDX ? Param4Proposal(0.0, 0.5) :
        m == FFA_IDX ? Param4Proposal(0.1, 0.3) :
        m == BPL_IDX ? Param4Proposal(0.5*(log(1e7)+log(1e12)), (log(1e12)-log(1e7))/4) :
                       Param4Proposal(0.0, 0.5)   
    end
    proposals = prior_proposals   

    burnin_theta_sum   = [zeros(3) for _ in 1:N_MODELS]
    burnin_theta_count = zeros(Int, N_MODELS)

    adapt_states = [AdaptState(MODEL_DIM[m]) for m in 1:N_MODELS]

    within_accepts    = zeros(Int, N_MODELS)
    within_proposals  = zeros(Int, N_MODELS)
    between_accepts   = zeros(Int, N_MODELS, N_MODELS)
    between_proposals = zeros(Int, N_MODELS, N_MODELS)
    model_visit_counts = zeros(Int, N_MODELS)

    model_idx_out     = Vector{Int32}(undef, n_post)
    params_shared_out = Matrix{Float64}(undef, n_post, 3)
    param4_out        = Vector{Float64}(undef, n_post)

    post_i = 0
    for step in 1:n_total
        in_burnin = step <= config.n_burnin

        if step == pilot_update + 1
            proposals = ntuple(Val(N_MODELS)) do m
                if m == PL_IDX || !(m in config.active_models)
                    prior_proposals[m]                   
                elseif burnin_theta_count[m] >= 3
                    θ_mean = burnin_theta_sum[m] ./ burnin_theta_count[m]
                    pilot_param4(m, θ_mean, ha_meas, ha_uncert,
                                  freq, data, data_err, config.n_pilot)
                else
                    prior_proposals[m]                   
                end
            end
        end

        for _ in 1:config.n_within
            within_proposals[state.model] += 1
            state, acc = within_step(state, ha_meas, ha_uncert, freq, data, data_err,
                                       adapt_states[state.model], config)
            acc && (within_accepts[state.model] += 1)
        end

        prev_model = state.model
        j = rand(filter(!=(state.model), config.active_models))
        between_proposals[prev_model, j] += 1
        state, acc = between_step_to(state, j, ha_meas, ha_uncert,
                                       freq, data, data_err, config, proposals)
        acc && (between_accepts[prev_model, j] += 1)

        if in_burnin && step <= pilot_update
            burnin_theta_sum[state.model]   .+= state.params[1:3]
            burnin_theta_count[state.model]  += 1
        end

        if !in_burnin
            post_i += 1
            model_idx_out[post_i]        = state.model
            params_shared_out[post_i, :] = state.params[1:3]
            param4_out[post_i]           = length(state.params) == 4 ? state.params[4] : NaN
            model_visit_counts[state.model] += 1
        end
    end

    diag = RJDiagnostics(model_visit_counts, within_accepts, within_proposals,
                           between_accepts, between_proposals)
    return model_idx_out, params_shared_out, param4_out, diag
end

function run_rjmcmc_hex(fd::FluxData, dc_freq::AbstractVector;
    config::RJConfig  = RJConfig(),
    output_path::String)

    n_hex  = size(fd.dmatrix, 1)
    n_post = config.n_samples
    mkpath(output_path)

    all_model_idx     = Array{Int32,2}(undef,  n_hex, n_post)
    all_params_shared = Array{Float64,3}(undef, n_hex, n_post, 3)
    all_param4        = Array{Float64,2}(undef, n_hex, n_post)
    all_ha_samples    = Array{Float64,2}(undef, n_hex, n_post)
    all_visit_fracs   = Array{Float64,2}(undef, n_hex, N_MODELS)
    all_within_rates  = Array{Float64,2}(undef, n_hex, N_MODELS)
    all_between_rates = Array{Float64,2}(undef, n_hex, N_MODELS)

    nt = Threads.nthreads()
    println("Running RJMCMC on $nt thread$(nt > 1 ? "s" : "") ($n_hex bins)")

    prog = Progress(n_hex; desc="Running RJMCMC: ")

    Threads.@threads :dynamic for i in 1:n_hex
        ha_total_noise = sqrt(fd.dHa_noise[i]^2 + (0.3 * fd.dHa[i])^2)
        ha_meas   = max(fd.dHa[i], 0.0)
        ha_uncert = ha_total_noise > 0.0 ? ha_total_noise : 1.0

        data     = fd.dmatrix[i, :]
        data_err = fd.dnoisematrix[i, :]

        midx, pshared, p4, diag = run_rj_chain(data, data_err, dc_freq,
                                                  ha_meas, ha_uncert, config)
        all_model_idx[i, :]        = midx
        all_params_shared[i, :, :] = pshared
        all_param4[i, :]           = p4
        all_ha_samples[i, :]       = rand(Normal(ha_meas, ha_uncert), n_post)

        visit_total = sum(diag.model_visit_counts)
        all_visit_fracs[i, :] = diag.model_visit_counts ./ max(visit_total, 1)

        for m in 1:N_MODELS
            wp = diag.within_proposals[m]
            all_within_rates[i, m] = wp > 0 ? diag.within_accepts[m] / wp : NaN
            bp = sum(diag.between_proposals[m, :])
            all_between_rates[i, m] = bp > 0 ? sum(diag.between_accepts[m, :]) / bp : NaN
        end

        unvisited_active = [MODEL_NAMES[m] for m in config.active_models
                            if diag.model_visit_counts[m] == 0]
        if !isempty(unvisited_active)
            @warn "Bin $i: model(s) never visited post-burnin: $(join(unvisited_active, ", "))"
        end

        next!(prog)
    end

    print_rj_run_summary(all_visit_fracs, all_within_rates, all_between_rates,
                          config.active_models)

    hex_pos_mat = collect(hcat([collect(p) for p in fd.hex_pos]...)')
    h5open(joinpath(output_path, "rjsamples.h5"), "w") do hdf
        HDF5.attributes(hdf)["mode"] = "RJMCMC"
        write(hdf, "model_index",              all_model_idx)
        write(hdf, "params_shared",            all_params_shared)
        write(hdf, "param4",                   all_param4)
        write(hdf, "ha_samples",               all_ha_samples)
        write(hdf, "hex_pos",                  hex_pos_mat)
        write(hdf, "model_visit_fractions",    all_visit_fracs)
        write(hdf, "within_acceptance_rates",  all_within_rates)
        write(hdf, "between_acceptance_rates", all_between_rates)
        write(hdf, "active_models",            Int32.(config.active_models))
    end
    println("Saved RJMCMC samples to ", joinpath(output_path, "rjsamples.h5"))
end

function print_rj_run_summary(visit_fracs, within_rates, between_rates,
                               active_models = collect(1:N_MODELS))
    println("\n── RJMCMC run summary (means across bins) ──────────────────")
    @printf("  %-18s  %8s  %12s  %13s\n",
            "Model", "Visit %", "Within acc.", "Between acc.")
    for m in active_models
        vf = mean(visit_fracs[:, m]) * 100
        wr = mean(filter(isfinite, within_rates[:, m]))
        br = mean(filter(isfinite, between_rates[:, m]))
        @printf("  %-18s  %7.1f%%  %11.3f  %12.3f\n",
                MODEL_NAMES[m], vf, wr, br)
    end
    println("────────────────────────────────────────────────────────────\n")
end

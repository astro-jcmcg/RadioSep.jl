using AffineInvariantMCMC
using Distributions
using HDF5
using Statistics
using ProgressMeter

function run_mcmc_hex(fd::FluxData, dc_freq::AbstractVector;
    n_walkers::Int=100,
    n_steps::Int=2000,
    n_burnin::Int=400,
    model::Union{PowerLaw,CurvedPowerLaw,ffa,BrokenPowerLaw}=PowerLaw(),
    output_path::String)
    n_hex    = size(fd.dmatrix, 1)
    n_post   = (n_steps - n_burnin) * n_walkers
    n_params = model isa PowerLaw ? 3 : 4
    n_cols   = n_params + 1

    nt = Threads.nthreads()
    println("Running MCMC on $nt thread$(nt > 1 ? "s" : "") ($n_hex bins)")

    mkpath(output_path)

    all_samples = Array{Float64,3}(undef, n_hex, n_post, n_cols)

    prog = Progress(n_hex; desc="Running MCMC: ")

    Threads.@threads :dynamic for i in 1:n_hex
        ha_total_noise = sqrt(fd.dHa_noise[i]^2 + (0.3 * fd.dHa[i])^2)
        ha_meas   = max(fd.dHa[i], 0.0)
        ha_uncert = ha_total_noise > 0.0 ? ha_total_noise : 1.0

        data     = fd.dmatrix[i, :]
        data_err = fd.dnoisematrix[i, :]

        logpdf_fn = if model isa PowerLaw
            params -> lnpost(params, ha_meas, ha_uncert, dc_freq, data, data_err)
        else
            params -> lnpost(params, ha_meas, ha_uncert, dc_freq, data, data_err, model)
        end

        mean_flux   = mean(filter(isfinite, data))
        therm_init  = max(ha_meas, mean_flux * 0.1, 1e-9)
        ntherm_init = max(mean_flux - therm_init, therm_init * 0.1, 1e-9)
        spec_init   = -0.8

        t_walkers  = therm_init  .* exp.(0.3 .* randn(n_walkers))
        n_walkers_ = ntherm_init .* exp.(0.3 .* randn(n_walkers))
        s_walkers  = spec_init   .+  0.2 .* randn(n_walkers)

        p0 = if model isa CurvedPowerLaw
            b_walkers = 0.0 .+ 0.1 .* randn(n_walkers)
            vcat(t_walkers', n_walkers_', s_walkers', b_walkers')
        elseif model isa ffa
            tau_walkers = rand(Exponential(0.01), n_walkers)
            vcat(t_walkers', n_walkers_', s_walkers', tau_walkers')
        elseif model isa BrokenPowerLaw
            nu_b_walkers = exp.(log(1e8) .+ (log(1e10) - log(1e8)) .* rand(n_walkers))
            vcat(t_walkers', n_walkers_', s_walkers', nu_b_walkers')
        else
            vcat(t_walkers', n_walkers_', s_walkers')
        end

        chain, _   = AffineInvariantMCMC.sample(logpdf_fn, n_walkers, p0, n_steps, 1)
        flat       = collect(reshape(chain[:, :, (n_burnin+1):end], n_params, n_post)')
        ha_samples = rand(Normal(ha_meas, ha_uncert), n_post)

        all_samples[i, :, :] = hcat(flat, ha_samples)
        next!(prog)
    end

    hex_pos_mat = collect(hcat([collect(p) for p in fd.hex_pos]...)')
    h5open(joinpath(output_path, "samples.h5"), "w") do hdf
        write(hdf, "samples", all_samples)
        HDF5.attributes(hdf)["model"] = model isa CurvedPowerLaw  ? "CurvedPowerLaw"  :
                                        model isa ffa            ? "ffa"              :
                                        model isa BrokenPowerLaw ? "BrokenPowerLaw"  : "PowerLaw"
        write(hdf, "hex_pos", hex_pos_mat)
    end

    println("Saved samples to ", joinpath(output_path, "samples.h5"))
end

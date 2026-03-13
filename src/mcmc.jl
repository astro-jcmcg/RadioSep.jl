using AffineInvariantMCMC
using Distributions
using HDF5
using Statistics

function run_mcmc_hex(fd::FluxData, dc_freq::AbstractVector;
    n_walkers::Int=100,
    n_steps::Int=2000,
    n_burnin::Int=400,
    output_path::String)
    n_hex = size(fd.dmatrix, 1)
    n_post = (n_steps - n_burnin) * n_walkers

    mkpath(output_path)

    h5open(joinpath(output_path, "samples.h5"), "w") do hdf
        ds = create_dataset(hdf, "samples",
            datatype(Float64),
            dataspace(n_hex, n_post, 4))

        hex_pos_mat = collect(hcat([collect(p) for p in fd.hex_pos]...)')
        write(hdf, "hex_pos", hex_pos_mat)

        @showprogress "Running MCMC: " for i in 1:n_hex
            ha_total_noise = sqrt(fd.dHa_noise[i]^2 + (0.3 * fd.dHa[i])^2)
            ha_meas = max(fd.dHa[i], 0.0)
            ha_uncert = ha_total_noise > 0.0 ? ha_total_noise : 1.0

            data = fd.dmatrix[i, :]
            data_err = fd.dnoisematrix[i, :]

            logpdf_fn = params -> lnpost(params, ha_meas, ha_uncert,
                dc_freq, data, data_err)

            mean_flux = mean(filter(isfinite, data))
            therm_init = max(ha_meas, mean_flux * 0.1, 1e-9)
            ntherm_init = max(mean_flux - therm_init, therm_init * 0.1, 1e-9)
            spec_init = -0.8
            t_walkers = therm_init .* exp.(0.3 .* randn(n_walkers))
            n_walkers_ = ntherm_init .* exp.(0.3 .* randn(n_walkers))
            s_walkers = spec_init .+ 0.2 .* randn(n_walkers)

            p0 = vcat(t_walkers', n_walkers_', s_walkers')

            chain, _ = AffineInvariantMCMC.sample(logpdf_fn, n_walkers, p0, n_steps, 1)
            post_chain = chain[:, :, (n_burnin+1):end]
            post_steps = size(post_chain, 3)
            n_post_actual = post_steps * n_walkers
            flat = collect(reshape(post_chain, 3, n_post_actual)')
            ha_samples = rand(Normal(ha_meas, ha_uncert), n_post_actual)
            ds[i, :, :] = hcat(flat, ha_samples)
        end
    end

    println("Saved samples to ", joinpath(output_path, "samples.h5"))
end
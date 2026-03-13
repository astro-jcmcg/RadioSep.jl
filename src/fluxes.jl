using Statistics

function recover_fluxes(dc::DataCube, ha::HaData, hg::HexGrid,
    source_mask::BitMatrix)::FluxData
    n_hex = length(hg.pixel_members)
    n_freq = length(dc.freq)

    dmatrix = zeros(Float64, n_hex, n_freq)
    dnoisematrix = zeros(Float64, n_hex, n_freq)
    dHa = zeros(Float64, n_hex)
    dHa_noise = zeros(Float64, n_hex)
    area = zeros(Float64, n_hex)

    d2sr = (π / 180)^2

    @showprogress "Recovering fluxes: " for i in 1:n_hex
        sample_pixels = filter(px -> source_mask[px], hg.pixel_members[i])

        n_pix = length(sample_pixels)
        area[i] = n_pix * abs(dc.header["CDELT1"]) * dc.header["CDELT2"] * d2sr

        for k in 1:n_freq
            flux_sum = sum(v for px in sample_pixels for v in (dc.data[px, k],)
                           if isfinite(v); init=0.0)
            dmatrix[i, k] = flux_sum / dc.beam_correction[k]
            dnoisematrix[i, k] = sqrt((0.05 * dmatrix[i, k])^2 +
                                      (n_pix / dc.beam_correction[k]) * dc.noise[k]^2)
        end

        ha_sum = sum(v for px in sample_pixels for v in (ha.flux_scaled[px],)
                     if isfinite(v); init=0.0)
        dHa[i] = ha_sum / ha.beam_correction
        dHa_noise[i] = (n_pix / ha.beam_correction) * ha.noise
    end

    hex_pos = [(cx, cy) for (cx, cy) in hg.centers]

    return FluxData(dmatrix, dnoisematrix, dHa, dHa_noise, area, hex_pos)
end
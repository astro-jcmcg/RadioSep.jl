using SpecialFunctions

function bfield_revised(flux_density, region_area::Float64, freq::Float64,
    spectral_index, k::Float64,
    scale_height::Float64, disk_inclination::Float64)
    
    α = abs.(spectral_index)

    depth_los = (scale_height * 2.0 / cos(deg2rad(disk_inclination))) * 1e3 * 3.08568025e18  # cm

    const_factor = @. 4π * (2α + 1) * (k + 1) / (2α - 1)

    inten = flux_density ./ region_area
    inu = inten .* 1e-23
    ep = @. (1.5033e-3)^(1 - 2α)
    nu = @. (freq * 1e9)^α
    c1 = @. (2 * 6.26428e18)^α
    c4i = @. (2.0 / 3.0)^((α + 1) / 2.0)
    c3 = 1.86558e-23
    c = @. 0.25 * (α + 5.0 / 3.0) / (α + 1.0)
    g1 = @. gamma((3α + 1) / 6.0)
    g2 = @. gamma((3α + 5) / 6.0)
    c2 = @. c * g1 * g2 * c3

    return @. (const_factor * inu * ep * nu / (c2 * depth_los * c4i * c1))^(1.0 / (α + 3.0))
end

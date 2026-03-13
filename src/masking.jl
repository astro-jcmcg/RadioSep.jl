function gen_mask(dc::DataCube; threshold::Float64=5.0)::BitMatrix
    ny, nx, n_freq = size(dc.data)
    mask_sum = zeros(Int, ny, nx)
    for k in 1:n_freq
        noise_lim = threshold * dc.noise[k]
        for col in 1:nx, row in 1:ny
            if dc.data[row, col, k] >= noise_lim
                mask_sum[row, col] += 1
            end
        end
    end
    return mask_sum .== n_freq
end
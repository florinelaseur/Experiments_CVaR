using TestItems: @testitem

@testitem "sobol_gaussian_samples shape" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 7
    N = 128
    μ = zeros(d)
    Σ = Matrix{Float64}(I, d, d)
    Z = sobol_gaussian_samples(N, μ, Σ; seed=1)
    @test size(Z) == (d, N)
    @test eltype(Z) == Float64
    @test all(isfinite, Z)
end

@testitem "sobol_gaussian_samples mean recovery" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 7
    N = 1024
    μ = [10.0, -5.0, 3.0, 0.0, 7.0, -2.0, 1.5]
    Σ = Matrix{Float64}(I, d, d) .* 2.0
    Z = sobol_gaussian_samples(N, μ, Σ; seed=1)
    sample_mean = vec(mean(Z; dims=2))
    @test all(abs.(sample_mean .- μ) .< 0.1)
end

@testitem "sobol_gaussian_samples covariance recovery" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 4
    N = 1024
    μ = zeros(d)
    # Non-trivial PSD covariance with off-diagonal structure.
    A = [1.0 0.3 0.0 0.1;
         0.3 2.0 0.5 0.0;
         0.0 0.5 1.5 0.2;
         0.1 0.0 0.2 1.0]
    Σ = A * A'                                  # ensure PSD
    Z = sobol_gaussian_samples(N, μ, Σ; seed=1)
    sample_cov = cov(Z'; dims=1)
    @test maximum(abs.(sample_cov .- Σ)) < 0.5  # generous; QMC, not iid
end

@testitem "sobol_gaussian_samples determinism" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 5
    N = 256
    μ = collect(1.0:d)
    Σ = Matrix{Float64}(I, d, d)
    Z1 = sobol_gaussian_samples(N, μ, Σ; seed=42)
    Z2 = sobol_gaussian_samples(N, μ, Σ; seed=42)
    @test Z1 == Z2
end

@testitem "sobol_gaussian_samples independence" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 5
    N = 256
    μ = collect(1.0:d)
    Σ = Matrix{Float64}(I, d, d)
    Z1 = sobol_gaussian_samples(N, μ, Σ; seed=1)
    Z2 = sobol_gaussian_samples(N, μ, Σ; seed=2)
    @test Z1 != Z2
end

@testitem "sobol_gaussian_samples_nonneg clip enforces bounds" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 7
    N = 256
    μ = zeros(d)                              # half the mass would be negative
    Σ = Matrix{Float64}(I, d, d)
    ub = fill(0.5, d)
    Z = sobol_gaussian_samples_nonneg(N, μ, Σ; mode=:clip, ub=ub, seed=1)
    @test size(Z) == (d, N)
    @test all(Z .>= 0.0)
    @test all(Z .<= reshape(ub, :, 1))
end

@testitem "sobol_gaussian_samples_nonneg reject enforces bounds AND preserves count" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 4
    N = 128
    # μ chosen so ~30–40% of samples have a negative coord (1σ below 0 on dim 1).
    μ = [1.0, 1.5, 2.0, 2.5]
    Σ = Matrix{Float64}(I, d, d)
    ub = fill(10.0, d)
    Z = sobol_gaussian_samples_nonneg(N, μ, Σ; mode=:reject, ub=ub, oversample=4.0, seed=1)
    @test size(Z) == (d, N)
    @test all(Z .>= 0.0)
    @test all(Z .<= reshape(ub, :, 1))
end

@testitem "shrink_covariance endpoints" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    Σ = [4.0  1.0  0.5;
         1.0  3.0  0.2;
         0.5  0.2  2.0]
    Σ_emp  = shrink_covariance(Σ; α=0.0)
    Σ_diag = shrink_covariance(Σ; α=1.0)
    @test Σ_emp ≈ Σ
    @test Σ_diag ≈ Diagonal(diag(Σ))
end

@testitem "shrink_covariance convexity" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    Σ = [4.0  1.0  0.5;
         1.0  3.0  0.2;
         0.5  0.2  2.0]
    expected = 0.5 .* Σ .+ 0.5 .* Diagonal(diag(Σ))
    @test shrink_covariance(Σ; α=0.5) ≈ expected
end

@testitem "sobol_gaussian_samples rank-deficient Σ" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 4
    N = 64
    μ = zeros(d)
    Σ = ones(d, d)                            # rank 1 — degenerate without jitter
    Z = sobol_gaussian_samples(N, μ, Σ; seed=1)   # default jitter must rescue it
    @test size(Z) == (d, N)
    @test all(isfinite, Z)
end

@testitem "scrambled_sobol_uniform_reject_to_target hits target, all accepted, in box" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 3
    N = 64
    lb = zeros(d); ub = fill(10.0, d)
    accept = x -> sum(x) >= 12.0                      # ~majority of the cube
    res = scrambled_sobol_uniform_reject_to_target(N, lb, ub; accept=accept, seed=1)
    @test size(res.samples) == (d, N)
    @test res.n_drawn >= N
    @test all(res.samples .>= 0.0) && all(res.samples .<= 10.0)
    @test all(sum(res.samples[:, j]) >= 12.0 for j in 1:N)   # every kept sample passes
end

@testitem "sobol_gaussian_reject_to_target honors box + accept, hits target" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    d = 4
    N = 64
    μ = fill(2.0, d); Σ = Matrix{Float64}(I, d, d)
    ub = fill(6.0, d)
    accept = x -> x[1] >= 1.0
    res = sobol_gaussian_reject_to_target(N, μ, Σ; accept=accept, ub=ub, seed=1)
    @test size(res.samples) == (d, N)
    @test all(res.samples .>= 0.0) && all(res.samples .<= reshape(ub, :, 1))
    @test all(res.samples[1, j] >= 1.0 for j in 1:N)
end

@testitem "reject_to_target is deterministic per seed" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    lb = zeros(3); ub = fill(10.0, 3)
    accept = x -> sum(x) >= 12.0
    a = scrambled_sobol_uniform_reject_to_target(40, lb, ub; accept=accept, seed=7)
    b = scrambled_sobol_uniform_reject_to_target(40, lb, ub; accept=accept, seed=7)
    @test a.samples == b.samples
    @test a.n_drawn == b.n_drawn
end

@testitem "reject_to_target returns fewer than N (no throw) when accept is unsatisfiable" setup = [SamplingSetup] tags = [:sampling, :unit] begin
    lb = zeros(2); ub = fill(1.0, 2)
    res = scrambled_sobol_uniform_reject_to_target(10, lb, ub; accept=(x -> false), max_draws=50)
    @test size(res.samples, 2) < 10               # cap hit → fewer, but no error
    @test res.n_drawn <= 128                       # soft cap (rounds up to a power of two ≥ max_draws)
end

# Quasi-random Gaussian sampling for SD-screening.
#
# Standalone — depends only on QuasiMonteCarlo, SpecialFunctions, LinearAlgebra,
# Statistics, Random. No JuMP/Tulipa/DuckDB so it can be unit-tested without
# spinning up a solver. `src/utils.jl` includes this file so existing
# `include(.../utils.jl)` callers pick the functions up unchanged.

using QuasiMonteCarlo: QuasiMonteCarlo
using LinearAlgebra: cholesky, Symmetric, Diagonal, diag, I
using Random: Random, Xoshiro
using SpecialFunctions: erfinv
using Statistics: Statistics

"""Standard-normal quantile Φ⁻¹(u) for uniform `u ∈ (0, 1)`."""
_standard_normal_quantile(u::Real) = √2 * erfinv(2u - 1)

"""
    sobol_gaussian_samples(N, μ, Σ; scramble=true, seed=1, jitter=1e-8)

Generate `N` quasi-random samples from `N(μ, Σ)` using Sobol' points pushed
through the standard-normal inverse CDF and a Cholesky transform. Returns a
`d × N` matrix; each column is one sample (matches the existing
`eachcol(samples)` consumer in `stochastic_dominance.jl`).

Pipeline:
  1. Sobol' uniforms `U ∈ [0,1]^{d×N}`, optionally Owen-scrambled.
  2. Inverse normal CDF via `√2·erfinv(2u − 1)`; `u` is clamped away from
     `{0, 1}` to avoid `±Inf`.
  3. Cholesky factor `L` of `Symmetric(Σ + jitter·I)` (jitter protects against
     numerical PSD violations on empirical covariances).
  4. `Z = μ .+ L * W`.
"""
function sobol_gaussian_samples(
    N::Int,
    μ::AbstractVector,
    Σ::AbstractMatrix;
    scramble::Bool=true,
    seed::Int=1,
    jitter::Real=1e-8,
)
    d = length(μ)
    size(Σ) == (d, d) || error("Σ must be $d×$d, got $(size(Σ))")

    sampler = scramble ?
        QuasiMonteCarlo.SobolSample(;
            R=QuasiMonteCarlo.OwenScramble(base=2, pad=32, rng=Xoshiro(seed)),
        ) :
        QuasiMonteCarlo.SobolSample()
    U = QuasiMonteCarlo.sample(N, zeros(d), ones(d), sampler)

    eps_safe = 1e-12
    U_safe = clamp.(U, eps_safe, 1 - eps_safe)
    W = _standard_normal_quantile.(U_safe)

    L = cholesky(Symmetric(Matrix(Σ) + jitter * I)).L

    return μ .+ L * W
end

"""
    sobol_gaussian_samples_nonneg(N, μ, Σ; mode=:clip, ub=nothing, ...)

Wrapper around [`sobol_gaussian_samples`](@ref) enforcing non-negativity (and
optional per-coordinate upper bound `ub`).

Modes:
  * `:clip`    — clamp coord-wise to `[0, ub_k]`. Fast; distorts the marginals
                 (mass that would have been negative piles up at 0).
  * `:reflect` — reflect via `z ↦ |z|`. Preserves total probability but
                 distorts the distribution.
  * `:reject`  — over-generate by a factor of `oversample`, drop any sample
                 with a negative or out-of-bounds coordinate, keep the first
                 `N` survivors. Preserves the conditional distribution exactly
                 but loses Sobol' low-discrepancy structure of the survivors.

The optional `accept` predicate (`:reject` mode only) additionally drops any
sample column `x` for which `accept(x)` is `false` — e.g. an adequacy-cut check
`x -> adequacy_verdict(x, cuts_list).passed`, so that every kept sample satisfies
the necessary feasibility condition.
"""
function sobol_gaussian_samples_nonneg(
    N::Int,
    μ::AbstractVector,
    Σ::AbstractMatrix;
    mode::Symbol=:clip,
    ub::Union{Nothing,AbstractVector}=nothing,
    scramble::Bool=true,
    seed::Int=1,
    jitter::Real=1e-8,
    oversample::Real=3.0,
    accept::Union{Nothing,Function}=nothing,
)
    if mode === :clip
        Z = sobol_gaussian_samples(N, μ, Σ; scramble, seed, jitter)
        Z .= max.(Z, 0.0)
        isnothing(ub) || (Z .= min.(Z, ub))
        return Z
    elseif mode === :reflect
        Z = sobol_gaussian_samples(N, μ, Σ; scramble, seed, jitter)
        Z .= abs.(Z)
        isnothing(ub) || (Z .= min.(Z, ub))
        return Z
    elseif mode === :reject
        Ngen = ceil(Int, oversample * N)
        Z = sobol_gaussian_samples(Ngen, μ, Σ; scramble, seed, jitter)
        keep = trues(Ngen)
        @inbounds for k in 1:Ngen
            col = view(Z, :, k)
            if any(<(0.0), col)
                keep[k] = false
                continue
            end
            if !isnothing(ub) && any(i -> col[i] > ub[i], eachindex(col))
                keep[k] = false
                continue
            end
            if !isnothing(accept) && !accept(col)
                keep[k] = false
            end
        end
        idx = findall(keep)
        length(idx) < N && error(
            "Only $(length(idx))/$N feasible samples after oversample=$oversample; " *
            "increase oversample or relax bounds.",
        )
        return Z[:, idx[1:N]]
    else
        throw(ArgumentError("mode must be :clip, :reflect, or :reject (got :$mode)"))
    end
end

"""
    shrink_covariance(Σ_emp; α=0.2, target=:diagonal)

Linear shrinkage of an empirical covariance toward a structured target.
Returns `(1 − α) · Σ_emp + α · T`.

  * `target=:diagonal` → `T = Diagonal(diag(Σ_emp))` (zeros off-diagonals).
  * `target=:identity` → `T = mean(diag(Σ_emp)) · I` (also equalises variances).

With ~144 samples in 7 dimensions the off-diagonals of the empirical Σ are
noisy; a moderate `α ∈ [0.1, 0.3]` dampens those without destroying the
marginal-variance information.
"""
function shrink_covariance(Σ_emp::AbstractMatrix; α::Real=0.2, target::Symbol=:diagonal)
    T = if target === :diagonal
        Matrix(Diagonal(diag(Σ_emp)))
    elseif target === :identity
        Statistics.mean(diag(Σ_emp)) * Matrix(I, size(Σ_emp, 1), size(Σ_emp, 1))
    else
        throw(ArgumentError("target must be :diagonal or :identity (got :$target)"))
    end
    return (1 - α) * Σ_emp + α * T
end

# ---- Reject-to-target sampling: keep drawing until N samples are accepted ----
#
# Unlike `:reject` mode (fixed oversample, errors if short), these grow the draw
# budget until at least `N` samples pass `accept`, so callers always get a full
# pool of "probable" candidates (e.g. samples that satisfy the adequacy cuts).
# Sampling is cheap vs the downstream LP, so regenerating the batch as it grows is
# fine and keeps the result deterministic for a given seed.

"""
    _reject_to_target(gen_batch, accept, N; init_oversample=3.0, max_draws=200_000)

`gen_batch(M)` returns a `d × M` candidate matrix (deterministic in `M`). Grows `M`
until ≥ `N` columns satisfy `accept`, or `max_draws` is hit (then warns). Returns
`(kept_indices::Vector{Int}, n_drawn::Int, batch::Matrix)`.
"""
function _reject_to_target(gen_batch, accept, N::Int; init_oversample::Real=3.0, max_draws::Int=200_000)
    # Scrambled-Sobol nets require a power-of-two point count, so the batch size is
    # always rounded up to a power of two and doubled (stays a power of two).
    M = nextpow(2, max(N, ceil(Int, init_oversample * N)))
    while true
        batch = gen_batch(M)
        kept = Int[j for j in 1:size(batch, 2) if accept(@view batch[:, j])]
        if length(kept) >= N || M >= max_draws
            length(kept) < N && @warn "reject_to_target: only $(length(kept))/$N accepted after $M draws (≥ max_draws=$max_draws); returning $(length(kept))."
            return (kept, M, batch)
        end
        M *= 2
    end
end

"""
    sobol_gaussian_reject_to_target(N, μ, Σ; accept, ub=nothing, seed=1, ...) -> (samples, n_drawn)

Scrambled-Sobol draws from `N(μ, Σ)`, keeping the first `N` that are ≥ 0, ≤ `ub`
(if given), and satisfy `accept` (a predicate on a sample vector), growing the draw
budget until `N` are found. `samples` is `d × N` (or fewer if `max_draws` is hit);
`n_drawn` is the total candidates generated (for acceptance-rate reporting).
"""
function sobol_gaussian_reject_to_target(
    N::Int, μ::AbstractVector, Σ::AbstractMatrix;
    accept::Function, ub::Union{Nothing,AbstractVector}=nothing,
    seed::Int=1, jitter::Real=1e-8, init_oversample::Real=3.0, max_draws::Int=200_000,
)
    gen = M -> sobol_gaussian_samples(M, μ, Σ; scramble=true, seed=seed, jitter=jitter)
    full_accept = x -> all(>=(0.0), x) &&
        (isnothing(ub) || all(i -> x[i] <= ub[i], eachindex(x))) &&
        accept(x)
    kept, n_drawn, batch = _reject_to_target(gen, full_accept, N; init_oversample, max_draws)
    take = min(N, length(kept))
    return (samples=batch[:, kept[1:take]], n_drawn=n_drawn)
end

"""
    scrambled_sobol_uniform_reject_to_target(N, lb, ub; accept, seed=1, ...) -> (samples, n_drawn)

Scrambled-Sobol uniform draws over `[lb, ub]` (no Gaussian centre), keeping the
first `N` that satisfy `accept`. Same return shape as
[`sobol_gaussian_reject_to_target`](@ref).
"""
function scrambled_sobol_uniform_reject_to_target(
    N::Int, lb::AbstractVector, ub::AbstractVector;
    accept::Function, seed::Int=1, init_oversample::Real=3.0, max_draws::Int=200_000,
)
    gen = M -> QuasiMonteCarlo.sample(
        M, lb, ub,
        QuasiMonteCarlo.SobolSample(; R=QuasiMonteCarlo.OwenScramble(base=2, pad=32, rng=Xoshiro(seed))),
    )
    kept, n_drawn, batch = _reject_to_target(gen, accept, N; init_oversample, max_draws)
    take = min(N, length(kept))
    return (samples=batch[:, kept[1:take]], n_drawn=n_drawn)
end

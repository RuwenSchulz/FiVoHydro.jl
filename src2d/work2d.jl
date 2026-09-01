# ==============================================================================
# src2d/work2d.jl — preallocated work arrays.
#
# Counterpart of src/work.jl. Everything is indexed by the FLAT cell index, so
# these arrays have exactly the shapes the 1-D `Work1D` has — that is what lets
# the SSPRK stages and the MOOD machinery be reused without reinterpreting `i`.
#
# The only structural additions are the second reconstruction/flux direction
# (`*y` beside `*x`) and the fourth reconstructed primitive: 1-D reconstructs
# (yT, φ, y=asinh u^r), 2-D reconstructs (yT, φ, u^x, u^y).
# ==============================================================================

const NPRIM_REC_2D = 4      # yT, φ, ux, uy

mutable struct Work2D
    # cell-centered primitives
    yT::Vector{Float64}
    phi::Vector{Float64}
    mu::Vector{Float64}
    alpha::Vector{Float64}
    ux::Vector{Float64}
    uy::Vector{Float64}
    P::Vector{Float64}
    n::Vector{Float64}
    e::Vector{Float64}
    ok::BitVector

    # Newton warm starts (previous-step values; the production seed path)
    x0_yT::Vector{Float64}
    x0_phi::Vector{Float64}
    x0_ux::Vector{Float64}
    x0_uy::Vector{Float64}

    # previous-substep values for the covariant NS drives (cf. Work1D.y_prev /
    # alpha_prev, whose ∂_τ pieces are worth a factor ~2 in ν — see §3 of
    # TWOD_PROGRAM.md)
    ux_prev::Vector{Float64}
    uy_prev::Vector{Float64}
    alpha_prev::Vector{Float64}

    # MUSCL slopes and face states, per direction
    sigx::Matrix{Float64}
    sigy::Matrix{Float64}
    ULpx::Matrix{Float64}
    URpx::Matrix{Float64}
    ULpy::Matrix{Float64}
    URpy::Matrix{Float64}

    ULcx::Matrix{Float64}
    URcx::Matrix{Float64}
    ULcy::Matrix{Float64}
    URcy::Matrix{Float64}

    Fhx::Matrix{Float64}
    Fhy::Matrix{Float64}

    S::Matrix{Float64}
    diss_snap::Matrix{Float64}   # pre-relaxation copy, for the upwind stencil
    k::Matrix{Float64}
    U1::Matrix{Float64}
    U2::Matrix{Float64}

    bad::BitVector
    bad_tmp::BitVector

    # per-thread scratch
    tmpFL::Vector{Vector{Float64}}
    tmpFR::Vector{Vector{Float64}}

    # kinematics used by the relaxation substep
    theta::Vector{Float64}
    vxC::Vector{Float64}
    vyC::Vector{Float64}

    # limited face-reconstructed gradient of alpha (see dissipation2d.jl)
    slope_tmp::Vector{Float64}
    alphaFx::Vector{Float64}
    alphaFy::Vector{Float64}
    gradAx::Vector{Float64}
    gradAy::Vector{Float64}

    # shear-constraint monitor (gate G1): per-cell tracelessness residual
    shear_res::Vector{Float64}

    # diagnostics reduction buffers
    primfail_tls::Vector{Int}
    vacuum_tls::Vector{Int}
    primfail_i_tls::Vector{Int}
    primfail_reason_tls::Vector{PrimRecReason}
    primfail_resnorm_tls::Vector{Float64}
    amax_tls::Vector{Float64}
end

# maxthreadid(): see the note on IdealPrimRec2D in primrec2d.jl.
function Work2D(nvar::Int, Ntot::Int; nthreads::Int = Threads.maxthreadid())
    vec()  = zeros(Float64, Ntot)
    mat()  = zeros(Float64, nvar, Ntot)
    pmat() = zeros(Float64, NPRIM_REC_2D, Ntot)

    Work2D(
        vec(), vec(), vec(), vec(), vec(), vec(), vec(), vec(), vec(), falses(Ntot),
        vec(), vec(), vec(), vec(),
        vec(), vec(), vec(),
        pmat(), pmat(), pmat(), pmat(), pmat(), pmat(),
        mat(), mat(), mat(), mat(),
        mat(), mat(),
        mat(), mat(), mat(), mat(), mat(),
        falses(Ntot), falses(Ntot),
        [zeros(Float64, nvar) for _ in 1:nthreads],
        [zeros(Float64, nvar) for _ in 1:nthreads],
        vec(), vec(), vec(),
        vec(), vec(), vec(), vec(), vec(),
        vec(),
        zeros(Int, nthreads), zeros(Int, nthreads), zeros(Int, nthreads),
        fill(PRR_UNSET, nthreads), zeros(Float64, nthreads),
        zeros(Float64, nthreads),
    )
end

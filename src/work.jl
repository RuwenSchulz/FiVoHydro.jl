# ==============================================================================
# src/work.jl
#
# Work arrays and initialization.
# ==============================================================================

mutable struct Work1D
    yT::Vector{Float64}
    phi::Vector{Float64}
    mu::Vector{Float64}
    alpha::Vector{Float64}
    y::Vector{Float64}
    P::Vector{Float64}
    n::Vector{Float64}
    e::Vector{Float64}
    ok::BitVector

    x0_yT::Vector{Float64}
    x0_phi::Vector{Float64}
    x0_y::Vector{Float64}

    σp::Matrix{Float64}
    ULp::Matrix{Float64}
    URp::Matrix{Float64}

    ULc::Matrix{Float64}
    URc::Matrix{Float64}
    Fh::Matrix{Float64}

    S::Matrix{Float64}
    k::Matrix{Float64}
    U1::Matrix{Float64}
    U2::Matrix{Float64}

    bad::BitVector
    bad_tmp::BitVector

    tmpFL::Vector{Vector{Float64}}
    tmpFR::Vector{Vector{Float64}}

    theta::Vector{Float64}
    vC::Vector{Float64}
    vF::Vector{Float64}
    dvdr::Vector{Float64}

    alphaF::Vector{Float64}
    gradAlpha::Vector{Float64}

    alpha_tmp::Vector{Float64}
    nur_tmp::Vector{Float64}
    n_tmp::Vector{Float64}

    visc_tmp1::Vector{Float64}
    visc_tmp2::Vector{Float64}
    visc_tmp3::Vector{Float64}

    primfail_tls::Vector{Int}
    bad_tls::Vector{Bool}
    amax_tls::Vector{Float64}
    kmax_tls::Vector{Float64}
end

function make_work(U)
    Nvars, Ntot = size(U)
    nt = Threads.nthreads()
    return Work1D(
        zeros(Ntot), zeros(Ntot), zeros(Ntot), zeros(Ntot), zeros(Ntot),
        zeros(Ntot), zeros(Ntot), zeros(Ntot), falses(Ntot),

        fill(log(0.25), Ntot), zeros(Ntot), zeros(Ntot),

        zeros(3, Ntot),
        zeros(3, Ntot-1),
        zeros(3, Ntot-1),

        zeros(Nvars, Ntot-1),
        zeros(Nvars, Ntot-1),
        zeros(Nvars, Ntot-1),

        zeros(Nvars, Ntot),
        zeros(Nvars, Ntot),
        similar(U),
        similar(U),

        falses(Ntot),
        falses(Ntot),

        [zeros(Nvars) for _ in 1:nt],
        [zeros(Nvars) for _ in 1:nt],

        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),

        zeros(Ntot-1),
        zeros(Ntot),

        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),

        zeros(Ntot),
        zeros(Ntot),
        zeros(Ntot),

        zeros(Int, nt),
        fill(false, nt),
        fill(1e-30, nt),
        fill(0.0, nt),
    )
end

# ------------------------------------------------------------
# Primitive cache builder (for dt-from-work correctness at t=τ0)
# ------------------------------------------------------------
function prime_work_from_U!(work::Work1D, U, grid, τ, model::IdealDiffViscModel)
    L   = layout(model)
    eos = model.eos
    Ntot = size(U,2)

    Threads.@threads for i in 1:Ntot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]

        D  = U[L.iDtau,i] / max(τ, 1e-50)
        Sr = U[L.iSr,i]
        E  = U[L.iE,i]

        nur_phys = (L.hasNur ? phys_from_stored(U[L.iNur,i]) : 0.0)
        Pi_phys  = (L.hasPi  ? phys_from_stored(U[L.iPi,i])  : 0.0)
        piR_phys = (L.hasPiR ? phys_from_stored(U[L.iPiR,i]) : 0.0)

        T, μ, ur, n, e, P, ok = cons_to_prim_ideal_phi_diff_visc!(
            wpr, D, Sr, E, nur_phys, Pi_phys, piR_phys, eos;
            yT0=work.x0_yT[i], φ0=work.x0_phi[i], y0=work.x0_y[i],
            maxit=40
        )

        if ok
            Tm = max(T, T_MIN)
            φ  = (μ - hq_mass(eos)) / Tm

            work.yT[i]    = log(Tm)
            work.phi[i]   = φ
            work.mu[i]    = μ
            work.alpha[i] = μ / Tm
            work.y[i]     = asinh(ur)

            work.P[i]     = P
            work.n[i]     = n
            work.e[i]     = e
            work.ok[i]    = true

            uτ = sqrt(1 + ur^2)
            work.vC[i]    = ur / max(uτ, 1e-50)

            work.x0_yT[i]  = work.yT[i]
            work.x0_phi[i] = work.phi[i]
            work.x0_y[i]   = work.y[i]
        else
            # consistent vacuum fallback
            Tv = T_MIN
            μv = hq_mass(eos)
            Pv, _, ev = eos_Pne(Tv, μv, eos)

            work.yT[i]    = log(Tv)
            work.phi[i]   = 0.0
            work.mu[i]    = μv
            work.alpha[i] = μv / Tv
            work.y[i]     = 0.0

            work.P[i]     = Pv
            work.n[i]     = 0.0
            work.e[i]     = max(ev, 0.0)
            work.ok[i]    = true

            work.vC[i]    = 0.0
        end
    end

    # axis symmetry cache (safe even for ghosts)
    i0 = grid.nghost + 1
    work.y[i0]  = 0.0
    work.vC[i0] = 0.0
    return nothing
end

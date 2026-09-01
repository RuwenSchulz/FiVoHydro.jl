# ==============================================================================
# src2d/rhs2d.jl — the 2+1D right-hand side.
#
# Counterpart of src/rhs.jl. Structure is deliberately the same so the two can be
# read side by side:
#
#   apply BCs -> cons->prim everywhere -> MUSCL reconstruct -> HLLE fluxes
#             -> geometric sources -> assemble dU
#
# UNSPLIT: the x- and y-flux divergences are summed into one dU per RK stage.
#
# Differences from 1-D, all of them simplifications except the second direction:
#   * the flux divergence is `(Fx[i] - Fx[i-Nytot])/dx + (Fy[i] - Fy[i-1])/dy`,
#     with no `1/r ∂_r(r F)` weighting;
#   * no special first-physical-cell treatment (no axis);
#   * the momentum source is just `-S^i/τ`.
# ==============================================================================

"""
    compute_divv_2d!(work, g)

Spatial 3-divergence `∇·v = ∂_x v^x + ∂_y v^y` on cell centres, from
face-averaged velocities. This is the quantity the advected dissipative scalars
need: they are transported as SCALARS, so writing `∂_τ q + ∇·(qv) = q ∇·v`
recovers `∂_τ q + v·∇q = 0`. In 1-D the same object is `theta - 1/τ`
(`src/rhs.jl`, `divrv`); here there is no `1/τ` piece to remove because `θ`'s
Bjorken part is not folded into the Cartesian divergence.
"""
function compute_divv_2d!(work::Work2D, g::Grid2D)
    fill!(work.theta, 0.0)
    invdx = 1/g.dx; invdy = 1/g.dy
    @inbounds Threads.@threads for ix in 2:(g.Nxtot-1)
        for iy in 2:(g.Nytot-1)
            i = lin(g, ix, iy)
            ixm = i - g.Nytot; ixp = i + g.Nytot
            work.theta[i] = 0.5*(work.vxC[ixp] - work.vxC[ixm])*invdx +
                            0.5*(work.vyC[i+1] - work.vyC[i-1])*invdy
        end
    end
    return nothing
end

# ------------------------------------------------------------------------------

"""
    update_primitives_2d!(U, g, τ, model, work; bc=:outflow)

Apply boundary conditions and recover primitives on every cell into `work`.

Split out of `rhs_2d!` so the operator-split relaxation can run on CURRENT
primitives. The 1-D solver relaxes using whatever `work` was left by the last
`rhs!` of the RK step — i.e. the primitives of the intermediate stage, not of the
accepted state. That is an O(Δτ) staleness inside an already first-order
splitting, but the shear NS target depends on velocity GRADIENTS, so it is worth
one extra recovery sweep per step to remove it here.
"""
function update_primitives_2d!(U::AbstractMatrix, g::Grid2D, τ::Float64,
                               model::IdealDiffVisc2DModel, work::Work2D;
                               bc::Symbol = :outflow)
    apply_bc_2d!(U, g, model.layout; bc = bc)
    L = model.layout; eos = model.eos
    fill!(work.primfail_tls, 0); fill!(work.vacuum_tls, 0)
    _primitive_pass!(U, g, τ, model, work, L, eos)
    return sum(work.primfail_tls)
end

function _primitive_pass!(U::AbstractMatrix, g::Grid2D, τ::Float64,
                          model::IdealDiffVisc2DModel, work::Work2D, L, eos)
    @inbounds Threads.@threads for ix in 1:g.Nxtot
        tid = Threads.threadid()
        wpr = model.primrec.work[tid]
        for iy in 1:g.Nytot
            i = lin(g, ix, iy)

            D  = safe_div(U[L.iDtau,i], τ)
            Sx = U[L.iSx,i]; Sy = U[L.iSy,i]; E = U[L.iE,i]
            nux, nuy, Pi, pixx, pixy, piyy, pieta = dissipatives_at(U, i, L)

            T, μ, ux, uy, n, e, P, ok = cons_to_prim_2d!(
                wpr, D, Sx, Sy, E, nux, nuy, Pi, pixx, pixy, piyy, pieta, τ, eos;
                yT0 = work.x0_yT[i], φ0 = work.x0_phi[i],
                ux0 = work.x0_ux[i], uy0 = work.x0_uy[i])

            if !ok
                # PRR_VACUUM is an EXPECTED branch, not a failure: outside the
                # fireball there is no state to recover. Counting it as primfail
                # made a first realistic run look like a 1.5-3% failure rate when
                # every flagged cell was simply vacuum (measured: T_max over
                # flagged cells = T_MIN in every sector configuration).
                if wpr.last_reason == PRR_VACUUM
                    work.vacuum_tls[tid] += 1
                else
                    work.primfail_tls[tid] += 1
                    # keep one representative failure per thread for diagnosis
                    if work.primfail_i_tls[tid] == 0
                        work.primfail_i_tls[tid] = i
                        work.primfail_reason_tls[tid] = wpr.last_reason
                        work.primfail_resnorm_tls[tid] = wpr.last_resnorm
                    end
                end
                # Same fallback as the 1-D solver: drop the cell to a cold vacuum
                # state rather than propagating a non-finite primitive. MOOD sees
                # the resulting stencil and repairs.
                T = T_MIN; μ = 0.0; ux = 0.0; uy = 0.0
                P, n, e = eos_Pne(T, μ, eos)
                e = max(e, 0.0); n = 0.0
            end

            Tm = max(T, T_MIN)
            work.yT[i]    = log(Tm)
            work.phi[i]   = clamp((μ - hq_mass(eos))/Tm, -PHI_CAP, PHI_CAP)
            work.mu[i]    = μ
            work.alpha[i] = μ/Tm
            work.ux[i]    = ux
            work.uy[i]    = uy
            work.P[i]     = P
            work.n[i]     = n
            work.e[i]     = e
            work.ok[i]    = ok

            uτ = sqrt(1 + ux*ux + uy*uy)
            invuτ = safe_inv(uτ)
            work.vxC[i] = ux*invuτ
            work.vyC[i] = uy*invuτ

            # Warm start for the next step. On FAILURE store NaN, not the
            # fallback state: the fallback is T_MIN, and log(T_MIN) is a perfectly
            # finite number, so `cons_to_prim_2d!` would accept it as a guess and
            # skip `seed_from_conserved_2d` — starting the next Newton 37 e-folds
            # below the answer. That makes a single failure self-sustaining, and
            # was the whole of the residual primfail population in the G4 run
            # (every one of them in the r > 20 corners, all reporting
            # PRR_RESIDUAL_TOO_LARGE at exactly res = 1.0, i.e. e recovered as 0).
            # NaN forces a fresh seed from the conserved variables.
            if ok
                work.x0_yT[i]  = work.yT[i]
                work.x0_phi[i] = work.phi[i]
                work.x0_ux[i]  = ux
                work.x0_uy[i]  = uy
            else
                work.x0_yT[i]  = NaN
                work.x0_phi[i] = NaN
                work.x0_ux[i]  = NaN
                work.x0_uy[i]  = NaN
            end

            if L.hasShear
                _, rel = shear_constraint_residual_2d(ux, uy, uτ, pixx, pixy, piyy, pieta)
                work.shear_res[i] = rel
            end
        end
    end

    return nothing
end

# ------------------------------------------------------------------------------

"""
    rhs_2d!(dU, U, g, τ, model, work; bc=:outflow)

Assemble `dU` for the conserved + advected state `U`. Returns `true`.
"""
function rhs_2d!(dU::AbstractMatrix, U::AbstractMatrix, g::Grid2D, τ::Float64,
                 model::IdealDiffVisc2DModel, work::Work2D;
                 bc::Symbol = :outflow, Emin::Float64 = E_FLOOR)

    apply_bc_2d!(U, g, model.layout; bc = bc)

    L   = model.layout
    eos = model.eos
    nv  = nvars(L)
    ng  = g.nghost
    Nyt = g.Nytot

    fill!(work.primfail_tls, 0); fill!(work.vacuum_tls, 0)
    _primitive_pass!(U, g, τ, model, work, L, eos)

    # ---------------- 2. reconstruct + flux, per direction ----------------
    reconstruct_muscl_2d!(work.ULpx, work.URpx, work.sigx,
                          work.yT, work.phi, work.ux, work.uy, g, :x)
    reconstruct_muscl_2d!(work.ULpy, work.URpy, work.sigy,
                          work.yT, work.phi, work.ux, work.uy, g, :y)

    fill!(work.Fhx, 0.0)
    fill!(work.Fhy, 0.0)
    fill!(work.amax_tls, 0.0)

    _faces_2d!(work.Fhx, work.ULpx, work.URpx, work.ULcx, work.URcx,
               U, g, τ, model, work, :x, Emin)
    _faces_2d!(work.Fhy, work.ULpy, work.URpy, work.ULcy, work.URcy,
               U, g, τ, model, work, :y, Emin)

    # ---------------- 3. sources ----------------
    need_div = (L.hasNu && model.advect_nu) || (L.hasPi && model.advect_Pi) ||
               (L.hasShear && model.advect_pi)
    need_div ? compute_divv_2d!(work, g) : fill!(work.theta, 0.0)

    fill!(work.S, 0.0)
    @inbounds Threads.@threads for ix in (ng+1):(ng+g.Nx)
        for iy in (ng+1):(ng+g.Ny)
            i = lin(g, ix, iy)
            _, _, Pi, _, _, _, pieta = dissipatives_at(U, i, L)

            source_cell_2d!(work.S, i, U[L.iSx,i], U[L.iSy,i], U[L.iE,i],
                            work.P[i], Pi, pieta, τ, L)

            if need_div
                divv = work.theta[i]
                if L.hasNu && model.advect_nu
                    work.S[L.iNux,i] += U[L.iNux,i]*divv
                    work.S[L.iNuy,i] += U[L.iNuy,i]*divv
                end
                if L.hasPi && model.advect_Pi
                    work.S[L.iPi,i] += U[L.iPi,i]*divv
                end
                if L.hasShear && model.advect_pi
                    work.S[L.iPixx,i]  += U[L.iPixx,i]*divv
                    work.S[L.iPixy,i]  += U[L.iPixy,i]*divv
                    work.S[L.iPiyy,i]  += U[L.iPiyy,i]*divv
                    work.S[L.iPieta,i] += U[L.iPieta,i]*divv
                end
            end
        end
    end

    # ---------------- 4. assemble ----------------
    fill!(dU, 0.0)
    invdx = 1/g.dx; invdy = 1/g.dy
    @inbounds Threads.@threads for ix in (ng+1):(ng+g.Nx)
        for iy in (ng+1):(ng+g.Ny)
            i = lin(g, ix, iy)
            ixm = i - Nyt
            for a in 1:nv
                dU[a,i] = -(work.Fhx[a,i] - work.Fhx[a,ixm])*invdx -
                           (work.Fhy[a,i] - work.Fhy[a,i-1])*invdy +
                           work.S[a,i]
            end
        end
    end

    return true
end

# ------------------------------------------------------------------------------
# Face loop for one direction.
#
# Face `i` in direction `dir` separates cell `i` from cell `i + stride`. The
# reconstructed primitives supply (T, μ, u^x, u^y); the DISSIPATIVE dofs are taken
# cell-centered, not reconstructed — the same choice the 1-D solver makes
# (src/rhs.jl uses `phys_from_stored(U[L.iNur, i])` directly at faces).
#
# If either reconstructed face state fails to build a valid conserved state, the
# face falls back to first-order (cell-centered) states, mirroring the 1-D
# `use_pc` path.
# ------------------------------------------------------------------------------
function _faces_2d!(Fh::AbstractMatrix, ULp::AbstractMatrix, URp::AbstractMatrix,
                    ULc::AbstractMatrix, URc::AbstractMatrix,
                    U::AbstractMatrix, g::Grid2D, τ::Float64,
                    model::IdealDiffVisc2DModel, work::Work2D,
                    dir::Symbol, Emin::Float64)

    L = model.layout; eos = model.eos
    st = (dir === :x) ? g.Nytot : 1

    fxlo, fxhi, fylo, fyhi = if dir === :x
        (1, g.Nxtot - 1, 1, g.Nytot)
    else
        (1, g.Nxtot, 1, g.Nytot - 1)
    end

    @inbounds Threads.@threads for ix in fxlo:fxhi
        tid = Threads.threadid()
        tmpFL = work.tmpFL[tid]; tmpFR = work.tmpFR[tid]
        for iy in fylo:fyhi
            i  = lin(g, ix, iy)
            ip = i + st

            nuxL, nuyL, PiL, pixxL, pixyL, piyyL, pietaL = dissipatives_at(U, i,  L)
            nuxR, nuyR, PiR, pixxR, pixyR, piyyR, pietaR = dissipatives_at(U, ip, L)

            TL = exp(ULp[1,i]); μL = hq_mass(eos) + TL*ULp[2,i]
            TR = exp(URp[1,i]); μR = hq_mass(eos) + TR*URp[2,i]

            okL, primL = prim_to_cons_2d!(ULc, i, TL, μL, ULp[3,i], ULp[4,i],
                                          nuxL, nuyL, PiL, pixxL, pixyL, piyyL, pietaL,
                                          τ, eos, L)
            okR, primR = prim_to_cons_2d!(URc, i, TR, μR, URp[3,i], URp[4,i],
                                          nuxR, nuyR, PiR, pixxR, pixyR, piyyR, pietaR,
                                          τ, eos, L)

            good = okL && okR &&
                   ULc[L.iDtau,i] >= 0.0 && URc[L.iDtau,i] >= 0.0 &&
                   ULc[L.iE,i]    >= Emin && URc[L.iE,i]    >= Emin

            if !good
                # first-order fallback from cell-centered primitives
                TLc = exp(work.yT[i]);  μLc = work.mu[i]
                TRc = exp(work.yT[ip]); μRc = work.mu[ip]
                okL, primL = prim_to_cons_2d!(ULc, i, TLc, μLc, work.ux[i], work.uy[i],
                                              nuxL, nuyL, PiL, pixxL, pixyL, piyyL, pietaL,
                                              τ, eos, L)
                okR, primR = prim_to_cons_2d!(URc, i, TRc, μRc, work.ux[ip], work.uy[ip],
                                              nuxR, nuyR, PiR, pixxR, pixyR, piyyR, pietaR,
                                              τ, eos, L)
                okL && okR || continue      # leave the face flux at zero
            end

            sL, sR = hlle_flux_2d!(Fh, i, ULc, URc, primL, primR, eos, τ, dir,
                                   L, model, tmpFL, tmpFR)
            a = max(abs(sL), abs(sR))
            a > work.amax_tls[tid] && (work.amax_tls[tid] = a)
        end
    end
    return nothing
end

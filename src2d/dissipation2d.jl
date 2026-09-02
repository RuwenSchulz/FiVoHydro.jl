# ==============================================================================
# src2d/dissipation2d.jl — Navier–Stokes targets and IS relaxation in 2+1D.
#
# Counterpart of the shear/bulk half of src/dissipation.jl.
#
# ------------------------------------------------------------------------------
# THE SHEAR TARGET — derivation, and why it agrees with the 1-D code exactly
# ------------------------------------------------------------------------------
# In Milne (τ,x,y,η) with g = diag(-1,1,1,τ²), u^η = 0, ∂_η = 0, the only non-zero
# Christoffels are Γ^τ_{ηη} = τ and Γ^η_{τη} = 1/τ. Hence ∇_μ u^ν = ∂_μ u^ν for
# μ,ν ∈ {τ,x,y}, and ∇_η u^η = u^τ/τ, so
#
#     θ = ∂_τ u^τ + ∂_x u^x + ∂_y u^y + u^τ/τ.
#
# With Δ^{μν} = g^{μν} + u^μu^ν and D = u^α∂_α,
#
#     Δ^{iα}∇_α u^j = ∂_i u^j + u^i D u^j        (i,j transverse)
#
# so, writing a^μ = Du^μ for the four-acceleration,
#
#     σ^{ij} = ½(∂_i u^j + ∂_j u^i) + ½(u^i a^j + u^j a^i) - ⅓(δ^{ij} + u^i u^j) θ
#     σ^η_η  = τ²σ^{ηη} = u^τ/τ - θ/3
#
# `σ^η_η` reproduces `src/shear_tensor.jl`'s `ση` term for term. More importantly,
# setting u^y = 0 gives `σ^{yy} = ∂_y u^y - θ/3`, whose radial counterpart is
# `u^r/r - θ/3` — the 1-D code's `σφ`, EXACTLY.
#
# That is not a coincidence and it explains a subtlety worth recording: the 1-D
# solver's NS target LOOKS as though it drops the acceleration piece `u^i Du^j`,
# but it does not. It evolves only π^φ_φ and π^η_η and reconstructs π^r_r by
# tracelessness — and for those two components u^φ = u^η = 0, so the acceleration
# term vanishes identically. The 1-D form is exact for what it evolves.
#
# In 2-D the stored components are LAB-frame π^{xx}, π^{xy}, π^{yy}, and there
# u^i ≠ 0, so the acceleration term is genuinely present and must be carried.
# That is the one real difference between the two solvers' shear targets, and it
# is measured rather than assumed by `test_dissipation2d.jl`.
#
# ------------------------------------------------------------------------------
# THE PROJECTED COMOVING DERIVATIVE
# ------------------------------------------------------------------------------
# The IS equation is  τ_π Δ^{μν}_{αβ} D π^{αβ} + π^{μν} = π_NS^{μν}.  Using
# u_α π^{αβ} = 0 (so u_α Dπ^{αβ} = -π^{αβ}Du_α) and g_{αβ}π^{αβ} = 0 (so the
# trace part of the projector drops out entirely),
#
#     Δ^{ij}_{αβ} Dπ^{αβ} = Dπ^{ij} - u^i c^j - u^j c^i,
#     c^j ≡ π^{jβ}Du_β = -π^{jτ}a^τ + π^{jx}a^x + π^{jy}a^y.
#
# So the equation actually integrated is
#
#     τ_π D π^{ij} + π^{ij} = π_NS^{ij} + τ_π (u^i c^j + u^j c^i).
#
# The correction vanishes when u^i = 0 — which is again why it is absent from the
# 1-D code. Dropping it here would be a physics error of order τ_π·a·π, not a
# constraint violation, so it is carried explicitly.
#
# ------------------------------------------------------------------------------
# THE CHARGE SECTOR (closes D2)
# ------------------------------------------------------------------------------
# ν_NS^μ = -κ ∇^{⟨μ⟩}α = -κ Δ^{μν}∂_ν α. For a transverse index,
#
#     Δ^{iν}∂_ν α = ∂_i α + u^i (u^α ∂_α α) = ∂_i α + u^i Dα.
#
# Setting u^y = 0 gives -κ(∂_r α + u^r(u^τ∂_τα + u^r∂_rα)) = -κ((u^τ)²∂_rα +
# u^r u^τ ∂_τα), using 1 + (u^r)² = (u^τ)². That is EXACTLY the 1-D expression in
# src/dissipation.jl, temporal piece included — the one whose omission the comment
# there records as leaving ν^r "≈2x under-driven vs Fluidum". So D2 is closed by
# derivation, and gated in the u^y = 0 limit by test_charge2d.jl.
#
# Projected relaxation, from u_ν ν^ν = 0 ⇒ u_ν Dν^ν = -ν^ν Du_ν:
#
#     Δ^i_ν Dν^ν = Dν^i - u^i (ν·a),      ν·a = -ν^τ a^τ + ν^x a^x + ν^y a^y
#
# so  τ_n D ν^i + ν^i = ν_NS^i + τ_n u^i (ν·a).
#
# ⚠ The 1-D code writes this term via `Dy = (∂_τu^r + u^r ∂_ru^r)/u^τ`
# (src/dissipation.jl, the `- τn v Dy` entry in `damp`). With y = asinh u^r the
# comoving derivative is Dy = u^τ∂_τ y + u^r∂_r y = ∂_τu^r + (u^r/u^τ)∂_ru^r — the
# ∂_τ piece should NOT carry the 1/u^τ. See TWOD_PROGRAM.md D5: algebra only,
# magnitude UNMEASURED, production untouched.
#
# ------------------------------------------------------------------------------
# TRANSPORT: the dissipative dofs are advected INSIDE this substep by upwinding
# (`relax_advect_*`), NOT as passive scalars through the finite-volume flux. That
# is main.jl's production convention (advect_* = false, relax_advect_* = true,
# main.jl:540-564). See TWOD_PROGRAM.md §6e for what getting it backwards cost.
#
# ------------------------------------------------------------------------------
# SIGN CONVENTION:  π_NS = -2 η σ,  Π_NS = -ζ θ.
#
# This matches src/shear_tensor.jl (`shear_NS_target_contravariant`) and is fixed
# by physics, not taste: for Bjorken, σ^η_η = u^τ/τ - θ/3 = 2/(3τ) > 0 while the
# longitudinal pressure must be REDUCED, π^η_η = -4η/(3τ). Gate G2 checks this
# against the analytic viscous-Bjorken solution.
# ==============================================================================

# ---- transport coefficients (thin wrappers on the shared 1-D models) ----------

@inline function _thermo_2d(T, μ, n, e, P, eos)
    local_thermo(T, μ, n, e, P, eos)
end

@inline function shear_coeffs_2d(T, μ, n, e, P, model::IdealDiffVisc2DModel)
    model.enable_shear || return (0.0, 0.0, 0.0)
    th = _thermo_2d(T, μ, n, e, P, model.eos)
    η  = viscosity(T, th, model.shear)
    τπ = τ_shear(T, th, model.shear)
    δπ = model.deltaShear_factor * τπ
    return η, τπ, δπ
end

@inline function bulk_coeffs_2d(T, μ, n, e, P, model::IdealDiffVisc2DModel)
    model.enable_bulk || return (0.0, 0.0)
    th = _thermo_2d(T, μ, n, e, P, model.eos)
    ζ  = bulk_viscosity(T, th, model.bulk)
    τΠ = τ_bulk(T, th, model.bulk)
    return ζ, τΠ
end

# ---- kinematics --------------------------------------------------------------

"""
    kinematics_2d(work, g, i, Δ, τ) -> (θ, ax, ay, aτ, dxux, dxuy, dyux, dyuy)

Velocity gradients, expansion scalar and four-acceleration at cell `i`.

`∂_τ u^i` comes from `work.ux_prev` / `work.uy_prev`, the same device the 1-D
solver uses for `∂_τ u^r` and `∂_τ α`. The comment in `src/dissipation.jl` records
that dropping the `∂_τ α` piece left `ν^r` ≈2× under-driven against Fluidum, so
the temporal pieces are carried here from the start rather than added later.
"""
@inline function kinematics_2d(work::Work2D, g::Grid2D, i::Int, Δ::Float64, τ::Float64)
    invdx = 1/(2*g.dx); invdy = 1/(2*g.dy)
    ixm = i - g.Nytot; ixp = i + g.Nytot

    @inbounds begin
        dxux = (work.ux[ixp] - work.ux[ixm]) * invdx
        dxuy = (work.uy[ixp] - work.uy[ixm]) * invdx
        dyux = (work.ux[i+1] - work.ux[i-1]) * invdy
        dyuy = (work.uy[i+1] - work.uy[i-1]) * invdy

        ux = work.ux[i]; uy = work.uy[i]
        uτ = sqrt(1 + ux*ux + uy*uy)

        uxp = work.ux_prev[i]; uyp = work.uy_prev[i]
        have_prev = isfinite(uxp) && isfinite(uyp)
        dtux = have_prev ? (ux - uxp)/Δ : 0.0
        dtuy = have_prev ? (uy - uyp)/Δ : 0.0
        dtuτ = (ux*dtux + uy*dtuy) * safe_inv(uτ)      # from u^τ = sqrt(1+u_⊥²)

        θ = dtuτ + dxux + dyuy + uτ*safe_inv(τ)

        ax = uτ*dtux + ux*dxux + uy*dyux
        ay = uτ*dtuy + ux*dxuy + uy*dyuy
        aτ = (ux*ax + uy*ay) * safe_inv(uτ)            # from u·a = 0
    end
    return θ, ax, ay, aτ, dxux, dxuy, dyux, dyuy
end

"""
    ns_shear_target_2d(ux, uy, θ, ax, ay, dxux, dxuy, dyux, dyuy, η)
        -> (pixx_NS, pixy_NS, piyy_NS, pieta_NS)

`π_NS^{ij} = -2η σ^{ij}` from the derivation in this file's header, plus the
independently computed `π_NS^η_η = -2η(u^τ/τ - θ/3)`.

`pieta_NS` is returned for CHECKING ONLY. Because `σ^{μν}` is traceless and
u-orthogonal by construction, projecting `π_NS^{ij}` must reproduce it — an
analytic identity that `test_dissipation2d.jl` verifies numerically. The solver
itself always takes `pieta` from the projection, never from this value.
"""
@inline function ns_shear_target_2d(ux::Float64, uy::Float64, uτ::Float64, τ::Float64,
                                    θ::Float64, ax::Float64, ay::Float64,
                                    dxux::Float64, dxuy::Float64,
                                    dyux::Float64, dyuy::Float64, η::Float64)
    third = θ/3
    σxx = dxux + ux*ax - third*(1 + ux*ux)
    σyy = dyuy + uy*ay - third*(1 + uy*uy)
    σxy = 0.5*(dxuy + dyux) + 0.5*(ux*ay + uy*ax) - third*(ux*uy)
    σeta = uτ*safe_inv(τ) - third

    return (-2η*σxx, -2η*σxy, -2η*σyy, -2η*σeta)
end

"""
    ns_diffusion_target_2d(ux, uy, uτ, dxa, dya, dta, κ) -> (nux_NS, nuy_NS)

`ν_NS^i = -κ ∇^{⟨i⟩}α = -κ(∂_i α + u^i Dα)` with `Dα = u^τ∂_τα + u^k∂_kα`.

Extracted so gate `test_charge2d.jl` can check the 1-D limit directly: with
`u^y = 0` this must equal the production expression
`-κ[(u^τ)²∂_rα + u^r u^τ ∂_τα]` (src/dissipation.jl), since
`1 + (u^x)² = (u^τ)²`. That identity is what closes D2.
"""
@inline function ns_diffusion_target_2d(ux::Float64, uy::Float64, uτ::Float64,
                                        dxa::Float64, dya::Float64, dta::Float64,
                                        κ::Float64)
    Dα = uτ*dta + ux*dxa + uy*dya
    return (-κ*(dxa + ux*Dα), -κ*(dya + uy*Dα))
end

"""
    compute_alpha_grad_fv_2d!(work, g; limiter=mc_limiter)

Limited, face-reconstructed gradient of the fugacity `α`, in both directions.

This is NOT a raw central difference, and the difference matters. `src/dissipation.jl`
builds `work.gradAlpha` with `compute_field_grad_fv!`: MC-limited slopes -> face
values -> face difference, a TVD construction. `α` on the production IC is steep
(the profile runs -1.75 to +1.89 and the loader then tapers it to 0), and a raw
central difference there produces oscillatory gradients, hence oscillatory
`ν_NS`, hence conserved states the recovery cannot invert. Measured on the
production IC: raw central differences gave 40k recovery failures in the charge
sector where every other sector had ~50.
"""
function compute_alpha_grad_fv_2d!(work::Work2D, g::Grid2D; limiter = mc_limiter)
    ng = g.nghost
    a = work.alpha

    for (dir, st, face, grad, invd) in ((:x, g.Nytot, work.alphaFx, work.gradAx, 1/g.dx),
                                        (:y, 1,       work.alphaFy, work.gradAy, 1/g.dy))
        sl = work.slope_tmp
        fill!(sl, 0.0); fill!(face, 0.0); fill!(grad, 0.0)

        lo, hi = (dir === :x) ? (2, g.Nxtot-1) : (1, g.Nxtot)
        ylo, yhi = (dir === :x) ? (1, g.Nytot) : (2, g.Nytot-1)
        @inbounds for ix in lo:hi, iy in ylo:yhi
            i = lin(g, ix, iy)
            sl[i] = limiter(a[i] - a[i-st], a[i+st] - a[i])
        end

        flo, fhi = (dir === :x) ? (1, g.Nxtot-1) : (1, g.Nxtot)
        gylo, gyhi = (dir === :x) ? (1, g.Nytot) : (1, g.Nytot-1)
        @inbounds for ix in flo:fhi, iy in gylo:gyhi
            i = lin(g, ix, iy); ip = i + st
            k    = (dir === :x) ? ix : iy
            kmax = (dir === :x) ? g.Nxtot : g.Nytot
            if k <= ng || k >= kmax - ng
                face[i] = 0.5*(a[i] + a[ip])
            else
                face[i] = 0.5*((a[i] + 0.5*sl[i]) + (a[ip] - 0.5*sl[ip]))
            end
        end

        @inbounds for ix in (ng+1):(ng+g.Nx), iy in (ng+1):(ng+g.Ny)
            i = lin(g, ix, iy)
            grad[i] = (face[i] - face[i-st]) * invd
        end
    end
    return nothing
end

# ---- the relaxation substep --------------------------------------------------

"""
    relax_dissipative_2d!(U, g, τ, Δ, model, work)

Operator-split relaxation of `(Π, π^{ij}, ν^i)` toward their Navier–Stokes
targets, applied once at the end of a full RK step — the same splitting the 1-D
solver uses (`src/timestepper.jl:139`).

Discretisation mirrors `relax_dissipative!`: backward Euler in the comoving
derivative with `A = τ_π u^τ/Δ`, the `δ_ππ θ` damping in the denominator, and the
explicit second-order couplings in the numerator.

Returns the number of interior cells (a plain count: accumulating inside the
threaded loop would be a race, and a bare `ncell += 1` in Julia's soft scope
would silently create a new local — the trap CLAUDE.md warns about for gate flags).
"""
function relax_dissipative_2d!(U::AbstractMatrix, g::Grid2D, τ::Float64, Δ::Float64,
                               model::IdealDiffVisc2DModel, work::Work2D)
    L = model.layout
    (model.enable_shear || model.enable_bulk || model.enable_diff) || return 0
    ng = g.nghost

    # Snapshot before any in-place update: the upwind transport stencil below
    # reads neighbours, which threads would otherwise see half-relaxed. This is
    # the role `work.nur_tmp` plays in the 1-D `relax_dissipative!`.
    copyto!(work.diss_snap, U)
    Q = work.diss_snap
    model.enable_diff && compute_alpha_grad_fv_2d!(work, g)
    invdx = 1/g.dx; invdy = 1/g.dy

    # Upwind advective derivative of stored row `a` at cell `i`, following
    # src/dissipation.jl: `ur >= 0 ? (q[i]-q[i-1])/dr : (q[i+1]-q[i])/dr`.
    @inline function upw(a::Int, i::Int, ux::Float64, uy::Float64)
        @inbounds begin
            dqx = ux >= 0 ? (Q[a,i] - Q[a,i-g.Nytot])*invdx : (Q[a,i+g.Nytot] - Q[a,i])*invdx
            dqy = uy >= 0 ? (Q[a,i] - Q[a,i-1])*invdy       : (Q[a,i+1] - Q[a,i])*invdy
        end
        return ux*dqx + uy*dqy
    end

    @inbounds Threads.@threads for ix in (ng+1):(ng+g.Nx)
        for iy in (ng+1):(ng+g.Ny)
            i = lin(g, ix, iy)
            work.ok[i] || continue

            ux = work.ux[i]; uy = work.uy[i]
            uτ = sqrt(1 + ux*ux + uy*uy)

            θ, ax, ay, aτ, dxux, dxuy, dyux, dyuy = kinematics_2d(work, g, i, Δ, τ)

            T = exp(work.yT[i]); μ = work.mu[i]
            n = work.n[i]; e = work.e[i]; P = work.P[i]

            # ---------------- bulk ----------------
            Pi_new = L.hasPi ? phys_from_stored(U[L.iPi,i]) : 0.0
            if model.enable_bulk && L.hasPi
                ζ, τΠ = bulk_coeffs_2d(T, μ, n, e, P, model)
                ΠNS = -ζ*θ
                A   = safe_div(τΠ*uτ, Δ)
                advΠ = model.relax_advect_Pi ? -τΠ*upw(L.iPi, i, ux, uy) : 0.0
                Pi_new = (A*Pi_new + ΠNS + advΠ)/(A + 1)
                # POSITIVITY of the total pressure. Not a tuning knob: measured on
                # the production IC with bulk+diffusion, |Pi|/P reached 3.50 and
                # min(P+Pi) went NEGATIVE (-2.7e-4), which makes the effective
                # enthalpy and the sound speed meaningless. Keep P + Pi > 0.
                Pi_new = max(Pi_new, -0.99*max(P, 0.0))
                if model.Pi_clip_factor > 0
                    cap = model.Pi_clip_factor * abs(P)
                    Pi_new = clamp(Pi_new, -cap, cap)
                end
                U[L.iPi,i] = stored_from_phys(Pi_new)
            end

            # ---------------- shear ----------------
            if model.enable_shear && L.hasShear
                η, τπ, δπ = shear_coeffs_2d(T, μ, n, e, P, model)

                pixx = phys_from_stored(U[L.iPixx,i])
                pixy = phys_from_stored(U[L.iPixy,i])
                piyy = phys_from_stored(U[L.iPiyy,i])
                pieta = phys_from_stored(U[L.iPieta,i])

                nsxx, nsxy, nsyy, _ = ns_shear_target_2d(ux, uy, uτ, τ, θ, ax, ay,
                                                         dxux, dxuy, dyux, dyuy, η)

                # projected-comoving-derivative correction: c^j = π^{jβ}Du_β
                Π = shear_tensor_contravariant_2d(ux, uy, uτ, τ, pixx, pixy, piyy, pieta)
                cx = -Π.tx*aτ + Π.xx*ax + Π.xy*ay
                cy = -Π.ty*aτ + Π.xy*ax + Π.yy*ay

                A = safe_div(τπ*uτ, Δ)
                den = A + 1 + δπ*θ
                den = max(den, 1e-12)

                pd = model.shear_projected_deriv ? τπ : 0.0
                ra = model.relax_advect_pi
                axx = ra ? -τπ*upw(L.iPixx, i, ux, uy) : 0.0
                axy = ra ? -τπ*upw(L.iPixy, i, ux, uy) : 0.0
                ayy = ra ? -τπ*upw(L.iPiyy, i, ux, uy) : 0.0
                pixx = (A*pixx + nsxx + pd*(2*ux*cx)      + axx) / den
                pixy = (A*pixy + nsxy + pd*(ux*cy + uy*cx) + axy) / den
                piyy = (A*piyy + nsyy + pd*(2*uy*cy)      + ayy) / den

                if model.pi_clip_factor > 0
                    cap = model.pi_clip_factor * abs(P)
                    pixx = clamp(pixx, -cap, cap)
                    pixy = clamp(pixy, -cap, cap)
                    piyy = clamp(piyy, -cap, cap)
                end

                # tracelessness closes the system: pieta is DERIVED, never evolved
                pieta_new, _ = project_shear_traceless_2d(ux, uy, uτ, pixx, pixy, piyy, pieta)

                U[L.iPixx,i]  = stored_from_phys(pixx)
                U[L.iPixy,i]  = stored_from_phys(pixy)
                U[L.iPiyy,i]  = stored_from_phys(piyy)
                U[L.iPieta,i] = stored_from_phys(pieta_new)
            end

            # ---------------- charge diffusion ----------------
            if model.enable_diff && L.hasNu
                wv = vacuum_weight_2d(n, model)
                if wv <= 0.0
                    # true-vacuum backstop: no charge, no current
                    U[L.iNux,i] = 0.0
                    U[L.iNuy,i] = 0.0
                    continue
                end

                κ, τn, δN = diff_coeffs_2d(T, μ, n, model)

                nux = phys_from_stored(U[L.iNux,i])
                nuy = phys_from_stored(U[L.iNuy,i])

                # ∇^{⟨i⟩}α = ∂_i α + u^i Dα, with the LIMITED gradient
                dxa = work.gradAx[i]
                dya = work.gradAy[i]
                αp  = work.alpha_prev[i]
                dta = isfinite(αp) ? (work.alpha[i] - αp)/Δ : 0.0

                nsx, nsy = ns_diffusion_target_2d(ux, uy, uτ, dxa, dya, dta, κ)
                nsx *= wv; nsy *= wv        # ramp the drive out through the tail

                # 2026-09-02 (D9): ramp the RELAXATION TIME by the same weight,
                # not just the drive.
                #
                # The ramp was written to kill ν_NS in the dilute tail, and that
                # was sufficient only while τ_n was 6x too short (D8): killing the
                # DRIVE killed the CURRENT, because ν relaxed to the new target
                # within a step. At the corrected τ_n it does not. ν made earlier
                # in the fluid is then FROZEN into the tail while n collapses
                # around it, |ν|/n diverges, and the charge row `n u^τ + ν^τ = J^τ`
                # goes degenerate -- D6's mechanism, reached from the other side.
                # MEASURED at N=300 on the production IC: primfail 0 -> 12292 with
                # the corrected τ_n, every failure in the tail, and turning
                # diffusion off restores every field exactly.
                #
                # With τ_eff = wv τ_n the tail relaxes to wv ν_NS -> 0 FAST instead
                # of freezing, and the deep-vacuum limit is ν -> 0 in one step.
                #
                # ⚠ NOT inert, and my first claim that it was "bitwise identical
                # wherever the fluid is" was WRONG -- measured, not argued. The
                # ramp band reaches ABOVE freeze-out: on the production IC at
                # N=300 the minimum wv over cells with T > T_fo is 0.22 (0.41 at
                # the old τ_n), so this acts where observables come from.
                #
                # What IS true, and is the licence:
                #   * it does not move the FIXED POINT. Both variants relax toward
                #     the same ramped target wv·ν_NS; only the approach rate
                #     differs. A clip would move the answer; this moves the rate.
                #   * in the regime where the solver was HEALTHY (the pre-D8 τ_n,
                #     primfail 0 either way) the A/B is T 1.5e-6, n 1.4e-4,
                #     ν 2.4e-3 -- it changes no conclusion there.
                #   * at the corrected τ_n it is the difference between a run that
                #     works and one that fails in 12292 cells.
                # `vacuum_ramp_relax = false` reproduces the old behaviour for A/B.
                model.vacuum_ramp_relax && (τn = wv*τn)

                # projected-derivative correction: ν·a with ν^τ from orthogonality
                nut = nu_tau_2d(ux, uy, uτ, nux, nuy)
                nua = -nut*aτ + nux*ax + nuy*ay

                A = safe_div(τn*uτ, Δ)
                den = max(A + 1 + δN*θ, 1e-12)

                pdn = model.diff_projected_deriv ? τn : 0.0
                anx = model.relax_advect_nu ? -τn*upw(L.iNux, i, ux, uy) : 0.0
                any_ = model.relax_advect_nu ? -τn*upw(L.iNuy, i, ux, uy) : 0.0
                nux = (A*nux + nsx + pdn*ux*nua + anx) / den
                nuy = (A*nuy + nsy + pdn*uy*nua + any_) / den

                # causality guard, as in the 1-D `nur_clip_factor`
                if model.nu_clip_factor > 0
                    cap = model.nu_clip_factor * max(n, 0.0) * uτ
                    mag = hypot(nux, nuy)
                    if mag > cap && mag > TINY
                        sc = cap/mag
                        nux *= sc; nuy *= sc
                    end
                end

                U[L.iNux,i] = stored_from_phys(nux)
                U[L.iNuy,i] = stored_from_phys(nuy)
            end
        end
    end

    # store the current velocity for the next step's ∂_τ u^i
    copyto!(work.ux_prev, work.ux)
    copyto!(work.uy_prev, work.uy)
    # α_prev is stored AFTER the substep so ∂_τα is a genuine backward difference
    # over the step, matching src/dissipation.jl's note that refreshing it on entry
    # makes ∂_τα ≡ 0 and under-drives ν by ~2x.
    copyto!(work.alpha_prev, work.alpha)

    return g.Nx*g.Ny
end

# =================================================================================================
# export_background2d.jl — write a FiVo 2+1D run as a background the transport can read.
#
# This is the bridge that was missing. `LangevInMedium` can now propagate on a genuine transverse
# plane (a 3-element (xgrid, ygrid, tgrid), T[x,y,t] and a velocity VECTOR pair), but nothing built
# such a table from a FiVo 2-D solve, so nothing hydro-vs-transport was possible in 2+1D.
#
# ⚠ THE ONE CONVERSION THAT MATTERS: FiVo stores the FOUR-VELOCITY u^x, u^y (u^tau = sqrt(1+u^2)),
# while the transport reads a VELOCITY -- it clamps |v| < 1 and forms gamma = 1/sqrt(1-v^2).
# Handing it u^i would be silently wrong wherever the flow is relativistic: at u = 1.4 (routine at
# the rim by tau ~ 6) the transport would clamp to |v| = 1 and boost with a garbage gamma instead
# of the correct v = 0.81. So this writes
#
#       v^i = u^i / u^tau
#
# and asserts max |v| < 1 on what it wrote. Getting this backwards is the single most likely way to
# produce a plausible-looking but wrong 2-D transport run.
#
# Also written, so a consumer never has to guess: T in GeV, the Milne times in fm/c, the charm
# density n (not needed by the transport, but it is what a sampler would draw production points
# from), and the run's provenance.
#
# Usage:
#   julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/tools/export_background2d.jl \
#         [ic.csv] [out.jld2] [N] [tau_final]
# =================================================================================================

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H = hydro2d
using JLD2, Printf

const DEFAULT_IC = normpath(joinpath(_ROOT, "..", "Projects", "ALICE_IC_Creation",
                                     "PbPb", "data", "ic2d_20-30_ev02.csv"))

"""
    export_background2d(ic; out, N, τ0, τf, nt, DsT, ...) -> path

Run the 2-D solver and write `(xgrid, ygrid, tgrid, T_grid, vx_grid, vy_grid, n_grid)`.

The snapshots are taken on a UNIFORM tau grid because the consumer interpolates trilinearly and a
non-uniform time axis would be silently under-resolved between widely spaced early snapshots --
where the flow is changing fastest.
"""
function export_background2d(ic::AbstractString;
                             out::AbstractString = joinpath(_ROOT, "data", "background2d.jld2"),
                             N::Int = 200, τ0::Float64 = 0.4, τf::Float64 = 8.0,
                             nt::Int = 39, xmax::Float64 = 16.0,
                             DsT::Float64 = 0.1163, ηs::Float64 = 0.10, ζs::Float64 = 0.10)
    g = H.make_grid2d(N, N; xmax = xmax, ymax = xmax)
    m = H.build_model_2d(; eos = H.LatticeHRGEOS(),
        enable_shear = true, eta_over_s = ηs, tauShear_coeff = 0.2, deltaShear_factor = 4/3,
        enable_bulk  = true, zeta_over_s = ζs, tauPi_coeff = 15.0,
        enable_diff  = true, kappa_coeff = DsT, tauN_coeff = 1.0,
        pi_clip_factor = 1.0)
    U = H.allocate_state(g, m); wk = H.make_work(g, m)
    H.initialize_from_grid_csv!(U, g, m, τ0, ic)

    ng = g.nghost
    xs = [g.xC[ix] for ix in (ng+1):(ng+g.Nx)]
    ys = [g.yC[iy] for iy in (ng+1):(ng+g.Ny)]
    ts = collect(range(τ0, τf; length = nt))

    T  = zeros(length(xs), length(ys), nt)
    vx = zeros(length(xs), length(ys), nt)
    vy = zeros(length(xs), length(ys), nt)
    nn = zeros(length(xs), length(ys), nt)

    @printf("FiVo 2-D background: %s\n  N=%d  box=±%.1f fm  tau %.2f→%.2f in %d slices  D_sT=%.4f\n",
            basename(ic), N, xmax, τ0, τf, nt, DsT)
    τ = τ0
    maxv = 0.0; maxu = 0.0
    for (k, ts_k) in enumerate(ts)
        if ts_k > τ + 1e-12
            r = H.run_sim_2d!(U, g, m; τ0 = τ, τfinal = ts_k, CFL = 0.2, CFLτ = 0.02, work = wk)
            r.ok || error("export_background2d: the solve failed at τ = $(r.τ)")
            τ = r.τ
        end
        H.update_primitives_2d!(U, g, τ, m, wk)
        for (i, ix) in enumerate((ng+1):(ng+g.Nx)), (j, iy) in enumerate((ng+1):(ng+g.Ny))
            l = H.lin(g, ix, iy)
            ux = wk.ux[l]; uy = wk.uy[l]; uτ = sqrt(1 + ux*ux + uy*uy)
            # ⚠ four-velocity -> velocity. See the header: the transport reads v, not u.
            T[i,j,k]  = exp(wk.yT[l])
            vx[i,j,k] = ux/uτ
            vy[i,j,k] = uy/uτ
            nn[i,j,k] = wk.n[l]
            maxu = max(maxu, hypot(ux, uy))
            maxv = max(maxv, hypot(vx[i,j,k], vy[i,j,k]))
        end
    end

    # the assertion that makes the conversion checkable rather than assumed
    maxv < 1.0 || error("export_background2d: |v| = $(maxv) >= 1 — the four-velocity conversion is wrong")
    @printf("  max |u| = %.3f  ->  max |v| = %.4f   (a table of u would have been clamped to 1)\n",
            maxu, maxv)
    @printf("  T range %.4f … %.4f GeV\n", minimum(T), maximum(T))

    mkpath(dirname(out))
    jldsave(out;
        label = "fivo_2d_background",
        xgrid = xs, ygrid = ys, tgrid = ts,
        T_grid = T, vx_grid = vx, vy_grid = vy, n_grid = nn,
        # provenance: what produced it and under which conventions
        ic = ic, N = N, xmax = xmax, tau0 = τ0, tauf = τf,
        DsT = DsT, eta_over_s = ηs, zeta_over_s = ζs,
        velocity_convention = "v^i = u^i/u^tau (VELOCITY, not four-velocity)",
        note = "post-D8 tau_n (bare, g_hq cancels) and post-D9 vacuum-ramp relaxation")
    @printf("  wrote %s  (%d × %d × %d)\n", out, length(xs), length(ys), nt)
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    ic  = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_IC
    out = length(ARGS) >= 2 ? ARGS[2] : joinpath(_ROOT, "data", "background2d.jld2")
    N   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 200
    τf  = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 8.0
    export_background2d(ic; out = out, N = N, τf = τf)
end

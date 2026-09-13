# ==============================================================================
# src/api1d.jl — the 1+1D solver's front door, name for name the 2+1D one.
#
#     g = make_grid_1d(400; rmax = 15.0)
#     m = build_model_1d(; enable_shear = true, eta_over_s = 0.1,
#                          enable_diff = true, kappa_coeff = 0.1163,
#                          consistent_fm = true, terms = :homogeneous)
#     show_equations(m)
#     U = allocate_state(g, m)
#     initialize_from_radial!(U, g, m, 0.4, r -> 0.05 + 0.4exp(-r^2/8), r -> -4.0)
#     res = run_sim_1d!(U, g, m; τ0 = 0.4, τfinal = 5.0)
#     f = fields_1d(g, U, m; τ = res.τ, work = res.work)   # every field, physical units
#
# WHY A SECOND ENTRY POINT. `run_sim_ideal_diff_visc` (main.jl) is the PRODUCTION
# driver: ~80 keywords, file output only, and legacy defaults (shear and bulk ON,
# δ_ππ = 0, tauShear_coeff = 1, D_sT = 0.24) that ten callers and the DPM recipes rely
# on. It is kept exactly as it was. This file is the library interface: in memory,
# composable, and with the SAME defaults as `build_model_2d` — the bare ideal scheme,
# every sector opt-in, δ_ππ = 4/3 τ_π, C_s = 0.2, C_ζ = 15 — so a 1-D and a 2-D model
# built with the same keywords integrate the same equations (TWOD_PROGRAM.md D16 was
# the trap of the two solvers defaulting δ_ππ differently).
#
# Both entry points run the same numerics (rhs!, step_ssprk2!/3!, relax_dissipative!).
# ==============================================================================

"""
    make_grid_1d(Nr; rmax = 15.0, nghost = 3) -> Grid1D

Uniform radial grid on `[0, rmax]`, the axis at a face, `nghost` ghost cells on
both sides (reflecting at the axis, outflow at `rmax`). Cell centres `g.rC`, faces
`g.rF`, spacing `g.dr`. `Nr > 32`.
"""
make_grid_1d(Nr::Int; rmax::Float64 = 15.0, nghost::Int = 3) = make_grid(Nr; rmax, nghost)

"""
    sector_on_1d(model) -> (sector::Symbol -> Bool)

The term sectors a 1-D bulk model carries (`src/terms.jl`). The bulk solver has no
charm second moment, so `:consistent_m2` is always false here — that closure lives
in the charm solver, main2IS2.jl.
"""
sector_on_1d(m::IdealDiffViscModel) = s -> s === :enable_diff   ? m.enable_diff :
                                           s === :consistent_fm ? m.consistent_fm :
                                           s === :enable_shear  ? m.enable_shear : false

"""
    build_model_1d(; eos = LatticeHRGEOS(), enable_shear = false, enable_bulk = false,
                     enable_diff = false, consistent_fm = false, terms = :default, kwargs...)

Assemble a 1+1D model (radial Milne, boost invariant). Defaults are the BARE ideal
scheme, exactly as `build_model_2d`; every sector and every regulator is opt-in.

Sectors and coefficients

| keyword | default | meaning |
|---|---|---|
| `eos` | `LatticeHRGEOS()` | `ConformalHQEOS(m_hq = 0, g_hq = 0)` for a charge-free conformal fluid |
| `enable_shear`, `eta_over_s`, `tauShear_coeff` | false, 0.1, 0.2 | η = (η/s)s, τ_π = η/(C_s T s) |
| `deltaShear_factor` | 4/3 | δ_ππ/τ_π (⚠ `run_sim_ideal_diff_visc` defaults to 0) |
| `taupi_pi_factor`, `lambda_pi_Pi_factor` | 0, 0 | τ_ππ/τ_π, λ_πΠ/τ_π (DNMR couplings) |
| `enable_bulk`, `zeta_over_s`, `tauPi_coeff` | false, 0.1, 15 | peaked ζ/s, τ_Π from C_ζ |
| `deltaPi_factor`, `lambda_Pi_pi_factor` | 0, 0 | δ_ΠΠ/τ_Π, λ_Ππ/τ_Π |
| `enable_diff`, `kappa_coeff` | false, 0.0 | charge diffusion with D_sT = `kappa_coeff` [GeV fm]. ⚠ set it |
| `consistent_fm` | false | the full-∇P charm first moment (the O+O IS2 closure, in the bulk solver) |
| `terms` | `:default` | per-term switches — a preset, `without(...)`, or a NamedTuple (src/terms.jl) |
| `tauN_coeff`, `deltaN_factor`, `lambda_NN_factor` | 1, 0, 0 | τ_n multiplier (refused with `consistent_fm`), δ_nn/τ_n, λ_nn/τ_n |
| `charge_mode` | `:mis` | `:density_frame` = no ν field, parabolic flux (no `consistent_fm`) |

Numerics and regulators: every other field of `IdealDiffViscModel` (`nur_clip_factor`,
`pi_clip_factor`, `Pi_clip_factor`, the `*_filter_eps` / `*_smooth_len` filters,
`do_soft_project_nur`, `do_axis_project_nur`, `relax_advect_*`, `advect_*`, the
`*_dt_coeff` step caps), all OFF / bare by default. README.md lists them.
"""
function build_model_1d(; eos = LatticeHRGEOS(),
        charge_mode::Symbol = :mis,
        # charge
        enable_diff::Bool = false, kappa_coeff::Float64 = 0.0,
        diffusion_drive::Symbol = :alpha, tauN_coeff::Float64 = 1.0,
        deltaN_factor::Float64 = 0.0, lambda_NN_factor::Float64 = 0.0,
        consistent_fm::Bool = false, consistent_m2::Bool = false,
        # shear / bulk
        enable_shear::Bool = false, eta_over_s::Float64 = 0.1, tauShear_coeff::Float64 = 0.2,
        deltaShear_factor::Float64 = 4/3, taupi_pi_factor::Float64 = 0.0,
        lambda_pi_Pi_factor::Float64 = 0.0,
        enable_bulk::Bool = false, zeta_over_s::Float64 = 0.1, tauPi_coeff::Float64 = 15.0,
        deltaPi_factor::Float64 = 0.0, lambda_Pi_pi_factor::Float64 = 0.0,
        terms = :default,
        # step caps
        diff_dt_coeff::Float64 = 0.02, shear_dt_coeff::Float64 = 0.3, bulk_dt_coeff::Float64 = 0.3,
        # transport choice for the dissipatives
        advect_nur::Bool = false, relax_advect_nur::Bool = true,
        advect_Pi::Bool = false, relax_advect_Pi::Bool = true,
        advect_pi::Bool = false, relax_advect_pi::Bool = true,
        # regulators — all OFF
        nur_clip_factor::Float64 = -1.0, Pi_clip_factor::Float64 = -1.0, pi_clip_factor::Float64 = -1.0,
        alpha_filter_eps::Float64 = 0.0, nur_filter_eps::Float64 = 0.0, visc_filter_eps::Float64 = 0.0,
        alpha_smooth_len::Float64 = 0.0, nur_smooth_len::Float64 = 0.0, visc_smooth_len::Float64 = 0.0,
        dtau_u_smooth_len::Float64 = 0.0,
        do_soft_project_nur::Bool = false, do_axis_project_nur::Bool = false, axis_project_nfit::Int = 2)

    consistent_m2 && error("build_model_1d: `consistent_m2` — the charm second moment is not " *
        "carried by the 1+1D BULK solver. It lives in the charm solver on a frozen background: " *
        "main2IS2.jl, `run_static_IS2_test(; consistent_m2 = true)`.")
    if consistent_fm
        enable_diff || error("build_model_1d: `consistent_fm = true` needs `enable_diff = true` " *
                             "(it corrects the charge current's drive).")
        charge_mode === :mis || error("build_model_1d: `consistent_fm` needs `charge_mode = :mis` " *
                                      "(the density frame has no ν field to drive).")
        diffusion_drive === :alpha || error("build_model_1d: `consistent_fm` is derived for " *
                                            "`diffusion_drive = :alpha`.")
        tauN_coeff == 1.0 || error("build_model_1d: `tauN_coeff = $tauN_coeff` with `consistent_fm`: " *
            "it rescales τ_n but not the enthalpy h the sources use, so τ_n T/D_s ≠ h and the " *
            "row's relaxation and its sources sit on different clocks (the 2-D solver refuses the same).")
    end
    warn_if_acausal_shear(; enable_shear, eta_over_s, tauShear_coeff, solver = "build_model_1d")
    enable_diff && kappa_coeff == 0.0 && @warn "build_model_1d: enable_diff = true with kappa_coeff = 0 — " *
        "D_sT = 0 means no diffusion at all. Set kappa_coeff (D_s·T in GeV·fm)."

    layout = charge_mode === :density_frame ?
        StateLayout([:Dtau,:Sr,:E,:Pi,:piR,:piEta]; odd_syms=[:Sr]) :
        StateLayout([:Dtau,:Sr,:E,:nur,:Pi,:piR,:piEta]; odd_syms=[:Sr,:nur])
    sh = enable_shear ? QGPViscosity(eta_over_s, tauShear_coeff) : ZeroViscosity()
    bu = (enable_bulk && zeta_over_s != 0.0) ? SimpleBulkViscosity(zeta_over_s, tauPi_coeff) :
                                               ZeroBulkViscosity()
    _check_transport_flags!(; advect_nur, relax_advect_nur, advect_Pi, relax_advect_Pi,
                              advect_pi, relax_advect_pi)

    son = s -> s === :enable_diff ? enable_diff : s === :consistent_fm ? consistent_fm :
               s === :enable_shear ? enable_shear : false
    t = resolve_terms(terms; sector_on = son)
    check_terms(t, son; solver = "build_model_1d")

    return IdealDiffViscModel(eos, layout, IdealPrimRec(),
        enable_diff, diffusion_drive, kappa_coeff, tauN_coeff, deltaN_factor,
        diff_dt_coeff, shear_dt_coeff, bulk_dt_coeff, nur_clip_factor,
        alpha_filter_eps, nur_filter_eps, alpha_smooth_len, nur_smooth_len,
        do_soft_project_nur, do_axis_project_nur, axis_project_nfit, advect_nur, relax_advect_nur,
        enable_shear, enable_bulk, sh, bu,
        deltaPi_factor, deltaShear_factor,
        taupi_pi_factor, lambda_Pi_pi_factor, lambda_pi_Pi_factor, lambda_NN_factor,
        visc_filter_eps, visc_smooth_len, Pi_clip_factor, pi_clip_factor,
        advect_Pi, advect_pi, relax_advect_Pi, relax_advect_pi,
        charge_mode, consistent_fm, t, dtau_u_smooth_len)
end

"""
    allocate_state(g, model) -> Matrix  (nvars × (Nr + 2 nghost))

The conserved state, zero. Rows follow `model.layout` (`Dtau = τJ^τ`, `Sr = T^{τr}`,
`E = T^{ττ}`, then the stored dissipatives).
"""
allocate_state(g::Grid1D, model::IdealDiffViscModel) =
    zeros(length(model.layout.names), g.Nr + 2*g.nghost)

"""
    set_cell!(U, i, T, alpha, ur, τ, model, g; Pi = 0, piR = 0, piEta = 0, nur = 0)

Write cell `i` from primitives: temperature `T` [GeV], fugacity `alpha = μ/T`, radial
four-velocity `ur`, and the dissipatives in PHYSICAL units — bulk `Pi`, the shear
channels `piR` (along the boosted radial direction l = (u^r, u^τ)) and `piEta`
(= τ²π^{ηη}); π^φ_φ = −(piR + piEta) by tracelessness. `nur` = ν^r. Follow with
`finalize_ic!`.
"""
function set_cell!(U::AbstractMatrix, i::Int, T::Float64, alpha::Float64, ur::Float64,
                   τ::Float64, model::IdealDiffViscModel, g::Grid1D;
                   Pi::Float64 = 0.0, piR::Float64 = 0.0, piEta::Float64 = 0.0, nur::Float64 = 0.0)
    Tm = max(T, T_MIN)
    φ  = alpha - hq_mass(model.eos) / Tm
    ok, _ = prim_to_cons_col_ideal_phi_diff_visc!(U, i, log(Tm), φ, asinh(ur),
                                                 nur, Pi, piR, piEta, g.rC[i], τ, model.eos, model.layout)
    ok || error("set_cell!: EOS not finite at T = $T, α = $alpha")
    return nothing
end

"""
    finalize_ic!(U, g, model; τ0)

Boundary conditions, floors and the S–E admissibility bound, as `initialize!` does.
Not optional after `set_cell!`: a vacuum tail starts below the energy floor.
"""
function finalize_ic!(U::AbstractMatrix, g::Grid1D, model::IdealDiffViscModel; τ0::Float64)
    apply_bc!(U, g, τ0, model)
    enforce_floors!(U, g, τ0, model; Emin = E_FLOOR, diag = nothing)
    enforce_Sr_energy_constraint!(U, g, model; χ = χ_SrE, mask = nothing, diag = nothing)
    apply_bc!(U, g, τ0, model)
    return nothing
end

"""
    initialize_uniform!(U, g, model, τ0; T0, alpha0 = 0.0)

A radially uniform state at rest — Bjorken flow.
"""
function initialize_uniform!(U::AbstractMatrix, g::Grid1D, model::IdealDiffViscModel, τ0::Float64;
                             T0::Float64, alpha0::Float64 = 0.0)
    fill!(U, 0.0)
    for i in (g.nghost+1):(size(U,2)-g.nghost)
        set_cell!(U, i, T0, alpha0, 0.0, τ0, model, g)
    end
    finalize_ic!(U, g, model; τ0)
end

"""
    initialize_from_radial!(U, g, model, τ0, Tof, alphaof; urof = r -> 0.0)

From radial profiles `T(r)`, `α(r)` and optionally `u^r(r)` (functions of r in fm).
The dissipatives start at zero.
"""
function initialize_from_radial!(U::AbstractMatrix, g::Grid1D, model::IdealDiffViscModel,
                                 τ0::Float64, Tof, alphaof; urof = r -> 0.0)
    fill!(U, 0.0)
    for i in (g.nghost+1):(size(U,2)-g.nghost)
        r = g.rC[i]
        set_cell!(U, i, Float64(Tof(r)), Float64(alphaof(r)), Float64(urof(r)), τ0, model, g)
    end
    finalize_ic!(U, g, model; τ0)
end

"""
    run_sim_1d!(U, g, model; τ0, τfinal, CFL = 0.2, CFLτ = 0.05, integrator = :ssprk2,
                on_dump = nothing, dump_dt = Inf, work = nothing,
                reset_history = (work === nothing)) -> NamedTuple

Evolve `U` in place from `τ0` to `τfinal`. `on_dump(τ, U, work)` is called at `τ0`
and every `dump_dt`. Returns

    (ok, τ, nsteps, work, Q0, Q1, dQ, maxu, primfail, floorE, mood)

`ok` means every step completed — not that the result is physical: read `maxu`, `dQ`
(relative drift of the conserved charge ∫τJ^τ dA) and the correction counters as well.
To continue a run pass `work = res.work, reset_history = false`; the ∂_τ history of
u^r, α and T lives there. `integrator = :ssprk3` is third order for smooth ideal
flow; either is first order once a dissipative sector is on (the operator split).
"""
function run_sim_1d!(U::AbstractMatrix, g::Grid1D, model::IdealDiffViscModel;
                     τ0::Float64, τfinal::Float64, CFL::Float64 = 0.2, CFLτ::Float64 = 0.05,
                     integrator::Symbol = :ssprk2, on_dump = nothing, dump_dt::Float64 = Inf,
                     work = nothing, reset_history::Bool = (work === nothing),
                     Emin::Float64 = E_FLOOR, χ::Float64 = χ_SrE, max_steps::Int = typemax(Int))
    integrator in (:ssprk2, :ssprk3) ||
        throw(ArgumentError("integrator = $integrator: use :ssprk2 or :ssprk3"))
    check_terms(model.terms, sector_on_1d(model); solver = "run_sim_1d!")
    wk = work === nothing ? make_work(U) : work
    if reset_history
        fill!(wk.y_prev, NaN); fill!(wk.alpha_prev, NaN); fill!(wk.T_prev, NaN)
    end
    diag = DiagCounters()
    τ = τ0
    prime_work_from_U!(wk, U, g, τ, model)
    Q0 = charge_integral_Dtau(U, g, model)
    on_dump === nothing || on_dump(τ, U, wk)
    next_dump = τ0 + dump_dt
    nsteps = 0; ok = true; maxu = 0.0
    while τ < τfinal - 1e-12 && nsteps < max_steps
        Δτ = compute_dt_from_work(wk, g, τ, model; CFL, CFLτ)
        τ + Δτ > τfinal && (Δτ = τfinal - τ)
        Δ = try
            integrator === :ssprk2 ?
                step_ssprk2!(U, g, τ, Δτ, model, wk; Emin, χ, diag) :
                step_ssprk3!(U, g, τ, Δτ, model, wk; Emin, χ, diag)
        catch err
            err isa ErrorException || rethrow()
            @warn "run_sim_1d!: step failed" τ exception = err
            ok = false
            break
        end
        τ += Δ; nsteps += 1
        vmax, _ = _velocity_stats(wk, g)
        maxu = max(maxu, vmax / sqrt(max(1 - vmax^2, 1e-300)))
        if on_dump !== nothing && τ >= next_dump - 1e-12
            on_dump(τ, U, wk); next_dump += dump_dt
        end
    end
    Q1 = charge_integral_Dtau(U, g, model)
    dQ = abs(Q0) > 0 ? (Q1 - Q0) / Q0 : Q1 - Q0
    return (; ok, τ, nsteps, work = wk, Q0, Q1, dQ, maxu,
              primfail = diag.prim_fail_cells, floorE = diag.floor_E_cells,
              mood = diag.mood_stage1_bad + diag.mood_stage2_bad)
end

"""
    fields_1d(g, U, model; τ, work = nothing) -> NamedTuple of vectors over the interior cells

Every field in physical units: `r`, `T`, `mu`, `alpha`, `n`, `e`, `P`, `ur`, `utau`,
`v`, and the dissipatives the layout carries — `nur` (ν^r), `Pi` (Π), `piR`, `piPhi`,
`piEta` (the shear in the comoving orthonormal frame: along l = (u^r, u^τ), along φ̂,
and τ²π^{ηη}; they sum to zero). `ok` flags cells whose primitive recovery succeeded.

Pass the run's `work` (`res.work`) to warm-start the recovery from the solver's own
primitives — then the fields are exactly what the solver sees. Without it the
recovery starts cold, which can fail for extreme states (1-D recovery with
`ConformalHQEOS` at α ≲ −20; README.md, known limitations).
"""
function fields_1d(g::Grid1D, U::AbstractMatrix, model::IdealDiffViscModel; τ::Float64,
                   work = nothing)
    L = model.layout
    idx = (g.nghost+1):(size(U,2)-g.nghost)
    N = length(idx)
    r = g.rC[idx]
    T = zeros(N); mu = zeros(N); n = zeros(N); e = zeros(N); P = zeros(N); ur = zeros(N)
    ok = falses(N)
    wpr = model.primrec.work[1]
    phys_row(row) = [phys_from_stored(U[row, i]) for i in idx]
    nur   = L.hasNur   ? phys_row(L.iNur)   : zeros(N)
    Pi    = L.hasPi    ? phys_row(L.iPi)    : zeros(N)
    piR   = L.hasPiR   ? phys_row(L.iPiR)   : zeros(N)
    piEta = L.hasPiEta ? phys_row(L.iPiEta) : zeros(N)
    for (k, i) in enumerate(idx)
        kw = work === nothing ? NamedTuple() :
             (yT0 = work.yT[i], φ0 = work.phi[i], y0 = work.y[i])
        Ti, μi, uri, ni, ei, Pii, oki = cons_to_prim_ideal_phi_diff_visc!(
            wpr, U[L.iDtau, i]/τ, U[L.iSr, i], U[L.iE, i], nur[k], Pi[k], piR[k], piEta[k],
            g.rC[i], τ, model.eos; maxit = 100, kw...)
        T[k] = Ti; mu[k] = μi; n[k] = ni; e[k] = ei; P[k] = Pii; ur[k] = uri; ok[k] = oki
    end
    utau = sqrt.(1 .+ ur.^2)
    return (; r, T, mu, alpha = mu ./ max.(T, T_MIN), n, e, P, ur, utau, v = ur ./ utau,
              nur, Pi, piR, piPhi = -(piR .+ piEta), piEta, ok, τ)
end

"""
    show_equations([io,] model::IdealDiffViscModel)

Print the equations a 1+1D bulk model integrates — every sector and every
switchable term, `[x]` when carried, `[ ]` when not, `≡0` when it vanishes in
radial symmetry, with the knob that controls it. Conventions and derivations:
EQUATIONS1D.md.
"""
show_equations(m::IdealDiffViscModel) = show_equations(stdout, m)
function show_equations(io::IO, m::IdealDiffViscModel)
    t = m.terms
    mk(b) = b ? "[x]" : "[ ]"
    son = sector_on_1d(m)
    println(io, "FiVo 1+1D bulk — the equations this model integrates")
    println(io, "  radial Milne (τ,r,φ,η), g = diag(−1,1,r²,τ²), boost invariant; D = u^τ∂_τ + u^r∂_r")
    println(io, "  medium       ∂_τ(τ r T^{τν}) + ∂_r(τ r T^{rν}) = geometric sources   (always)")
    println(io, "               T^{μν} = (e + P + Π) u^μu^ν + (P + Π) g^{μν} + π^{μν},   EoS: ", nameof(typeof(m.eos)))
    println(io, "  charge       ∂_τ(τ r J^τ) + ∂_r(τ r J^r) = 0,   J^μ = n u^μ + ν^μ")
    println(io)
    println(io, "  shear  ", mk(m.enable_shear), "  τ_π Dπ_i + π_i = −2η σ_i − …   (channels i = φ, η; π_l = −π_φ − π_η)")
    if m.enable_shear
        @printf(io, "               %s δ_ππ θ π_i,              δ_ππ = %.4g τ_π     deltaShear_factor\n",
                mk(m.deltaShear_factor != 0), m.deltaShear_factor)
        @printf(io, "               %s τ_ππ (π_iσ_i − π:σ/3),  τ_ππ = %.4g τ_π     taupi_pi_factor\n",
                mk(m.taupi_pi_factor != 0), m.taupi_pi_factor)
        @printf(io, "               %s λ_πΠ Π σ_i,             λ_πΠ = %.4g τ_π     lambda_pi_Pi_factor\n",
                mk(m.lambda_pi_Pi_factor != 0), m.lambda_pi_Pi_factor)
        println(io, "               ", mk(m.relax_advect_pi), " τ_π u^r ∂_r π_i  (advection)  relax_advect_pi")
        println(io, "               ≡0  2τ_π π^{λ⟨i}ω_λ^{j⟩}  (no vorticity in radial flow)  terms.shear_vorticity")
        println(io, "               ≡0  the Δ-projector on Dπ (π_φ, π_η are orthogonal to the (τ,r) plane)")
        println(io, "               η = (η/s)·s, τ_π = η/(C_s T s):  η/s = ", _shear_label_1d(m.shear))
    end
    println(io, "  bulk   ", mk(m.enable_bulk), "  τ_Π DΠ + Π = −ζ θ − …")
    if m.enable_bulk
        @printf(io, "               %s δ_ΠΠ θ Π,               δ_ΠΠ = %.4g τ_Π     deltaPi_factor\n",
                mk(m.deltaPi_factor != 0), m.deltaPi_factor)
        @printf(io, "               %s λ_Ππ π:σ,               λ_Ππ = %.4g τ_Π     lambda_Pi_pi_factor\n",
                mk(m.lambda_Pi_pi_factor != 0), m.lambda_Pi_pi_factor)
        println(io, "               ", mk(m.relax_advect_Pi), " τ_Π u^r ∂_r Π  (advection)  relax_advect_Pi")
    end
    if m.charge_mode === :density_frame
        println(io, "  charge [x]  density frame: no ν field, J^r = −κ(u^τ)²∂_rα as a parabolic flux  charge_mode")
    else
        println(io, "  charge ", mk(m.enable_diff), "  τ_n Δ^r_ν Dν^ν + ν^r = drive^r − δ_nn θ ν − λ_nn σ_ll ν")
        if m.enable_diff
            println(io, "               [x] τ_n u^r (ν·a)  (projector, always carried)")
            println(io, "               ", mk(m.relax_advect_nur), " τ_n u^r ∂_r ν^r  (advection)  relax_advect_nur")
            println(io, "      drive^r =")
            for (name, sector, _, what) in TERM_REGISTER
                (sector === :enable_diff || sector === :consistent_fm) || continue
                println(io, "               ", mk(getfield(t, name) && son(sector)), " ", what, "   terms.", name,
                        sector === :consistent_fm ? "  (consistent_fm)" : "")
            end
            m.consistent_fm || println(io, "               (consistent_fm = false: only the fugacity drive — the homogeneous-medium reduction)")
            @printf(io, "               D_sT = %.4g GeV fm,  τ_n = %.3g × D_s z K₃/K₂/T\n", m.kappa_coeff, m.tauN_coeff)
        end
    end
    println(io, "  charm second moment: not in the bulk solver — main2IS2.jl (`show_equations_IS2`)")
    regs = String[]
    m.nur_clip_factor > 0 && push!(regs, "nur_clip_factor=$(m.nur_clip_factor)")
    m.pi_clip_factor >= 0 && push!(regs, "pi_clip_factor=$(m.pi_clip_factor)")
    m.Pi_clip_factor >= 0 && push!(regs, "Pi_clip_factor=$(m.Pi_clip_factor)")
    (m.alpha_filter_eps + m.nur_filter_eps + m.visc_filter_eps + m.alpha_smooth_len +
     m.nur_smooth_len + m.visc_smooth_len) > 0 && push!(regs, "filters")
    m.dtau_u_smooth_len > 0 && push!(regs, "dtau_u<$(m.dtau_u_smooth_len)fm")
    m.do_soft_project_nur && push!(regs, "do_soft_project_nur")
    m.do_axis_project_nur && push!(regs, "do_axis_project_nur")
    println(io, "  regulators   ", isempty(regs) ? "none (the bare scheme)" : join(regs, ", "))
    return nothing
end
_shear_label_1d(sh) = hasproperty(sh, :ηs) ? string(sh.ηs, ", C_s = ", sh.Cs) : "0"

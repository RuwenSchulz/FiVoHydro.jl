# ==============================================================================
# src/fields_io.jl — save a solver's output WITH what produced it, and load it back.
# Shared by the three drivers (main.jl, main2D.jl, main2IS2.jl).
#
#     f = fields_1d(g, U, m; τ = res.τ, work = res.work)        # or fields_2d, or an IS2 Dict
#     save_fields("run.jld2", f; model = m, meta = (event = 3, note = "vorticity on"))
#     d = load_fields("run.jld2");  d.fields.T, d.terms, d.model
#
# A number read back later is only as good as the record of which equations made it.
# The file therefore carries, beside the fields, the resolved term switches, the
# `show_equations` printout of the model (every sector, every term [x]/[ ], every
# knob), every scalar field of the model struct, and the git revision of the solver
# (with a "-dirty" suffix if its tree had uncommitted changes).
#
# SCHEMA (JLD2, plain types only — readable without FiVo loaded):
#     "fields"        the output as the solver returned it: a NamedTuple of arrays
#                     (fields_1d / fields_2d) or a Dict{String,Any} (run_static_IS2_test)
#     "solver"        "FiVo 1+1D bulk" | "FiVo 2+1D" | "FiVo 1+1D charm IS2"
#     "terms"         NamedTuple of Bools — the `Terms` that were integrated (or nothing)
#     "model"         String — the `show_equations` printout (or "")
#     "model_fields"  Dict{String,String} — every scalar field of the model, repr'd
#     "meta"          Dict{String,Any} — whatever the caller passed as `meta`
#     "created"       ISO-8601 time stamp;  "fivo_revision"  git short hash (+ "-dirty")
# ==============================================================================

using JLD2
using Dates

"""`Terms` → a NamedTuple of Bools (plain data, loadable without FiVo)."""
terms_namedtuple(t::Terms) = NamedTuple{fieldnames(Terms)}(Tuple(getfield(t, n) for n in fieldnames(Terms)))

_plain(x) = x
_plain(t::Terms) = terms_namedtuple(t)
_plain(d::AbstractDict) = Dict{String,Any}(string(k) => _plain(v) for (k, v) in d)

function _fivo_revision()
    dir = normpath(joinpath(@__DIR__, ".."))
    try
        rev = readchomp(`git -C $dir rev-parse --short HEAD`)
        dirty = !isempty(readchomp(`git -C $dir status --porcelain --untracked-files=no`))
        return dirty ? rev * "-dirty" : rev
    catch
        return "unknown"
    end
end

function _model_record(model)
    model === nothing && return ("", Dict{String,String}(), nothing)
    txt = try
        sprint(show_equations, model)
    catch
        ""
    end
    flds = Dict{String,String}()
    for n in fieldnames(typeof(model))
        v = getfield(model, n)
        (v isa Number || v isa Symbol || v isa Bool || v isa AbstractString) && (flds[string(n)] = repr(v))
    end
    flds["eos"] = string(nameof(typeof(model.eos)))
    for n in fieldnames(typeof(model.eos))
        flds["eos." * string(n)] = repr(getfield(model.eos, n))
    end
    terms = hasproperty(model, :terms) ? terms_namedtuple(model.terms) : nothing
    return (txt, flds, terms)
end

_solver_name(model) = model === nothing ? "FiVo" :
    occursin("2D", string(nameof(typeof(model)))) ? "FiVo 2+1D" : "FiVo 1+1D bulk"

"""
    save_fields(path, fields; model = nothing, meta = (;), solver = …) -> path

Write `fields` (what `fields_1d`, `fields_2d` or `run_static_IS2_test` returned) to the
JLD2 file `path`, together with the model's term switches, its `show_equations`
printout, every scalar knob, the git revision and a time stamp (schema: this file's
header). For an IS2 result pass no `model`: the result Dict already carries
`"terms"`, `"consistent_fm"`, `"consistent_m2"`.
"""
function save_fields(path::AbstractString, fields; model = nothing, meta = (;),
                     solver::AbstractString = fields isa AbstractDict && model === nothing ?
                                              "FiVo 1+1D charm IS2" : _solver_name(model))
    txt, flds, terms = _model_record(model)
    if terms === nothing && fields isa AbstractDict && haskey(fields, "terms")
        terms = _plain(fields["terms"])
    end
    mkpath(dirname(abspath(path)))
    jldsave(path; fields = _plain(fields), solver = String(solver), terms = terms,
            model = txt, model_fields = flds,
            meta = Dict{String,Any}(string(k) => v for (k, v) in pairs(meta)),
            created = string(Dates.now()), fivo_revision = _fivo_revision())
    return path
end

"""
    load_fields(path) -> NamedTuple (fields, solver, terms, model, model_fields, meta, created, fivo_revision)

Read a file written by `save_fields`. `print(d.model)` shows the equations that made it.
"""
function load_fields(path::AbstractString)
    jldopen(path, "r") do f
        g(k) = haskey(f, k) ? f[k] : nothing
        return (fields = g("fields"), solver = g("solver"), terms = g("terms"), model = g("model"),
                model_fields = g("model_fields"), meta = g("meta"), created = g("created"),
                fivo_revision = g("fivo_revision"))
    end
end

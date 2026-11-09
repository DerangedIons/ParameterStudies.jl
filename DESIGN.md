# ParameterStudies.jl — Design Specification

**Status:** draft for review · **Date:** 2026-11-09

This document specifies how ParameterStudies executes a batch of simulations, at the level of abstractions and type hierarchy. It stops above implementation detail: keyword surfaces, compat bounds, and porting order are settled after this shape is. It deliberately does not redesign the study lifecycle the README describes (study designs, canonical identities, resumable storage, provenance), which remains future work; the two meet at one point, which is that a study design will produce the cases this document's `batch` consumes. The style draws on three studied inspirations: **SciML's ensemble interface** (one prototype problem, a per-trajectory modification, and the ensemble algorithm as a typed object), **CardiacAbstractions.jl** (a light base package owns the verbs and the heavy packages add methods), and **LockstepODE.jl** (N systems sharing a timestep as one array, which is the execution shape this package must be able to request without implementing).

**Revision 2026-11-09: the cable solver is not ported here.** The first draft of this document moved ArmyHeart's batched 1D cable solver into this package, with its own `CableGrid`, `CableModel`, Rush-Larsen integrator, and diffusion kernel, and declared the package cardiac-aware. That is reversed. A cable is a monodomain simulation, and monodomain simulations on structured grids are FlashBang.jl's job; a second implementation here would be a second set of numerics to validate and keep in agreement. The caller now builds **one runnable simulation** with the package that owns the physics, and this package executes a batch of it. ParameterStudies is domain-neutral again, as its README and ArmyHeart's `SPINOUT_PLAN.md` always said. The consequence that matters is stated in §6: the single-kernel-launch GPU sweep now depends on FlashBang growing a batched form, where the first draft would have had it on day one.

## 1. What this adds

A parameter sweep asks the same question thousands of times with a few numbers changed: run a simulation, record what it did. The caller already knows how to run it once. What they lack is a way to say "now run it for all of these" without rewriting the simulation as a batch.

The central claim of this document is that **there are two distinct execution shapes, and this package owns the choice between them but the physics of neither**. *Mapping* runs independent cases one at a time: it isolates failures and asks nothing of the simulation beyond being runnable. *Lockstep* advances many cases as one set of shared arrays with one kernel launch per step, which is where a GPU earns its speedup: run independently, every case pays for its own launches and its own array traffic, though the cases differ only by a few numbers. Lockstep is fast because the cases are forced to share everything except their parameters, and for the same reason they share fate.

Building the shared arrays requires knowing the simulation's state layout, and the layout belongs to the simulation package. So lockstep is split along that line: this package decides *which* cases run together and *where*, and asks the simulation package to produce the stacked simulation. The heat-transfer analogy is a batch furnace: the scheduler decides which parts go in which load, but only the part's designer knows how it can be racked.

## 2. Design principles

1. **This package runs simulations; it never contains one.** No grids, models, integrators, or kernels live here, so there is exactly one implementation of each physics to validate.
2. **One runnable simulation is the unit of input.** The caller supplies a simulation that already works alone, which means a batch can always be debugged by running one of its cases by hand.
3. **A case is opaque.** This package carries cases to the caller's function or to the simulation package and never interprets a parameter name, which is what keeps it domain-neutral.
4. **Sweeping asks nothing of the simulation package.** Only lockstep requires methods, so a package pays for the interface only when it has a batched form to offer.
5. **The simulation package owns its layout.** Stacking N cases into shared arrays is layout code, so it is a method the simulation package provides, never something this package derives.
6. **Execution strategy is a typed value.** Serial, threaded, worker-pool, and lockstep execution are objects passed positionally, replacing a keyword whose meaning changed by method.
7. **A batch changes throughput and nothing else.** Any executor must return, for each case, what running that case alone returns, within a tolerance the simulation package declares.

## 3. The grammar

```
one runnable sim  +  cases  →  batch(apply, sim, cases, executor)  →  one result per case, in case order
(built by the physics       (apply: how one case changes the sim;
 package, as if run alone)   executor: which cases run together, and where)
```

The three canonical programs, which become the docs quickstarts and acceptance examples:

```julia
# The single runnable sim: a FlashBang cable, built exactly as if it were going to run alone.
model = MonodomainModel(; κ, ion = cell, stim)
f     = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)
sim   = Simulation(OperatorSplittingProblem(f, u0, tspan), alg; dt, saveat)
simulate(sim)                                        # works alone; nothing below changes it

# (a) a sweep on a workstation: every combination of two channel scalings, one case per task
cases   = [(ICaL_Multiplier = a, IKr_Multiplier = b) for a in As for b in Bs]
results = batch(sim, cases, Threaded()) do sim, case
    with_channel_scalings(sim, case)                 # the caller's own function: one case in, one sim out
end

# (b) the same sweep as shared arrays on every GPU: FlashBang reads the cases itself
results = batch(sim, cases, Lockstep(CUDABackend(); batch_size))

# (c) work that is not a simulation at all, such as post-processing saved cases
batch(classify_and_save, paths, Workers())
```

## 4. The type system

```
Simulation{P, A, O}                # one runnable sim: a problem, its algorithm, its solve options

simulate(sim)                      # run one sim; defaults to solve, so nobody implements it

# the lockstep interface: the only methods a simulation package ever writes, and only to opt in
stacked(problem, cases)            #   one problem advancing all cases as shared arrays
unstacked(result)                  #   split a stacked result into one result per case

AbstractMapExecutor                # runs cases independently; needs no methods from anyone
├── Serial                         #   in order, on the calling task
├── Threaded                       #   OhMyThreads tasks
└── Workers                        #   Distributed.pmap on an existing pool; isolates per-case failures

Lockstep{B}                        # runs cases as stacked sub-batches on backend B; needs the lockstep interface
```

The mockups below are indicative, not final: they show what each type owns and how it is parameterized. Names may move in the implementation pass, constructors and validation are omitted, and `…` marks deferred detail.

**`Simulation`** is what "runnable" means: a problem alone cannot be run, because the algorithm and the timestep are not part of it. Bundling the three makes the unit of input a value that `simulate` can execute with no further arguments, and it gives the lockstep interface one place to reach the problem. It follows CommonSolve's shape, so any `solve`-able problem is a simulation without its package knowing this one exists; a package that does not speak `solve` implements `simulate` on its own type instead. The trade-off is that this type knows nothing: it cannot validate a problem, a case, or an option, and every such error surfaces from the simulation package.

```julia
struct Simulation{P, A, O}
    problem :: P        # opaque here; built and understood by the simulation package
    alg     :: A
    options :: O        # dt, saveat, callbacks: whatever the single run would pass
end
```

**Applying a case is the caller's function, not an interface.** Mapping needs two things: a way to turn the prototype into one case, and a way to run it. Running is `simulate`, which defaults to `solve`. Turning the prototype into a case is a plain function `(sim, case) -> sim` passed to `batch`, because the caller built the simulation and already knows how to rebuild it with different numbers. So a simulation package writes nothing to get serial, threaded, and distributed sweeps. The trade-off is that the function is opaque: this package cannot tell what it changed, which is fine for mapping and is exactly why lockstep cannot use it.

**The lockstep interface** is the one place a simulation package must write methods, because it is the one thing nobody outside the package can do. Laying N cases out as shared arrays means knowing the state layout, where parameters live, and how to cut the result apart again. `stacked` receives the raw cases as data, so the package can turn them into a parameter array along a case axis instead of N separate models; for FlashBang the natural reading is that a case's parameters are overrides varying along that axis, the way its existing `overrides` vary along space. Under `Lockstep` the caller's apply function is therefore not used, and the two readings of a case are held together by the batch-equals-solo rule in §6.

```julia
function simulate end       # simulate(sim) -> result;   default: solve(sim.problem, sim.alg; sim.options...)

function stacked end        # stacked(problem, cases) -> problem whose state carries a case axis
function unstacked end      # unstacked(result) -> one result per case, in the order given to stacked
```

**`AbstractMapExecutor`** owns "run these independently" and makes no assumption about what is being run. `batch(apply, sim, cases, executor)` is sugar over the plain form `batch(f, items, executor)` of program (c), with `f = case -> simulate(apply(sim, case))`. It replaces ArmyHeart's `backend=:threads` and friends. `Workers` uses the pool the caller already set up: ArmyHeart's in-function `addprocs` created workers with no packages loaded, and `on_error` then reduced every resulting failure to a warning, so a run could complete having done nothing.

```julia
abstract type AbstractMapExecutor end

struct Serial <: AbstractMapExecutor end

struct Threaded <: AbstractMapExecutor
    ntasks :: Int
end

struct Workers <: AbstractMapExecutor end    # caller owns addprocs and @everywhere using
```

**`Lockstep`** owns the scheduling half of a shared-array run: cases are divided evenly across devices, then into sub-batches of `batch_size` on each device, because a device has finite memory no matter how many cases a sweep needs. Each sub-batch is stacked on the host, moved to its device with `Adapt`, run with the same `simulate` a single case uses, and unstacked. Stacking on the host and adapting afterwards replaces the first draft's `create_model` factory: a host-built simulation is not pinned to any device's memory, so one prototype serves every GPU. `Lockstep` is not an `AbstractMapExecutor`, on purpose: a map executor accepts any function, while lockstep accepts only a simulation whose package implements `stacked`, so one hierarchy would make `batch(f, items, Lockstep(…))` type-check and then fail at runtime.

```julia
struct Lockstep{B}
    backend     :: B        # the device family; a KernelAbstractions backend value
    batch_size  :: Int      # cases per stacked sub-batch, limited by device memory
    max_devices :: Int      # defaults from the backend's device count
end
```

## 5. Alternatives considered

**Port ArmyHeart's cable solver here (the first draft).** It delivers the GPU sweep immediately: ArmyHeart's batched Rush-Larsen integrator, diffusion kernel, and device loop already exist, run ToRORdTrauma today, and are simple to develop as one unit. FlashBang, by contrast, has no batched layout and lists Rush-Larsen as out of scope, so the fast path now waits on work in another package. It loses because the cost compounds: a second monodomain discretization that must be validated against the first, cardiac types in a package chartered as domain-neutral, and a `CableGrid` that every future FlashBang feature (anisotropy, heterogeneity, higher dimensions) would have to be re-declined for. ArmyHeart's history is the warning: its batched path silently dropped half its diffusion increments until `39726ee`, and a separate batched solver is exactly where such a bug can hide, because nothing forces it to agree with the unbatched one.

**Stack generically inside this package.** DiffEqGPU's `EnsembleGPUArray` shows it is possible to batch a problem without its package's help, by treating the right-hand side as a kernel body over a trajectory axis. This would make lockstep work for any ODE with no contract at all. It loses for the simulations this package exists to serve: a FlashBang problem is operator-split, keeps its parameters in functor fields rather than in `p`, and has a state layout with exactly one owner by its own design rules. Stacking it from outside means reaching into all three.

**Use `EnsembleProblem` directly and add nothing.** It is the established interface, already has `prob_func` for per-trajectory variation and threaded and distributed algorithms, and SciML users know it. The map path here is deliberately the same shape and may be implemented on top of it. It is not sufficient alone because it has no way to ask a simulation package for a stacked problem, and because the study lifecycle this package is chartered for must also run simulations that are not SciML problems.

**Make case application a method too (`specialize(problem, case)`, the previous revision).** One interface for both shapes is tidy, and it guarantees mapping and lockstep read a case the same way. It loses because it makes every simulation package write and maintain a method before anyone can run a threaded sweep, for something the caller can express in a line. `stacked` already receives the cases as data, so lockstep never needed it.

**Drop the prototype entirely and take a factory `case -> sim`.** This is the least possible input. It works for mapping and forecloses lockstep, because given N opaque simulations there is no way to know they share a grid, a timestep, and a model, which is the precondition for one array. Keeping the prototype as a separate argument is what lets program (b) differ from (a) by the executor alone.

**Keep ArmyHeart's keyword-driven `batch`.** It is familiar, and porting it verbatim would be less work. It overloads `backend` to mean a `Symbol` in one method and a device in the other, and lets an old GPU script run on CPU without complaint. See open question 1.

## 6. Cross-cutting rules

**What this asks of FlashBang.** Mapping asks nothing: it works today with any simulation FlashBang can already run. Lockstep needs three things FlashBang does not have today: a batched layout (its own DESIGN.md already names this as a second layout type over the `StateBlockedLayout` seam), `stacked` and `unstacked` methods over it, and, for ArmyHeart's actual sweeps, a Rush-Larsen inner solver and an isbits ToRORd. Until those land, GPU lockstep sweeps of ToRORdTrauma stay in ArmyHeart, and this package ships the map executors and the lockstep interface. That gap is the price of having one solver.

**Batch equals solo.** For every case, `batch` under any executor must agree with that case run alone. Under map executors this is identity: the result is `simulate(apply(sim, case))`. Under `Lockstep` the reference is the simulation package's own single run with the case's parameters, and the package declares the tolerance, since a stacked kernel need not be bitwise identical to an unstacked one. A caller whose apply function and the package's `stacked` disagree about what a case means will see it here. This is the package's one golden-master rule, and it is aimed at the class of bug ArmyHeart's half-diffusion error belonged to: a batched run that looks plausible and is wrong only relative to the same case run alone.

**Lockstep cases share fate.** One case going non-finite poisons nothing else numerically, since cases are not coupled, but there is no per-case abort, and an error on any device fails the whole run. Only `Workers` isolates per-case failures. Studies that need isolation pick a map executor and pay for it in throughput.

**Heterogeneous cases are the simulation package's call.** Whether cases with different grids, timesteps, or stimulus protocols can share an array is decided by `stacked`, which must throw on a set it cannot stack rather than silently run a subset. This package does not pre-filter, because it cannot interpret a case.

**API surface.** Exported: `Simulation`, `batch`, `simulate`, the executors. Public by qualified name and documented as the extension interface: `stacked`, `unstacked`. No cardiac name appears anywhere in this package.

**Migrating from ArmyHeart.** A sweep script changes in two places: the cable is built with FlashBang, and `batch(config, params; backend)` becomes `batch(apply, sim, cases, executor)`. Results from the two solvers are not expected to match numerically, since the discretizations differ and ArmyHeart at `c45ffdd` integrated $\kappa/2$; κ values calibrated against ArmyHeart must be recalibrated against FlashBang. Establishing that agreement is FlashBang's validation work, not this package's.

## 7. Non-goals

- **Grids, tissue models, cell models, integrators, kernels.** FlashBang, CytoZoo, and their neighbors own these; this is principle 1.
- **Designing FlashBang's batched layout.** That belongs in FlashBang's DESIGN.md. This document fixes only the two functions it must answer to.
- **Per-case failure isolation under `Lockstep`.** See §6.
- **Interpreting, validating, or generating cases.** Generation is the study-design layer's future job; interpretation is the simulation package's.
- **Sensitivity estimators, optimization, calibration, surrogates.** Excluded by the README's boundaries, unchanged here.
- **The cardiac sweep layer, prepace caches, and walk viewer.** `sweep_helpers.jl`, `prepace_disk_cache.jl`, and `boundary_walk.jl` stay in ArmyHeart until their own spin-out rows are designed.
- **Changes to ArmyHeart.** It remains the frozen reference and keeps its own batching while FlashBang catches up.

## 8. Ecosystem alignment

**FlashBang.** The relationship is the one FlashBang has with CardiacAbstractions, turned around: this package is the light base that owns two functions, and FlashBang is the heavy implementer that adds methods to them. FlashBang's principle that the layout has exactly one owner is why `stacked` is its method and not this package's code.

**SciML.** `Simulation` is a `solve` call waiting to happen, and the map path mirrors `EnsembleProblem` on purpose. The divergence is `stacked`, which SciML's ensemble interface has no slot for.

**LockstepODE.** FlashBang's DESIGN.md already considers delegating its reaction half to LockstepODE's batched machinery. That decision sits entirely behind `stacked` and is invisible here, which is the contract working as intended.

**KernelAbstractions.** A device family is a backend value, as in the rest of the ecosystem. This package now writes no kernels, so its only uses are counting and selecting devices (open question 4).

## 9. Open questions

1. **`batch` calling convention.** (a) Typed executors passed positionally, as this document specifies; (b) ArmyHeart's keywords. **Recommend (a):** the package has no existing callers, so now is the cheapest moment, and it removes the overloaded `backend`.
2. **Where FlashBang's methods live.** (a) In FlashBang, as a package extension triggered by ParameterStudies; (b) in a ParameterStudies extension triggered by FlashBang; (c) hoist the two functions into a zero-dependency base both depend on. **Recommend (a):** the methods are layout code and FlashBang's rules give its layout one owner. (c) becomes right only if a second simulation package wants the interface without the executors.
3. **Return type.** (a) A `Vector` of whatever `simulate` returns, in case order; (b) a typed result carrying the cases and provenance, with an optional per-case reducer applied as sub-batches finish so a large sweep never holds every full solution. **Recommend (a) for now, then (b)** once the study store consumes results, since the store is what would define its fields.
4. **KernelAbstractions as a dependency.** (a) Hard dependency, with `Lockstep{B <: Backend}`; (b) weak dependency, with device counting and selection in an extension, leaving the core package dependency-light. **Recommend (b):** with no kernels written here the hard dependency buys only a type bound, and KernelAbstractions 0.10 brings PoCL along, which light consumers such as CardiacViz and a FlashBang extension would otherwise inherit.
5. **Proving the lockstep interface before FlashBang can answer it.** (a) Ship map executors and the interface now, and exercise `Lockstep` in this package's tests against a toy stacked simulation (N scalar ODEs as one array); (b) also write `stacked`/`unstacked` methods over ArmyHeart's existing batched solver, as a temporary adapter outside this package, so real sweeps run through `Lockstep` immediately. **Recommend (a):** the toy is the interface's second implementer and keeps CI free of unregistered dependencies; (b) is worth doing only if FlashBang's batched layout is far off.

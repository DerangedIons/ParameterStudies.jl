# ParameterStudies.jl — Design Specification

**Status:** draft for review · **Date:** 2026-11-09

This document specifies how ParameterStudies takes on ArmyHeart's 1D cable batching, at the level of abstractions and type hierarchy. It stops above implementation detail: keyword surfaces, compat bounds, and step-by-step porting order are settled after this shape is. It deliberately does not redesign the study lifecycle the README describes (study designs, canonical identities, resumable storage, provenance), which remains future work and interacts with this design only through open question 2. The source being ported is ArmyHeart at `c45ffdd` with the diffusion fix from `39726ee`. This design reverses the ParameterStudies rows of ArmyHeart's `SPINOUT_PLAN.md` (domain-neutral, no CUDA, RallypointOne): the package moved to DerangedIons and is now cardiac-aware by decision. The style draws on three studied inspirations: **LockstepODE.jl** (N systems sharing a timestep, backend-generic through KernelAbstractions), **FlashBang.jl** (a monodomain model as an inert declaration, kept separate from its discretization), and **DiffEqGPU.jl** (the ensemble algorithm as a typed object holding the backend).

## 1. What this adds

A cardiac parameter sweep asks the same question thousands of times with different ion-channel scalings: run a cable, classify what it did. Run independently, every cable pays for its own kernel launches and its own array traffic. But every cable has the same grid, timestep, and cell model code, and differs only by a few numbers. That makes the sweep a single 3D array, `[N_cell, N_spatial, N_cables]`, advanced by one kernel launch, which is where a GPU earns its speedup.

The central claim of this document is that **there are two distinct execution shapes, and the type system should say so**. *Mapping* runs an arbitrary function over independent cases: it isolates failures and imposes no structure on the cases. *Lockstep* advances many cases as one array: it is fast because the cases are forced to share everything except their declared parameters, and for the same reason they also share fate. ArmyHeart merges these under one `batch` name with a `backend` keyword that means a `Symbol` in one method and a device in the other. Here they are separate types with separate contracts.

## 2. Design principles

1. **Lockstep is the unit of GPU execution.** The kernel launch covers all cables at once; anything that would break that shared launch is a non-goal, not a feature request.
2. **What varies is declared; everything else is shared.** Per-cable parameters and initial states are the only degrees of freedom, so the batch's structure is visible in its configuration.
3. **Execution strategy is a typed value.** Serial, threaded, worker-pool, and lockstep execution are objects passed positionally, replacing a keyword whose meaning changed by method.
4. **KernelAbstractions is the only device layer.** Every CUDA call in the source maps to a backend-generic call, so the package loads and runs with no GPU stack and a user selects a device by passing `CUDABackend()`.
5. **The cell model is a duck-typed contract.** This package depends on no cell-model package and none depends on it, which is what lets ArmyHeart's existing models run without changes.
6. **No state is captured across a buffer swap.** Anything derived from the integrator's state is recomputed after the swap; §6 explains the bug this rule exists to prevent.
7. **The numbers are the contract.** A cable run must match ArmyHeart at `39726ee` under declared tolerances, and any divergence is named and justified.

## 3. The grammar

```
CableGrid + CableModel  →  CableBatchConfig  →  batch(config, params, Lockstep(backend))  →  (times, voltages)
(shared geometry, law)     (shared vs varying)       (one array per device)                  (host matrices)
```

The three canonical programs, which become the docs quickstarts and acceptance examples:

```julia
# (a) GPU sweep: every combination of two channel scalings, spread across all GPUs
config = CableBatchConfig(;
    create_model = () -> CableModel(; κ = κ_calibrated, ion = make_cell(CuVector{T}(constants))),
    grid = CableGrid(N_nodes, L),
    u0 = prepaced_state,
    parameter_index = ToRORdTrauma.parameter_index,
    varying = ["ICaL_Multiplier", "IKr_Multiplier"],
    tspan = (t_start, t_end),
    dt,
    coordinates = Vec.(cable_coordinates(grid)),
)
params = [Dict("ICaL_Multiplier" => a, "IKr_Multiplier" => b) for a in As for b in Bs]
times, voltages = batch(config, params, Lockstep(CUDABackend(); batch_size); saveat)

# (b) the same study on a workstation: only the executor changes
times, voltages = batch(config, params, Lockstep(CPU()); saveat)

# (c) independent cases that cannot share an array, such as cables with different grids
batch(run_and_save_case, cases, Threaded())
```

## 4. The type system

```
CableGrid{T}                       # one uniform 1D geometry, shared by every cable in a batch
CableModel{T, I}                   # the tissue law: κ and the cell model (the ion slot)
CableBatchConfig{T, …}             # everything the batch shares, and the names of what varies

AbstractMapExecutor                # runs any f over independent cases
├── Serial                         #   in order, on the calling task
├── Threaded                       #   OhMyThreads tasks
└── Workers                        #   Distributed.pmap on an existing pool; isolates per-item failures

Lockstep{B, E}                     # runs a CableBatchConfig as shared arrays on backend B

# internal machinery, public for those who write their own solve loop
BatchedRushLarsenIntegrator{…}     # reaction: owns state, swaps u and u_new each step
BatchedDiffusionOperator{…}        # diffusion: second-order, no-flux Laplacian
BatchedMonodomainSetup{…}          # Lie splitting of the two; the object step_batched! advances
```

The mockups below are indicative, not final: they show what each type owns and how it is parameterized. Field names may move in the implementation pass, constructors and validation are omitted, and `…` marks deferred detail.

**`CableGrid` and `CableModel`** replace `StructuredGrid` + `FiniteDifferenceDiscretization` and Thunderbolt's `MonodomainModel`. Thunderbolt is unregistered and cannot be a dependency, and the batched path used little of it anyway: `.stim`, `χ`, and `Cₘ` were never read, and `disc` was accepted and silently ignored, so a caller who passed fourth order or Dirichlet got second-order no-flux with no warning. Narrowing the types to what the solver actually does turns that silent lie into an impossible input. The trade-off is that these types are deliberately not general; anything beyond a uniform 1D cable belongs in FlashBang.

```julia
struct CableGrid{T <: Real}
    N      :: Int       # nodes per cable
    L      :: T         # cable length
    origin :: T
end

struct CableModel{T <: Real, I}
    κ   :: T            # uniform diffusivity
    ion :: I            # any object meeting the cell-model contract in §6
end
```

**`CableBatchConfig`** is the root object of a lockstep study: it separates what the batch shares from what varies. `create_model` is a factory rather than a value because a GPU-resident model has to be built after its device is selected, once per device; passing a ready-made model would pin every device's work to one GPU's memory. `parameter_index` is a plain function so the config works with any module's lookup, where ArmyHeart reached into a `CELLMODULE`. It replaces `GPUBatchConfig`, whose name stopped being accurate once CPU runs became first-class. `coordinates` is overridable because ToRORdTrauma's stimulus falls back to a `stim(x::Vec{1}, t)` method, so the default coordinate type would throw a `MethodError` against that model.

```julia
Base.@kwdef struct CableBatchConfig{T <: Real, F, P, U, X}
    create_model    :: F                # () -> CableModel, called once per device after selecting it
    grid            :: CableGrid{T}
    u0              :: U                # [N_cell, N_spatial] shared, or [N_cell, N_spatial, N_cables] per cable
    parameter_index :: P                # name -> index into cell_parameters(ion)
    varying         :: Vector{String}   # the only parameters allowed to differ between cables
    tspan           :: Tuple{T, T}
    dt              :: T
    coordinates     :: X = cable_coordinates(grid)   # any isbits element the cell model accepts
end
```

**`AbstractMapExecutor`** owns "run this function over these cases" and makes no assumption about what the function does. It replaces `backend=:threads` and friends. `Workers` uses the pool the caller already set up: ArmyHeart's in-function `addprocs` created workers with no packages loaded, and `on_error` then reduced every resulting failure to a warning, so a run could complete having done nothing.

```julia
abstract type AbstractMapExecutor end

struct Serial <: AbstractMapExecutor end

struct Threaded <: AbstractMapExecutor
    ntasks :: Int
end

struct Workers <: AbstractMapExecutor end    # caller owns addprocs and @everywhere using
```

**`Lockstep`** owns where and how a shared-array solve runs. It is not an `AbstractMapExecutor`, on purpose: a map executor accepts any function, while lockstep accepts only a `CableBatchConfig`, so one hierarchy would make `batch(f, cases, Lockstep(…))` type-check and then fail at runtime. Cables are divided evenly across devices, one pinned thread per device, then into sub-batches of `batch_size` on each device, because a device has finite memory no matter how many cables a sweep needs. `inner` replaces the `internal_threading::Bool` flag and reuses the map vocabulary for CPU loops, so `Lockstep(CPU(); inner = Serial())` reads as what it does. GPU runs ignore it.

```julia
struct Lockstep{B <: KernelAbstractions.Backend, E <: AbstractMapExecutor}
    backend     :: B
    batch_size  :: Int                       # cables per kernel launch, limited by device memory
    max_devices :: Int                       # defaults from KernelAbstractions.ndevices(backend)
    inner       :: E                         # CPU loop strategy
end
```

**The internal machinery** ports nearly unchanged, with the integrator's abstract-typed fields tightened to concrete parameters. The reaction step and the diffusion step alternate as Lie splitting, and they meet in exactly one place: the voltage slice of the state. `BatchedMonodomainSetup` loses its `V_batched` field, which is the §6 bug. `BatchedDiffusionOperator` loses the `stim`, `t`, and `x_positions` arguments its kernel never read, and on `CPU` it runs a host loop rather than a kernel, because KernelAbstractions 0.10 makes `CPU` an alias of `POCLBackend` and a plain CPU run should not route through an OpenCL compiler.

```julia
mutable struct BatchedRushLarsenIntegrator{T, CM, B, A3, M, X, E}
    cellmodel    :: CM
    u            :: A3     # [N_cell, N_spatial, N_cables]
    u_new        :: A3     # swapped with u every step, so never hold a view of either
    x_positions  :: X
    p_all_cables :: M      # [N_params, N_cables], expanded once on the host from the varying set
    t            :: T
    dt           :: T
    backend      :: B
    inner        :: E
    …                      # N_cell, N_spatial, N_cables
end

struct BatchedDiffusionOperator{T, B, E}
    N_spatial :: Int
    N_cables  :: Int
    dx        :: T
    κ         :: T
    backend   :: B
    inner     :: E
end

struct BatchedMonodomainSetup{T, CI, DO, DV}
    cell_integrator :: CI
    diffusion_op    :: DO
    dV_batched      :: DV     # [N_spatial, N_cables] diffusion buffer
    dt              :: T
    tspan           :: Tuple{T, T}
    …                         # N_spatial, N_cables
end
```

## 5. Alternatives considered

**Split the code: generic executors here, cable solver in FlashBang.** This keeps ParameterStudies domain-neutral, gives the solver the home its physics suggests, and keeps KernelAbstractions and its PoCL dependency out of packages that only want study records. It was offered and declined in favor of one package. In its favor as a decision: FlashBang today lists Rush-Larsen inner solvers as out of scope and has no batching, so landing the solver there means building those first, while the integrator, kernel, and device loop change together and stay simple to develop as a unit. The cost is real and accepted: this package no longer matches its original charter.

**Build on LockstepODE's `BatchedFunction`.** LockstepODE already has backend-generic batching with tested zero-allocation evaluation, and `SPINOUT_PLAN.md` assigned it this job. But `BatchedFunction` is a right-hand side, `(du, u, p, t)`, handed to OrdinaryDiffEq, while Rush-Larsen is a step map, `u → u_new`, supplied by the cell model itself. Adapting one to the other means writing a Rush-Larsen algorithm for the SciML interface first. This becomes the better shape if a registered Rush-Larsen algorithm lands; the integrator could then be replaced without touching the config or executor layer.

**DiffEqGPU's `EnsembleGPUKernel`.** Its documented form is an out-of-place ODE on StaticArrays with `prob_func` for per-trajectory variation, which is an elegant fit for small systems. A cable of 65 states over hundreds of nodes is not a static-size value, and the spatial coupling within a cable is the point of the simulation. The typed-ensemble-algorithm idea is what this design keeps.

**CUDA as a weakdep extension.** This is the conventional shape and makes the GPU boundary explicit in the package's layout. Every CUDA use in the source maps to a KernelAbstractions or Adapt call, including device selection, so an extension would be empty. KernelAbstractions becomes the hard dependency instead.

**Keep ArmyHeart's keyword-driven `batch`.** It is familiar, and porting it verbatim would be less work. It overloads `backend`, leaves `internal_threading` as a boolean, and lets an old GPU script run on CPU without complaint. See open question 1.

## 6. Cross-cutting rules

**The cell-model contract.** A cell model is callable as `cell(u, t, dt, p, u_new, x)`, with `t` and `dt` of the state's element type. It must write every entry of `u_new`, since that buffer is reused. `u[1]` is voltage. It must be Adapt-compatible and isbits after adaptation. `cell_parameters(cell)` returns its parameter vector, defaulting to `cell.CONSTANTS` and overridable. `ToRORdTraumaModel` already satisfies this with no new methods.

**Never hold a view across the swap.** `step!` swaps `u` and `u_new` every step, but a Julia `SubArray` pins its parent *object*, not the struct field. ArmyHeart stored `V_batched` at construction, so after every odd step that view pointed at the scratch buffer: diffusion was written there, then overwritten wholesale by the next reaction step. Exactly half the diffusion increments vanished and the solver integrated $\kappa/2$; `39726ee` measured the ratio at 0.5. Nothing looked wrong, because a cable at $\kappa/2$ still conducts, only slower. The rule here: `voltage(setup)` recomputes the view, and is called after `step!` and at each save. `legacy_half_diffusion` is not carried over, since a new package has no old results to reproduce; `39726ee` remains available for that.

**A consequence for users.** The same $\kappa$ now conducts faster than on `c45ffdd`, roughly as $\sqrt{2}$, so κ values calibrated against the old solver must be recalibrated.

**Lockstep cables share fate.** One cable going non-finite poisons nothing else numerically, since cables are not coupled, but there is no per-cable abort, and an error on any device fails the whole run. Only `Workers` isolates per-item failures. Studies that need isolation pick a map executor and pay for it in throughput.

**Precision.** Float64 is the supported path. ArmyHeart's `PRECISION.md` reports non-finite results in a majority of the parameter space under Float32, and the time loop accumulates `t += dt`.

**API surface.** Exported: `CableGrid`, `CableModel`, `CableBatchConfig`, `batch`, the executors, `extract_cable_solution`. Reachable by qualified name and documented: `semidiscretize_rush_larsen_batched`, `step_batched!`, `solve_batched`, `voltage`, `cell_parameters`. The second tier exists because ArmyHeart's boundary walker forks the solve loop, a real pattern this package should support without exposing integrator fields.

**Equivalence.** A short ToRORdTrauma cable must match ArmyHeart at `39726ee` within tolerances declared before the comparison is run. Bitwise equality is not required, since the diffusion loop is rewritten for CPU.

## 7. Non-goals

- **2D, 3D, or unstructured domains.** FlashBang's job.
- **Spatially varying κ, non-uniform grids, or other boundary conditions.** The batched kernel assumes all three away, and that is why it is one short loop.
- **Different grids or spatial functions per cable.** Those break lockstep's shared launch; use a map executor.
- **Adaptive timestepping.** Cables would need different steps, which is the opposite of lockstep.
- **Per-cable failure isolation under `Lockstep`.** See §6.
- **The cardiac sweep layer, prepace caches, and walk viewer.** `sweep_helpers.jl`, `prepace_disk_cache.jl`, and `boundary_walk.jl` stay in ArmyHeart; only batching moves here.
- **Changes to ArmyHeart.** It remains the frozen reference, and active work on it continues.

## 8. Ecosystem alignment

**FlashBang.** Where the two overlap, this package conforms: `κ` and `ion` match `MonodomainModel`'s field names, so a cable model reads the same in either. It diverges in being 1D-only and in not adopting SciML's `solve` for the batch, because the Rush-Larsen step map does not fit that interface today.

**CytoZoo.** Its `AbstractCellModel` is where cell models are heading, with `rush_larsen_step!(u_new, u, p, t, dt, model)` and `parameter_index(model, name)`. Its argument order differs from ArmyHeart's contract. Because the contract here is duck-typed and `parameter_index` is a plain function, a CytoZoo adapter is a thin wrapper rather than a redesign (open question 3).

**KernelAbstractions.** A device is a backend value, as in the rest of the ecosystem. KernelAbstractions 0.10 changes what `CPU` means, so CPU code paths dispatch on `KernelAbstractions.CPU` alongside the generic backend method.

## 9. Open questions

1. **`batch` calling convention.** (a) Typed executors passed positionally, as this document specifies; (b) ArmyHeart's keywords, with `backend` required for the config method. **Recommend (a):** the package has no existing callers, so now is the cheapest moment, and it removes the overloaded `backend`.
2. **Does the README's study lifecycle stay in this package?** (a) Keep it as the roadmap, with cable batching as the first execution backend; (b) narrow the package to cardiac batching and move the lifecycle elsewhere or drop it. **Recommend (a):** a study produces evaluation requests and `Lockstep` is one way to run them, so the two are complementary, and CardiacViz planned to depend on the study layer.
3. **Cell-model contract.** (a) Duck-typed on ArmyHeart's call form now, CytoZoo adapter later; (b) adopt `CytoZoo.AbstractCellModel` immediately. **Recommend (a):** ToRORdTrauma satisfies it today, whereas (b) requires a rewrite of the model, which is what ToRORdLand is for.
4. **Return type.** (a) The existing `(times, Vector{Matrix})` tuple; (b) a typed `CableBatchSolution` carrying the config and parameters. **Recommend (a) for now, then (b)** once the study store consumes results, since the store is what would define its fields.
5. **KernelAbstractions versions.** (a) Allow 0.9 and 0.10; (b) pin 0.9 to keep PoCL and compiler JLLs out of light consumers. **Recommend (a):** recent CUDA versions require 0.10, and pinning would cap GPU support. Option (b) trades that away to make CardiacViz lighter, so it is worth revisiting only if the load cost proves material.

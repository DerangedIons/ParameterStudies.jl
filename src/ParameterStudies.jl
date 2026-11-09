module ParameterStudies

using Adapt: adapt
using CommonSolve: solve
using Distributed: pmap
using KernelInterface: Backend, device!, ndevices
using OhMyThreads: tmap

export Simulation, simulate, batch, Serial, Threaded, Workers, Lockstep
@static if VERSION >= v"1.11.0-DEV.469"
    eval(Meta.parse("public stacked, unstacked"))
end

"""
    Simulation(problem, alg; options...)

One runnable simulation: a problem, its algorithm, and the options a single run would pass
(`dt`, `saveat`, callbacks). All three are opaque here; errors in them surface from the
package that owns the problem.
"""
struct Simulation{P, A, O}
    problem::P
    alg::A
    options::O
end
Simulation(problem, alg; options...) = Simulation(problem, alg, values(options))

"""
    simulate(sim) -> result

Run one simulation. Defaults to `solve(sim.problem, sim.alg; sim.options...)`, so any
CommonSolve problem works unchanged; a package that does not speak `solve` adds a method on
its own problem type instead.
"""
simulate(sim::Simulation) = solve(sim.problem, sim.alg; sim.options...)

"""
    stacked(problem, cases) -> problem

Lockstep interface, implemented by the simulation package: one problem that advances all
`cases` as shared arrays along a case axis. Must throw on a set of cases it cannot stack.
"""
function stacked end

"""
    unstacked(result) -> results

Lockstep interface, implemented by the simulation package: split the result of a stacked
run into one result per case, in the order the cases were given to [`stacked`](@ref).
"""
function unstacked end

"Runs cases independently; asks nothing of the simulation package."
abstract type AbstractMapExecutor end

"Run cases in order on the calling task."
struct Serial <: AbstractMapExecutor end

"Run cases on `ntasks` OhMyThreads tasks."
struct Threaded <: AbstractMapExecutor
    ntasks::Int
end
Threaded() = Threaded(Threads.nthreads())

"""
Run cases with `Distributed.pmap` on the pool the caller already set up (`addprocs`, then
`@everywhere using` what the cases need). A case that throws does not stop the others: its
exception is returned in its slot.
"""
struct Workers <: AbstractMapExecutor end

"""
    Lockstep(backend; batch_size, max_devices = ndevices(backend))

Run cases as stacked sub-batches of at most `batch_size` on the devices of `backend`.
Needs the simulation package to implement [`stacked`](@ref) and [`unstacked`](@ref). Cases
share fate: an error in any sub-batch fails the whole run.
"""
struct Lockstep{B <: Backend}
    backend::B
    batch_size::Int
    max_devices::Int
end
function Lockstep(backend; batch_size, max_devices = ndevices(backend))
    return Lockstep(backend, batch_size, max_devices)
end

"""
    batch(f, items, executor::AbstractMapExecutor)
    batch(apply, sim, cases, executor::AbstractMapExecutor)
    batch(sim, cases, executor::Lockstep)

Return one result per item, in order. The second form maps `simulate(apply(sim, case))`
over `cases`, where `apply(sim, case) -> sim` is the caller's own function. The third hands
the raw cases to the simulation package's `stacked` and does not use an `apply`.

Under any executor, each result must agree with that case run alone.
"""
batch(f, items, ::Serial) = map(f, items)
batch(f, items, ex::Threaded) = tmap(f, items; ntasks = ex.ntasks)
batch(f, items, ::Workers) = pmap(f, items; on_error = identity)
function batch(apply, sim::Simulation, cases, ex::AbstractMapExecutor)
    return batch(case -> simulate(apply(sim, case)), cases, ex)
end

function batch(sim::Simulation, cases, ex::Lockstep)
    isempty(cases) && return []
    per_device = cld(length(cases), min(ex.max_devices, length(cases)))
    tasks = map(enumerate(Iterators.partition(cases, per_device))) do (id, device_cases)
        Threads.@spawn begin
            device!(ex.backend, id)  # task-local, so each device gets its own task
            mapreduce(vcat, Iterators.partition(device_cases, ex.batch_size)) do sub
                # Stack on the host, then move: the prototype is pinned to no device.
                problem = adapt(ex.backend, stacked(sim.problem, sub))
                unstacked(simulate(Simulation(problem, sim.alg, sim.options)))
            end
        end
    end
    return mapreduce(fetch, vcat, tasks)
end

end

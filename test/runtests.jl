using ParameterStudies
using Test
using JET
using Adapt: Adapt
using CommonSolve: CommonSolve
using KernelInterface: KernelInterface

# The toy simulation package: u' = -k u by explicit Euler, with a stacked form in which
# k and u carry a case axis. It is the lockstep interface's implementer for these tests.
struct Decay{K, U}
    k::K
    u0::U
end
struct Euler end

function CommonSolve.solve(p::Decay, ::Euler; dt, nsteps)
    u = p.u0
    for _ in 1:nsteps
        u = @. u - dt * p.k * u
    end
    return u
end

ParameterStudies.stacked(p::Decay, cases) = Decay([c.k for c in cases], fill(p.u0, length(cases)))
ParameterStudies.unstacked(u::AbstractVector) = collect(u)

# Two devices, so the device split is exercised without a GPU.
struct TwoDevices <: KernelInterface.Backend end
KernelInterface.ndevices(::TwoDevices) = 2
KernelInterface.device!(::TwoDevices, id::Int) = nothing
Adapt.adapt_storage(::TwoDevices, x::AbstractArray) = x

@testset "ParameterStudies.jl" begin
    sim = Simulation(Decay(1.0, 2.0), Euler(); dt = 0.01, nsteps = 100)
    cases = [(k = k,) for k in range(0.1, 3.0; length = 11)]
    with_k(sim, case) = Simulation(Decay(case.k, sim.problem.u0), sim.alg, sim.options)
    solo = [simulate(with_k(sim, case)) for case in cases]

    # 11 cases over 2 devices in sub-batches of 4: uneven device split, ragged last batch.
    executors = (Serial(), Threaded(), Workers(), Lockstep(TwoDevices(); batch_size = 4))
    @testset "batch equals solo: $(nameof(typeof(ex)))" for ex in executors
        results = ex isa Lockstep ? batch(sim, cases, ex) : batch(with_k, sim, cases, ex)
        @test results ≈ solo
    end

    @testset "Workers isolates a failing case" begin
        results = batch(x -> x == 2 ? error("case 2") : x, 1:3, Workers())
        @test results[[1, 3]] == [1, 3]
        @test results[2] isa ErrorException
    end

    @testset "Lockstep shares fate" begin
        # One unstackable case (no `k`) on one device fails the whole run.
        ex = Lockstep(TwoDevices(); batch_size = 4)
        @test_throws TaskFailedException batch(sim, [cases; (; κ = 1.0)], ex)
    end

    @testset "Code linting (JET.jl)" begin
        JET.test_package(ParameterStudies; target_defined_modules = true)
    end
end

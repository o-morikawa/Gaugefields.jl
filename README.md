# Gaugefields

# Abstract

This is a package for lattice QCD codes.
Treating gauge fields (links), gauge actions with MPI and autograd.

This is an extended version by Okuto Morikawa,
in order to implement higher-form gauge fields
 (i.e., 't Hooft twisted boundary condition/flux).

[~~NOTE: O.M. also provides a memory-safer code set than the original one.~~
This has been fixed on v0.4.0.]

<img src="LQCDjl_block.png" width=300> 

This package will be used in [LatticeQCD.jl](https://github.com/akio-tomiya/LatticeQCD.jl). 
See also the orginal package in [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl).

O.M. would like to thank Yuki Nagai and Akio Tomiya (the contributors of the original package). O.M. is also grateful to Hiroshi Suzuki.

# What this package can do:
This package has following functionarities

- SU(Nc) (Nc > 1) gauge fields in 2 or 4 dimensions with arbitrary actions.
- **Z(Nc) 2-form gauge fields in 4 dimensions, which are given as 't Hooft flux.**
- U(1) gauge fields in 2 dimensions with arbitrary actions. 
- Configuration generation
    - Heatbath
    - quenched Hybrid Monte Carlo
    - **quenched Hybrid Monte Carlo being subject to 't Hooft twisted b.c.**
        - **with external (non-dynamical) Z(Nc) 2-form gauge fields**
    - **quenched Hybrid Monte Carlo for SU(Nc)/Z(Nc) gauge theory**
        - **with dynamical Z(Nc) 2-form gauge fields**
- Gradient flow via RK3
    - Yang-Mills gradient flow
    - **Yang-Mills gradient flow being subject to 't Hooft twisted b.c.**
    - **Gradient flow for SU(Nc)/Z(Nc) gauge theory**
- I/O: ILDG and Bridge++ formats are supported ([c-lime](https://usqcd-software.github.io/c-lime/) will be installed implicitly with [CLIME_jll](https://github.com/JuliaBinaryWrappers/CLIME_jll.jl) )
- MPI parallel computation (experimental. See documents.)
    - **quenched HMC with MPI being subject to 't Hooft twisted b.c.**

**The implementation of higher-form gauge fields is based on
[arXiv:2303.10977 [hep-lat]](https://arxiv.org/abs/2303.10977).**

Dynamical fermions will be supported with [LatticeDiracOperators.jl](https://github.com/akio-tomiya/LatticeDiracOperators.jl).

In addition, this supports followings
- **Autograd for functions with SU(Nc) variables**
- Stout smearing (exp projecting smearing)
- Stout force via [backpropagation](https://arxiv.org/abs/2103.11965)
- **A 3D implementation of U(N) gauge fields** [by Shiozaki](https://arxiv.org/abs/2403.05291)
    - Discrete approximation of winding number, and gradient-flow improvement
    - See [Wind3D](https://github.com/o-morikawa/Wind3D)

Autograd can be worked for general Wilson lines except for ones have overlaps.

# Install

```
add Wilsonloop
add https://github.com/o-morikawa/Gaugefields.jl.git
```

## Development mode
This is a non-official package in Julia,
and you are recommended to use it as a develop (dev) package
if there's a possibility that you use the original Gaugefields.jl package
or modify it.

To install the oringinal package,
in Julia REPL in the package mode,
```
add Gaugefields.jl
```

Download the code locally, then in Julia REPL in the package mode,
```
dev /<your full path>/Gaugefields
```

When you use this package in Julia REPL, in the package mode,
```
activate Gaugefields
```
or, when in command line,
```
julia --project="Gaugefields" test.jl
```


# How to use

Please see the orginal docs in [Gaugefields.jl](https://github.com/akio-tomiya/Gaugefields.jl).
Basically, you can use this package in a same way as the original code
if the argument of any function, (..., U, ...), is rewritten by (..., U, B, ...).

# Generating configurations and File loading
## ILDG format for SU(N) guage fields
[ILDG](https://www-zeuthen.desy.de/~pleiter/ildg/ildg-file-format-1.1.pdf) format is one of standard formats for LatticeQCD configurations.

We can read ILDG format like: 

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

ildg = ILDG(filename)
i = 1
L = [NX,NY,NZ,NT]
load_gaugefield!(U,i,ildg,L,NC)
```
Then, we can calculate the plaquette: 

```julia
temps = Temporalfields(U[1], num=2)
comb, factor = set_comb(U,Dim)

@time plaq_t = calculate_Plaquette(U,B,temps)*factor
println("plaq_t = $plaq_t")
poly = calculate_Polyakov_loop(U,temps) 
println("polyakov loop = $(real(poly)) $(imag(poly))")
```

We can write a configuration as the ILDG format like 

```julia
filename = "hoge.ildg"
save_binarydata(U,filename)
```

## Text format for Bridge++
Gaugefields.jl also supports a text format for [Bridge++](https://bridge.kek.jp/Lattice-code/index_e.html). 

### File loading

```julia
using Gaugefields

filename = "testconf.txt"
load_BridgeText!(filename,U,L,NC)
```

### File saving

```julia
filename = "testconf.txt"
save_textdata(U,filename)
```

## Z(N) 2-form gauge fields

SU(N) gauge fields possess Z(N) center symmetry,
which is called 1-form global symmetry, a type of generalized symmetry.
To gauge the 1-form center symmetry,
we can define the Z(N) 2-form gauge fields in four dimensions, B, as
```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 0
NC = 3

flux=[1,0,0,0,0,1] # FLUX=[Z12,Z13,Z14,Z23,Z24,Z34]

println("Flux is ", flux)

U1 = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
B1 = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")

println("Initial conf of B at [1,2][2,2,:,:,NZ,NT]")
display(B1[1,2][2,2,:,:,NZ,NT])
```

# Hybrid Monte Carlo
## Non-dynamical higher-form gauge fields
We can do the HMC simulations. The example code is as follows.
```julia

using Random
using Gaugefields
using LinearAlgebra

function HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
    Dim = 4
    Nwing = 0

    flux = Flux
    println("Flux : ", flux)

    Random.seed!(123)


    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    temps = Temporalfields(U[1], num=9)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    MDsteps = 50
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temps)
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 5.7
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    Flux = [0,0,1,1,0,0]
    #HMC_test_4D(NX,NY,NZ,NT,NC,β)
    HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
end
main()
```

## Dynamical higher-form gauge fields
HMC simulations with dynamical B fields are as follows:
```julia

using Random
using Gaugefields
using Wilsonloop
using LinearAlgebra

function HMC_test_4D_dynamicalB(NX,NY,NZ,NT,NC,β)
    Dim = 4
    Nwing = 0

    Random.seed!(123)

    flux = [1,1,1,1,2,0]

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    L = [NX,NY,NZ,NT]
    filename = "test/confs/U_beta6.0_L8_F111120_4000.txt"
    load_BridgeText!(filename,U,L,NC)

    temps = Temporalfields(U[1], num=9)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U)
    Uold  = similar(U)
    Bold = similar(B)
    flux_old = zeros(Int, 6)

    MDsteps = 50 # even integer!!!
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(
                gauge_action,
                U,
                B,
                flux,
                p,
                MDsteps,
                Dim,
                Uold,
                Bold,
                flux_old,
                temps
            )
        end
        if get_myrank(U) == 0
             println("Flux : ", flux)
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temps)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 6.0
    NX = 8
    NY = 8
    NZ = 8
    NT = 8
    NC = 3
    HMC_test_4D_dynamicalB(NX,NY,NZ,NT,NC,β)
end
main()
```

# Gradient flow
## A simple case
We can use Lüscher's gradient flow.

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 0
NC = 3

flux=[1,0,0,0,0,1] # FLUX=[Z12,Z13,Z14,Z23,Z24,Z34]

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")


temps = Temporalfields(U[1], num=3)
comb, factor = set_comb(U,Dim)

g = Gradientflow(U, B)
for itrj=1:100
    flow!(U,B,g)
    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("$itrj plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
end
```

## Gradient flow with general terms
We can do the gradient flow with general terms with the use of Wilsonloop.jl, which is shown below.
The coefficient of the action can be complex. The complex conjugate of the action defined here is added automatically to make the total action hermitian.   
The code is 

```julia

using Random
using Test
using Gaugefields
using Wilsonloop

function gradientflow_test_4D(NX,NY,NZ,NT,NC)
    Dim = 4
    Nwing = 1

    Random.seed!(123)

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",randomnumber="Reproducible")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #Plaquette term
    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)
        end
    end

    #Rectangular term
    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
            loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
            
            push!(loops,loop1)
        end
    end

    listloops = [loops_p,loops]
    listvalues = [1+im,0.1]
    g = Gradientflow_general(U,listloops,listvalues,eps = 0.01)

    for itrj=1:100
        flow!(U,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps) 
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end
    return plaq_t

end


function gradientflow_test_2D(NX,NT,NC)
    Dim = 2
    Nwing = 1
    U = Initialize_Gaugefields(NC,Nwing,NX,NT,condition = "hot",randomnumber="Reproducible")

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("0 plaq_t = $plaq_t")
    poly = calculate_Polyakov_loop(U,temps) 
    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    #g = Gradientflow(U,eps = 0.01)
    #listnames = ["plaquette"]
    #listvalues = [1]
    loops_p = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end

            loop1 = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim)
            push!(loops_p,loop1)

        end
    end


    loops = Wilsonline{Dim}[]
    for μ=1:Dim
        for ν=μ:Dim
            if ν == μ
                continue
            end
            loop1 = Wilsonline([(μ,1),(ν,2),(μ,-1),(ν,-2)],Dim = Dim)
            push!(loops,loop1)
            loop1 = Wilsonline([(μ,2),(ν,1),(μ,-2),(ν,-1)],Dim = Dim)
            
            push!(loops,loop1)
        end
    end

    listloops = [loops_p,loops]
    listvalues = [1+im,0.1]
    g = Gradientflow_general(U,listloops,listvalues,eps = 0.01)

    for itrj=1:100
        flow!(U,g)
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
            poly = calculate_Polyakov_loop(U,temps)
            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
        end
    end

    return plaq_t

end



const eps = 0.1


println("2D system")
@testset "2D" begin
    NX = 4
    #NY = 4
    #NZ = 4
    NT = 4
    Nwing = 1

    @testset "NC=1" begin
        β = 2.3
        NC = 1
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end
    #error("d")
    
    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")
        @time plaq_t = gradientflow_test_2D(NX,NT,NC)
    end
end

println("4D system")
@testset "4D" begin
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1


    
    @testset "NC=2" begin
        β = 2.3
        NC = 2
        println("NC = $NC")
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end

    @testset "NC=3" begin
        β = 5.7
        NC = 3
        println("NC = $NC")
        @time plaq_t = gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end

    @testset "NC=4" begin
        β = 5.7
        NC = 4
        println("NC = $NC")

        val = 0.7301232810349298
        @time plaq_t =gradientflow_test_4D(NX,NY,NZ,NT,NC)
    end


end
```

# HMC with MPI
Here, we show the HMC with MPI.
the REPL and Jupyternotebook can not be used when one wants to use MPI.
At first, in Julia REPL in the package mode,
```
add MPI
```
Then,
```julia
using MPI
MPI.install_mpiexecjl()
```
and
```
export PATH="/<your home path>/.julia/bin/:$PATH"
```

The command is like:
```
mpiexecjl --project="Gaugefields" -np 2 julia mpi_sample.jl 1 1 1 2 true
```
```1 1 1 2``` means ```PEX PEY PEZ PET```. In this case, the time-direction is diveded by 2. 

The sample code is written as 
```julia

using Random
using Gaugefields
using Wilsonloop
using LinearAlgebra
using MPI

if length(ARGS) < 5
    error("USAGE: ","""
    mpirun -np 2 exe.jl 1 1 1 2 true
    """)
end
const pes = Tuple(parse.(Int64,ARGS[1:4]))
const mpi = parse(Bool,ARGS[5])

function HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
    Dim = 4
    Nwing = 0

    flux = Flux

    Random.seed!(123)

    if mpi
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot",mpi=true,PEs = PEs,mpiinit = false) 
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux",mpi=true,PEs = PEs,mpiinit = false)
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "hot")
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
    end

    if get_myrank(U) == 0
        println("Flux : ", flux)
    end

    if get_myrank(U) == 0
        println(typeof(U))
    end


    temps = Temporalfields(U[1], num=10)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    if get_myrank(U) == 0
        println("0 plaq_t = $plaq_t")
    end
    poly = calculate_Polyakov_loop(U,temps) 
    if get_myrank(U) == 0
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")
    end

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    MDsteps = 50
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temps)
        end
        if get_myrank(U) == 0
            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temps)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            plaq_t = calculate_Plaquette(U,B,temps)*factor
            if get_myrank(U) == 0
                println("$itrj plaq_t = $plaq_t")
            end
            poly = calculate_Polyakov_loop(U,temps) 
            if get_myrank(U) == 0
                println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
                println("acceptance ratio ",numaccepted/itrj)
            end
        end
    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 5.7
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    NC = 3
    Flux = [0,0,1,1,0,0]
    #HMC_test_4D(NX,NY,NZ,NT,NC,β)
    HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
end
main()
```

# Utilities

## Data structure
We can access the gauge field defined on the bond between two neigbohr points. 
In 4D system, the gauge field is like ```u[ic,jc,ix,iy,iz,it]```. 
There are four directions in 4D system. Gaugefields.jl uses the array like: 

```julia
NX = 4
NY = 4
NZ = 4
NT = 4
Nwing = 1
Dim = 4

U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

```

In the later exaples, we use, ``mu=1`` and ``u=U[mu]`` as an example.

## Hermitian conjugate (Adjoint operator)
If you want to get the hermitian conjugate of the gauge fields, you can do like 

```julia
u'
```

This is evaluated with the lazy evaluation. 
So there is no memory copy. 
This returms $U_\mu^\dagger$ for all sites.

## Shift operator
If you want to shift the gauge fields, you can do like 

```julia
shifted_u = shift_U(u, shift)
```
This is also evaluated with the lazy evaluation. 
Here ``shift`` is ``shift=(1,0,0,0)`` for example.

## matrix-field matrix-field product
If you want to calculate the matrix-matrix multiplicaetion on each lattice site, you can do like

As a mathematical expression, for matrix-valued fields ``A(n), B(n)``,
we define "matrix-field matrix-field product" as,

```math
[A(n)B(n)]_{ij} = \sum_k [A(n)]_{ik} [B(n)]_{kj}
```

for all site index n.
<!--<img src="https://latex.codecogs.com/svg.image?[A(n)B(n)]_{ij}&space;=&space;\sum_k&space;[A(n)]_{ik}&space;[B(n)]_{kj}" title="[A(n)B(n)]_{ij} = \sum_k [A(n)]_{ik} [B(n)]_{kj}" />-->

In our package, this is expressed as,

```julia
mul!(C,A,B)
```
which means ```C = A*B``` on each lattice site. 
Here ``A, B, C`` are same type of ``u``.

## Trace operation 
If you want to calculate the trace of the gauge field, you can do like 

```julia
tr(A)
```
It is useful to evaluation actions. 
This trace operation summing up all indecis, spacetime and color.

# Applications

This package and Wilsonloop.jl enable you to perform several calculations.
Here we demonstrate them.

Some of them will be simplified in LatticeQCD.jl.

## Wilson loops
We develop [Wilsonloop.jl](https://github.com/akio-tomiya/Wilsonloop.jl.git), which is useful to calculate Wilson loops. 
If you want to use this, please install like

```
add Wilsonloop.jl
```

For example, if you want to calculate the following quantity: 

```math
U_{1}(n)U_{2}(n+\hat{1}) U^{\dagger}_{1}(n+\hat{2}) U^{\dagger}_2(n) e^{-2\pi B_{12}(n) / N} ,
```
which is Z(Nc) 1-form gauge invariant [[arXiv:2303.10977 [hep-lat]](https://arxiv.org/abs/2303.10977)].

You can use Wilsonloop.jl as follows

```julia
using Wilsonloop
loop = [(1,1),(2,1),(1,-1),(2,-1)]
w = Wilsonline(loop)
```
The output is ```L"$U_{1}(n)U_{2}(n+e_{1})U^{\dagger}_{1}(n+e_{2})U^{\dagger}_{2}(n)$"```. 
Then, you can evaluate this loop with the use of the Gaugefields.jl like: 

```julia
using LinearAlgebra
NX = 4
NY = 4
NZ = 4
NT = 4
NC = 3
Nwing = 0
Dim = 4
U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")

flux=[1,0,0,0,0,1] # FLUX=[Z12,Z13,Z14,Z23,Z24,Z34]
B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")

temp1 = similar(U[1])
temp2 = similar(U[1])
temp3 = similar(U[1])
temp4 = similar(U[1])
V = similar(U[1])

evaluate_gaugelinks!(V,w,U,B,[temp1,temp2,temp3,temp4])
println(tr(V))
```

For example, if you want to calculate the clover operators, you can define like: 

```julia
function make_cloverloop(μ,ν,Dim)
    loops = Wilsonline{Dim}[]
    loop_righttop = Wilsonline([(μ,1),(ν,1),(μ,-1),(ν,-1)],Dim = Dim) # Pmunu
    push!(loops,loop_righttop)
    loop_rightbottom = Wilsonline([(ν,-1),(μ,1),(ν,1),(μ,-1)],Dim = Dim) # Qmunu
    push!(loops,loop_rightbottom)
    loop_leftbottom= Wilsonline([(μ,-1),(ν,-1),(μ,1),(ν,1)],Dim = Dim) # Rmunu
    push!(loops,loop_leftbottom)
    loop_lefttop = Wilsonline([(ν,1),(μ,-1),(ν,-1),(μ,1)],Dim = Dim) # Smunu
    push!(loops,loop_lefttop)
    return loops
end
```


## Calculating actions
We can calculate actions from this packages with fixed gauge fields U. 
We introduce the concenpt "Scalar-valued neural network", which is S(U) -> V, where U and V are gauge fields. 


```julia
using Gaugefields
using LinearAlgebra
function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    Dim = 4
    NC = 3
    flux=[1,0,0,0,0,1]

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")


    gauge_action = GaugeAction(U,B) #empty network
    plaqloop = make_loops_fromname("plaquette") #This is a plaquette loops. 
    append!(plaqloop,plaqloop') #We need hermitian conjugate loops for making the action real. 
    β = 1 #This is a coefficient.
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    Uout = evaluate_GaugeAction_untraced(gauge_action,U,B)
    println(tr(Uout))
end

test1()
```

The output is 

```
----------------------------------------------
Structure of the actions for Gaugefields
num. of terms: 1
-------------------------------
      1-st term: 
          coefficient: 1.0
      -------------------------
1-st loop
L"$U_{1}(n)U_{2}(n+e_{1})U^{\dagger}_{1}(n+e_{2})U^{\dagger}_{2}(n)$"	
2-nd loop
L"$U_{1}(n)U_{3}(n+e_{1})U^{\dagger}_{1}(n+e_{3})U^{\dagger}_{3}(n)$"	
3-rd loop
L"$U_{1}(n)U_{4}(n+e_{1})U^{\dagger}_{1}(n+e_{4})U^{\dagger}_{4}(n)$"	
4-th loop
L"$U_{2}(n)U_{3}(n+e_{2})U^{\dagger}_{2}(n+e_{3})U^{\dagger}_{3}(n)$"	
5-th loop
L"$U_{2}(n)U_{4}(n+e_{2})U^{\dagger}_{2}(n+e_{4})U^{\dagger}_{4}(n)$"	
6-th loop
L"$U_{3}(n)U_{4}(n+e_{3})U^{\dagger}_{3}(n+e_{4})U^{\dagger}_{4}(n)$"	
7-th loop
L"$U_{2}(n)U_{1}(n+e_{2})U^{\dagger}_{2}(n+e_{1})U^{\dagger}_{1}(n)$"	
8-th loop
L"$U_{3}(n)U_{1}(n+e_{3})U^{\dagger}_{3}(n+e_{1})U^{\dagger}_{1}(n)$"	
9-th loop
L"$U_{4}(n)U_{1}(n+e_{4})U^{\dagger}_{4}(n+e_{1})U^{\dagger}_{1}(n)$"	
10-th loop
L"$U_{3}(n)U_{2}(n+e_{3})U^{\dagger}_{3}(n+e_{2})U^{\dagger}_{2}(n)$"	
11-th loop
L"$U_{4}(n)U_{2}(n+e_{4})U^{\dagger}_{4}(n+e_{2})U^{\dagger}_{2}(n)$"	
12-th loop
L"$U_{4}(n)U_{3}(n+e_{4})U^{\dagger}_{4}(n+e_{3})U^{\dagger}_{3}(n)$"	
      -------------------------
----------------------------------------------
8928.0 + 0.0im

```

## Fractional topological charge and so on
We can calculate the topological charge and energy density by using gradient flow as
```julia
temps = Temporalfields(U[1], num=9)

U_copy = similar(U)
B_copy = similar(B)
temp_UμνTA= Matrix{typeof(U[1])}(undef,Dim,Dim)
# for calc energy density
W_temp = Matrix{typeof(U[1])}(undef,Dim,Dim)
for μ=1:Dim
    for ν=1:Dim
        W_temp[μ,ν] = similar(U[1])
    end
end

calc_Q_gradflow!(U_copy,U,temp_UμνTA,W_temp,temps,conditions=["Qclover","Qimproved","Eclover","Energydensity"])
```
or
```julia
calc_Q_gradflow!(U_copy,B_copy,U,B,temp_UμνTA,W_temp,temps,conditions=["Qclover","Qimproved","Eclover","Energydensity"])
```
Then,
```julia
Flowtime 1.0
Qclover:       0.1591786559310214 - 0.0im
Qimproved:     0.17536509762551222 + 0.0im
Eclover:       0.09954804832666195 - 0.0im
Energydensity: 0.09954804832666195
```
Conditions are "Qplaq", "Qclover", "Qimproved", "Eplaq", "Eclover", "Energydensity".

# Appendix: How to calculate derivatives
We can easily calculate the matrix derivative of the actions. The matrix derivative is defined as 

```math
\frac{\partial S}{\partial U_{\mu}(n)}]_{ij} = \frac{\partial S}{\partial U_{\mu,ji}(n)}
```

<!--<img src="https://latex.codecogs.com/svg.image?[\frac{\partial&space;S}{\partial&space;U_{\mu}(n)}]_{ij}&space;=&space;\frac{\partial&space;S}{\partial&space;U_{\mu,ji}(n)}" title="[\frac{\partial S}{\partial U_{\mu}(n)}]_{ij} = \frac{\partial S}{\partial U_{\mu,ji}(n)}" />-->


We can calculate this like 

```julia
dSdUμ = calc_dSdUμ(gauge_action,μ,U,B)
```

or

```julia
calc_dSdUμ!(dSdUμ,gauge_action,μ,U,B)
```

## Hybrid Monte Carlo

With the use of the matrix derivative, we can do the Hybrid Monte Carlo method. 
The simple code is as follows. 

```julia
using Gaugefields
using LinearAlgebra

function MDtest!(gauge_action,U,B,Dim)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 50
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end
```

We define the functions as 

```julia

function calc_action(gauge_action,U,B,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U,B)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,B,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Snew = calc_action(gauge_action,U,B,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,B,p,ϵ,Δτ,Dim,gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp  = gauge_action._temp_U[end]
    dSdUμ = similar(U[1])
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U,B)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
    end
end
```

Then, we can do the HMC: 

```julia
function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    Dim = 4
    NC = 3
    flux=[1,0,0,0,0,1]

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NY,condition = "tflux")


    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop') # add hermitian conjugate
    β = 5.7/2 # real part; re[p] = (p+p')/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    MDtest!(gauge_action,U,B,Dim)

end

test1()
```

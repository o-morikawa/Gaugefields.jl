using Gaugefields
using MPI
using LinearAlgebra

npe_arg = [1 1 1 1]
npe_arg[1] = parse(Int,　ARGS[1]) # npe_x
npe_arg[2] = parse(Int,　ARGS[2]) # npe_y
npe_arg[3] = parse(Int,　ARGS[3]) # npe_z
npe_arg[4] = parse(Int,　ARGS[4]) # npe_t

function MDtest!(snet,U,Dim,mpi=false)
    p = initialize_TA_Gaugefields(U)
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 100

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    numaccepted = 0

    numtrj = 100
    for itrj = 1:numtrj
        @time accepted = MDstep!(snet,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temps)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(snet,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(snet,U)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(snet,U,p,MDsteps,Dim,Uold)
    Δτ = 1/MDsteps
    gauss_distribution!(p)
    Sold = calc_action(snet,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,snet)

        P_update!(U,p,1.0,Δτ,Dim,snet)

        U_update!(U,p,0.5,Δτ,Dim,snet)

    end
    Snew = calc_action(snet,U,p)
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(Snew-Sold))
    if rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end


function test1()
    NX = 24
    NY = 24
    NZ = 24
    NT = 48
    Nwing = 1
    Dim = 4
    NC = 3

    mpi = true
    #mpi = false
    if mpi
        println("mpi=$mpi npe=$(npe_arg)")
    else
	println("mpi=$mpi")
    end		  
    if mpi
        PEs = npe_arg # (1,1,1,2)

        u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
        U = Array{typeof(u1),1}(undef,Dim)
        U[1] = u1
        for μ=2:Dim
            U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT,mpi=true,PEs = PEs,mpiinit = false)
        end
    else

        u1 = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
        U = Array{typeof(u1),1}(undef,Dim)
        U[1] = u1
        for μ=2:Dim
            U[μ] = IdentityGauges(NC,Nwing,NX,NY,NZ,NT)
        end
    end


    snet = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(snet,β,plaqloop)
    
    #show(snet)


    MDtest!(snet,U,Dim,mpi)

end


test1()

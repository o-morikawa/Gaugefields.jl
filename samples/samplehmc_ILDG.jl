using Gaugefields
using LinearAlgebra
using Random

function MDtest!(gauge_action,U,Dim)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 100

    temps = Temporalfields(U[1], num=2)
    comb, factor = set_comb(U, Dim)

    numaccepted = 0

    Random.seed!(123)

    numtrj = 100
    for itrj = 1:numtrj
        accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temps)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold)
    Δτ = 1/MDsteps
    NC,_,NN... = size(U[1])
    for μ=1:Dim
        pwork = gauss_distribution(prod(NN)*(NC^2-1))
        substitute_U!(p[μ],pwork)
    end
    println(p[1][1,1,1,1,1,1])

    #gauss_distribution!(p)
    Sold = calc_action(gauge_action,U,p)
    substitute_U!(Uold,U)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Snew = calc_action(gauge_action,U,p)
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
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 1
    Dim = 4
    NC = 3

    U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    filename = "./conf_00000008.ildg"
    ildg = ILDG(filename)
    i = 1
    L = [NX,NY,NZ,NT]
    load_gaugefield!(U,i,ildg,L,NC)

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.7/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    MDtest!(gauge_action,U,Dim)

end

function gauss_distribution(nv) 
    variance = 1
    nvh = div(nv,2)
    granf = zeros(Float64,nv)
    for i=1:nvh
        rho = sqrt(-2*log(rand())*variance)
        theta = 2pi*rand()
        granf[i] = rho*cos(theta)
        granf[i+nvh] = rho*sin(theta)
    end
    if 2*nvh == nv
        return granf
    end

    granf[nv] = sqrt(-2*log(rand())*variance) * cos(2pi*rand())
    return granf
end


test1()


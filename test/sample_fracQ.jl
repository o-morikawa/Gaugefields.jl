#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop

function HMC_test_4D(NX,NY,NZ,NT,NC,β)

    Dim = 4
    Nwing = 0

    Random.seed!(123)


    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    #"Reproducible"
#    println(typeof(U))

    temps = Temporalfields(U[1], num=9)

    U_copy = similar(U)
    temp_UμνTA= Matrix{typeof(U[1])}(undef,Dim,Dim)
    # for calc energy density
    W_temp = Matrix{typeof(U[1])}(undef,Dim,Dim)
    for μ=1:Dim
        for ν=1:Dim
            W_temp[μ,ν] = similar(U[1])
        end
    end

    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,temps)*factor
    println("0 plaq_t = $plaq_t")
#    poly = calculate_Polyakov_loop(U,temps) 
#    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    MDsteps = 100
    numaccepted = 0

    numtrj = 50

    for itrj = 1:numtrj
        t = @timed begin
            accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,temps)
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,temps)*factor
            println("$itrj plaq_t = $plaq_t")
#            poly = calculate_Polyakov_loop(U,temps) 
#            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

        if itrj % 10 == 0
            #res1,res2,res3=
            @time calc_Q_gradflow!(U_copy,U,temp_UμνTA,W_temp,
                             temps,
                             conditions=["Qclover","Qimproved","Eclover","Energydensity"])
        end

    end
    return plaq_t,numaccepted/numtrj

end


function HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)

    Dim = 4
    Nwing = 0

    flux = Flux

    println("Flux : ", flux)

    #Random.seed!(123)
    t0 = Dates.DateTime(2024,1,1,16,10,7)
    t  = Dates.now()
    Random.seed!(Dates.value(t-t0))

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
    #B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tloop",tloop_dis=2)

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

    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("0 plaq_t = $plaq_t")
#    poly = calculate_Polyakov_loop(U,temp1,temp2) 
#    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

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
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)

        #plaq_t = calculate_Plaquette(U,B,temps)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
#            poly = calculate_Polyakov_loop(U,temps)
#            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/itrj)
        end

        if itrj % 10 == 0
            #res1,res2,res3=
            @time calc_Q_gradflow!(U_copy,B_copy,U,B,temp_UμνTA,W_temp,
                             temps,
                             conditions=["Qclover","Qimproved","Eclover","Energydensity"])
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
    NC = 2
    Flux = [0,0,1,1,0,0]
    #@time HMC_test_4D(NX,NY,NZ,NT,NC,β)
    @time HMC_test_4D_tHooft(NX,NY,NZ,NT,NC,Flux,β)
end
main()

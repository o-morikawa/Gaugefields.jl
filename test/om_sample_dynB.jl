#using LatticeQCD

using Random
using Gaugefields
using LinearAlgebra
#using Wilsonloop

function calc_action(gauge_action,U,B,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U,B)/NC
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end

function MDstep!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,temp1,temp2)
    Δτ = 1.0/MDsteps
    gauss_distribution!(p)

    Sold = calc_action(gauge_action,U,B,p)

    substitute_U!(Uold,U)
    substitute_U!(Bold,B)
    flux_old[:] = flux[:]

    Flux_update!(B,flux)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,B,p,1.0,Δτ,Dim,gauge_action,temp1,temp2)

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end

    Snew = calc_action(gauge_action,U,B,p)
#    println("Sold = $Sold, Snew = $Snew")
#    println("Snew - Sold = $(Snew-Sold)")
    ratio = min(1,exp(-Snew+Sold))
    if rand() > ratio
        println("rejected! flux = ", flux_old)
        substitute_U!(U,Uold)
        substitute_U!(B,Bold)
        flux[:] = flux_old[:]
        return false
    else
        println("accepted! flux_old = ", flux_old, " -> flux_new = ", flux)
        return true
    end
end

function Flux_update!(B,flux)

    NC  = B[1,2].NC
    NDW = B[1,2].NDW
    NX  = B[1,2].NX
    NY  = B[1,2].NY
    NZ  = B[1,2].NZ
    NT  = B[1,2].NT

    i = rand(1:6)
    flux[i] += rand(-1:1)
    flux[i] %= NC
    flux[i] += (flux[i] < 0) ? NC : 0
#    flux = rand(0:NC-1,6)
    B = Initialize_Bfields(NC,flux,NDW,NX,NY,NZ,NT,condition = "tflux")

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

function P_update!(U,B,p,ϵ,Δτ,Dim,gauge_action,temp1,temp2) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temp  = temp1
    dSdUμ = temp2
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U,B)
        mul!(temp,U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temp)
    end
end

function HMC_test_4D_dynamicalB(NX,NY,NZ,NT,NC,β)

    Dim = 4
    Nwing = 0

    #Flux=rand(0:NC-1,6)
    #println("Flux : ", Flux)

    strtrj = 4000
    flux = [1,1,1,1,2,0]

    #println("Flux : ", flux)

    Random.seed!(123)

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    L = [NX,NY,NZ,NT]
    filename = "confs/U_beta6.0_L8_F$(flux[1])$(flux[2])$(flux[3])$(flux[4])$(flux[5])$(flux[6])_$strtrj.txt"
    load_BridgeText!(filename,U,L,NC)
    println("Load file: ", filename)

    temp1 = similar(U[1])
    temp2 = similar(U[1])

    if Dim == 4
        comb = 6 #4*3/2
    elseif Dim == 3
        comb = 3
    elseif Dim == 2
        comb = 1
    else
        error("dimension $Dim is not supported")
    end

    factor = 1/(comb*U[1].NV*U[1].NC)

    @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
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
    substitute_U!(Uold,U)
    Bold = similar(B)
    substitute_U!(Bold,B)
    flux_old = zeros(Int, 6)

    MDsteps = 50
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0

    numtrj = 2000 + strtrj

    for itrj = (strtrj+1):numtrj

        t = @timed begin
#            accepted = MDstep!(gauge_action,U,B,p,MDsteps,Dim,Uold,temp1,temp2)
            accepted = MDstep!(gauge_action,U,B,flux,p,MDsteps,Dim,Uold,Bold,flux_old,temp1,temp2)
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)
#        println("accepted : ", accepted)

        #plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temp1,temp2)*factor
            println("$itrj plaq_t = $plaq_t")
#            poly = calculate_Polyakov_loop(U,temp1,temp2) 
#            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/(itrj-strtrj))
        end

        if itrj % 100 == 0
            filename = "confs/U_beta$(2β)_L$(NX)_F$(flux[1])$(flux[2])$(flux[3])$(flux[4])$(flux[5])$(flux[6])_$itrj.txt"
            save_textdata(U,filename)
            println("Save conf: itrj=", itrj)
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




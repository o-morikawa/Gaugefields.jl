#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
#using Wilsonloop

#import Base.read
import Base.run


function make_filenamehead(NX,NY,NZ,NT,NC,β)
    if NC == 2
        b = "beta$(β)_"
    else
        b = "SU$(NC)_beta$(β)_"
    end

    if NX != NY
        l = "L$(NX)x$(NY)x$(NZ)x$(NT)"
    elseif NX != NZ
        if NZ != NT
            l = "L$(NX)x$(NY)x$(NZ)x$(NT)"
        else
            l = "L$(NX)xL$(NZ)"
        end
    elseif NX != NT
        l = "L$(NX)x$(NT)"
    else
        l = "L$(NX)"
    end

    return "confs/U_"*b*l*"_", "./conf_name/U_"*b*l*".txt", "./conf_name/flux_"*b*l*".txt"
end

function initial_confname(fh) ## import Base.run
    f_head, f_conf, f_flux = fh
    filename = ""
    if !isdir("confs")
        Base.run(`mkdir confs`)
        Base.run(`mkdir conf_name`)
        isInitial = true
    elseif !isdir("conf_name")
        Base.run(`mkdir conf_name`)
        isInitial = true
    elseif !isfile(f_conf)
        touchcmd = `touch $(f_conf[3:end])`
        Base.run(touchcmd)
        isInitial = true
    else
        open(f_conf, "r") do f
            filename *= readline(f)
        end
        if length(filename)==0
            isInitial = true
        else
            isInitial = false
        end
    end

    if !isfile(f_flux)
        touchcmd = `touch $(f_flux[3:end])`
        Base.run(touchcmd)
    end

    return filename, isInitial
end

function save_FluxConf(U,flux,itrj,fh)
    f_head, f_conf, f_flux = fh
    filename = f_head*"F$(flux[1])$(flux[2])$(flux[3])$(flux[4])$(flux[5])$(flux[6])_$itrj.txt"
    save_textdata(U,filename)
    open(f_flux, "a") do f
        write(f, filename * "\n")
    end
    open(f_conf, "w") do f
        write(f, filename)
    end
end

function show_Ploop(U,B,factor,temps,strtrj; condition="plaq")
    if condition=="all"
        @time plaq_t = calculate_Plaquette(U,B,temps)*factor
        println("$strtrj plaq_t = $plaq_t")
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")
    elseif condition=="plaq"
        @time plaq_t = calculate_Plaquette(U,B,temps)*factor
        println("$strtrj plaq_t = $plaq_t")
    elseif condition=="poly"
        poly = calculate_Polyakov_loop(U,temps) 
        println("0 polyakov loop = $(real(poly)) $(imag(poly))")
    elseif condition=="non"
        return
    end
end

function HMC_test_4D_dynamicalB(
    NX,
    NY,
    NZ,
    NT,
    NC,
    β;
    num_τ=2000,
    save_step=100,
    tau_total=1.0,
    show_Pcond="plaq", # "all", "poly", "non"
    show_step=10,
)

    Dim = 4
    Nwing = 0

    #Random.seed!(123)
    t0 = Dates.DateTime(2024,1,1,16,10,7)
    t  = Dates.now()
    Random.seed!(Dates.value(t-t0))

    flux = zeros(Int, 6)
    strtrj = 0

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")

    #filename = "U_beta6.0_L8_F111120_4000.txt"
    f_head = make_filenamehead(NX,NY,NZ,NT,NC,β)
    filename, isInitial = initial_confname(f_head)

    if isInitial
        flux = rand(0:NC-1,6)
        println("Flux : ", flux)
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
    else
        idx = findfirst("_F",filename)[2]
        for i = 1:6
            flux[i] = parse(Int,filename[idx+i])
        end
        idy = findfirst(".txt",filename)[1]
        strtrj = parse(Int, filename[idx+8:idy-1])

        println("Flux : ", flux)

        L = [NX,NY,NZ,NT]
        println("Load file: ", filename)
        load_BridgeText!(filename,U,L,NC)
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
    end

    temps = Temporalfields(U[1], num=3)
    comb, factor = set_comb(U, Dim)

    show_Ploop(U,B,factor,temps,strtrj; condition=show_Pcond)

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)
    #show(gauge_action)

    p = initialize_TA_Gaugefields(U)

    Uold = similar(U)
    Bold = similar(B)
    flux_old = zeros(Int, 6)

    MDsteps = 50 # even integer!!!
    numaccepted = 0

    numtrj = num_τ + strtrj

    for itrj = (strtrj+1):numtrj

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
                #temps,
                τ=tau_total,
            )
        end
        #println("elapsed time for MDsteps: $(t.time) [s]")
        numaccepted += ifelse(accepted,1,0)
        #println("accepted : ", accepted)

        if itrj % show_step == 0
            show_Ploop(U,B,factor,temps,itrj; condition=show_Pcond)
            println("acceptance ratio ",numaccepted/(itrj-strtrj))
        end

        if itrj % save_step == 0
            save_FluxConf(U,flux,itrj,f_head)
            println("Save conf: itrj=", itrj)
        end
    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 1.9
    L = 2

    NX = L
    NY = L
    NZ = L
    NT = L
    NC = 2

    HMC_test_4D_dynamicalB(
        NX,
        NY,
        NZ,
        NT,
        NC,
        β,
        num_τ=4000,
        save_step=10,
        tau_total=1.0,
        show_Pcond="plaq", # "all", "poly", "non"
        show_step=10,
    )
end
main()

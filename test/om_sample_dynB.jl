#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
#using Wilsonloop

#import Base.read
import Base.run

function HMC_test_4D_dynamicalB(
    NX,
    NY,
    NZ,
    NT,
    NC,
    β;
    num_τ=2000,
    save_step=100,
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
    filename = ""
    if !isdir("confs")
        Base.run(`mkdir confs`)
        Base.run(`mkdir conf_name`)
        isInitial = true
    elseif !isdir("conf_name")
        Base.run(`mkdir conf_name`)
        isInitial = true
    elseif !isfile("./conf_name/U_beta$(β)_L$(NX).txt")
        Base.run(`touch conf_name/U_beta$(β)_L$(NX).txt`)
        isInitial = true
    else
        open("./conf_name/U_beta$(β)_L$(NX).txt", "r") do f
            filename *= readline(f)
        end
        if length(filename)==0
            isInitial = true
        else
            isInitial = false
        end
    #else
    #    filename = replace(Base.read(pipeline(`find ./confs -iname "U_beta$(β)_L$(NX)_F*_*.txt"`, `xargs ls -t`, `head -n 1`), String), "\n"=>"")
    #    if filename == ""
    #        isInitial = true
    #    else
    #        isInitial = false
    #    end
    end

    if !isfile("./conf_name/flux_beta$(β)_L$(NX).txt")
        Base.run(`touch conf_name/flux_beta$(β)_L$(NX).txt`)
    end

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

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    println("$strtrj plaq_t = $plaq_t")
#    poly = calculate_Polyakov_loop(U,temps) 
#    println("0 polyakov loop = $(real(poly)) $(imag(poly))")

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

    numtrj = num_τ + strtrj

    for itrj = (strtrj+1):numtrj

        t = @timed begin
            accepted = MDstep_dynB!(
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
                #temps
            )
        end
        if get_myrank(U) == 0
#            println("elapsed time for MDsteps: $(t.time) [s]")
        end
        numaccepted += ifelse(accepted,1,0)
#        println("accepted : ", accepted)

        #plaq_t = calculate_Plaquette(U,temps)*factor
        #println("$itrj plaq_t = $plaq_t")
        
        if itrj % 10 == 0
            @time plaq_t = calculate_Plaquette(U,B,temps)*factor
            println("$itrj plaq_t = $plaq_t")
#            poly = calculate_Polyakov_loop(U,temps) 
#            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            println("acceptance ratio ",numaccepted/(itrj-strtrj))
        end

        if itrj % save_step == 0
            filename = "confs/U_beta$(2β)_L$(NX)_F$(flux[1])$(flux[2])$(flux[3])$(flux[4])$(flux[5])$(flux[6])_$itrj.txt"
            save_textdata(U,filename)
            open("./conf_name/flux_beta$(2β)_L$(NX).txt", "a") do f
                write(f, filename * "\n")
            end
            open("./conf_name/U_beta$(2β)_L$(NX).txt", "w") do f
                write(f, filename)
            end
            println("Save conf: itrj=", itrj)
        end
    end
    return plaq_t,numaccepted/numtrj

end



function main()
    β = 3.0
    L = 4

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
        save_step=10
    )

end
main()

#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop
using MPI

#import Base.read
import Base.run

MPI.Init()

########################################################################
if length(ARGS) < 5
    error("USAGE: ","""
    mpirun -np 2 exe.jl 1 1 1 2 true
    """)
end
const comm = MPI.COMM_WORLD
const pes = Tuple(parse.(Int64,ARGS[1:4]))
const mpi = parse(Bool,ARGS[5])
########################################################################

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

    if mpi
        PEs = pes#(1,1,1,2)
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",mpi=true,PEs = PEs,mpiinit = false) 
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux",mpi=true,PEs = PEs,mpiinit = false)
    else
        U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
    end

    #filename = "U_beta6.0_L8_F111120_4000.txt"
    filename = ""
    if !isdir("confs")
        if get_myrank(U)==0
            Base.run(`mkdir confs`)
            Base.run(`mkdir conf_name`)
        end
        isInitial = true
    elseif !isdir("conf_name")
        if get_myrank(U)==0
            Base.run(`mkdir conf_name`)
        end
        isInitial = true
    elseif !isfile("./conf_name/U_beta$(β)_L$(NX).txt")
        if get_myrank(U)==0
            Base.run(`touch conf_name/U_beta$(β)_L$(NX).txt`)
        end
        isInitial = true
    else
        open("./conf_name/U_beta$(β)_L$(NX).txt", "r") do f
            filename *= readline(f)
        end
        isInitial = false
    #else
    #    filename = replace(Base.read(pipeline(`find ./confs -iname "U_beta$(β)_L$(NX)_F*_*.txt"`, `xargs ls -t`, `head -n 1`), String), "\n"=>"")
    #    if filename == ""
    #        isInitial = true
    #    else
    #        isInitial = false
    #    end
    end

    if isInitial
        #flux = rand(0:NC-1,6)
        flux[:] = MPI.bcast(flux[:], 0, MPI.COMM_WORLD)
        if get_myrank(U)==0
            println("Flux : ", flux)
        end
        if mpi
            PEs = pes
            #B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux",mpi=true,PEs = PEs,mpiinit = false)
        else
            #B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
        end
    else
        idx = findfirst("_F",filename)[2]
        for i = 1:6
            flux[i] = parse(Int,filename[idx+i])
        end
        idy = findfirst(".ildg",filename)[1]
        strtrj = parse(Int, filename[idx+8:idy-1])

        if get_myrank(U)==0
            println("Flux : ", flux)
        end

        ildg = ILDG(filename)
        i = 1
        L = [NX,NY,NZ,NT]
        if get_myrank(U)==0
            println("Load file: ", filename)
        end
        load_gaugefield!(U,i,ildg,L,NC)

        if mpi
            PEs = pes
            B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux",mpi=true,PEs = PEs,mpiinit = false)
        else
            B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")
        end
    end

    temps = Temporalfields(U[1], num=3)
    comb, factor = set_comb(U, Dim)

    @time plaq_t = calculate_Plaquette(U,B,temps)*factor
    if get_myrank(U)==0
        println("$strtrj plaq_t = $plaq_t")
    end
#    poly = calculate_Polyakov_loop(U,temps)
#    if get_myrank(U)==0
#        println("0 polyakov loop = $(real(poly)) $(imag(poly))")
#    end

    gauge_action = GaugeAction(U,B)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = β/2
    push!(gauge_action,β,plaqloop)

    #if get_myrank(U)==0
    #    show(gauge_action)
    #end

    p = initialize_TA_Gaugefields(U)

    Uold  = similar(U)
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
                mpi=mpi,
                PEs=pes
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
            if get_myrank(U)==0
                println("$itrj plaq_t = $plaq_t")
            end
            #            poly = calculate_Polyakov_loop(U,temps) 
            #            println("$itrj polyakov loop = $(real(poly)) $(imag(poly))")
            if get_myrank(U)==0
                println("acceptance ratio ",numaccepted/(itrj-strtrj))
            end
        end

        if itrj % save_step == 0
            filename = "confs/U_beta$(2β)_L$(NX)_mpi$(get_myrank(U))_F$(flux[1])$(flux[2])$(flux[3])$(flux[4])$(flux[5])$(flux[6])_$itrj.ildg"
            if get_myrank(U)==0
                println("Save file: ", filename)
            end
            save_binarydata(U,filename)
            if get_myrank(U)==0
                open("./conf_name/U_beta$(2β)_L$(NX).txt", "w") do f
                    write(f, filename)
                end
                println("Save conf: itrj=", itrj)
            end
        end
    end
    return plaq_t,numaccepted/numtrj

end


function main()
    β = 3.0
    L = 12

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

MPI.Barrier(comm)

MPI.Finalize()

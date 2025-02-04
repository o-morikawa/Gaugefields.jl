#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop

using Glob

import Base.run

function make_filenamehead(NX,NY,NZ,NT,NC,β;obsname="Q")
    if NC == 2
        b = "beta$(β)_"
        bo = obsname*"_beta$(β)_"
    else
        b = "SU$(NC)_beta$(β)_"
        bo = obsname*"_SU$(NC)_beta$(β)_"
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

    return "./confs/U_"*b*l*"_", "./measure/measure_"*bo*l*".txt", "./measure/"*bo*l*".csv"
end

function initial_confnum(fh) ## import Base.run
    f_u, f_ms, f_q = fh
    filename = ""
    if !isdir("measure")
        Base.run(`mkdir measure`)
        isInitial = true
    end
    if !isfile(f_ms)
        touchcmd = `touch $(f_ms[3:end])`
        Base.run(touchcmd)
        isInitial = true
    else
        open(f_ms, "r") do f
            filename *= readline(f)
        end
        if length(filename)==0
            isInitial = true
        else
            isInitial = false
        end
    end

    if !isfile(f_q)
        touchcmd = `touch $(f_q[3:end])`
        Base.run(touchcmd)
    end

    if isInitial
        endtrj = 0
    else
        idx = findfirst("_F",filename)[2]
        idy = findfirst(".txt",filename)[1]
        endtrj = parse(Int, filename[idx+8:idy-1])
    end
    return endtrj
end

function get_confname(itrj,fh)
    f_u, f_ms, f_q = fh
    pattern = f_u*"*_$(itrj).txt"
    files = glob(pattern)
    for file in files
        return file
    end
    return nothing
end

function save_Measurement(filename,Q,itrj,fh)
    f_u, f_ms, f_q = fh
    open(f_q, "a") do f
        if typeof(Q)==Float64
            write(f, "$itrj, $Q\n")
        else
            write(f, "$itrj, $(real(Q)), $(imag(Q))\n")
        end
    end
    open(f_ms, "w") do f
        write(f, filename)
    end
end


function Measurement_test_tHooft(NX,NY,NZ,NT,NC,β)

    Dim = 4
    Nwing = 0

    flux = zeros(Int, 6)

    #Random.seed!(123)
    t0 = Dates.DateTime(2024,1,1,16,10,7)
    t  = Dates.now()
    Random.seed!(Dates.value(t-t0))

    U = Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold",randomnumber="Reproducible")
    B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

    fh_q = make_filenamehead(NX,NY,NZ,NT,NC,β;obsname="Q")
    fh_e = make_filenamehead(NX,NY,NZ,NT,NC,β;obsname="E")
    prev_trj = initial_confnum(fh_q)
    itrj = prev_trj + 10

    filename = get_confname(itrj,fh_q)
    isFile = true
    if filename == nothing
        isFile = false
        println("No files.")
        return
    end

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

    L = [NX,NY,NZ,NT]
    Δt = 0.1
    tmax=Int(ceil((0.7*NX)^2/8.0)/Δt)
    while isFile
        idx = findfirst("_F",filename)[2]
        for i = 1:6
            flux[i] = parse(Int,filename[idx+i])
        end

        println("Flux : ", flux)

        println("Load file: ", filename)
        load_BridgeText!(filename,U,L,NC)
        B = Initialize_Bfields(NC,flux,Nwing,NX,NY,NZ,NT,condition = "tflux")

        res1, res2 = calc_Q_gradflow!(U_copy,B_copy,U,B,temp_UμνTA,W_temp,
                         temps,
                         Δt = Δt,
                         tstep = tmax,
                         meas_step = tmax,
                         displayon = true,
                                      conditions=["Qimproved", "Energydensity"])
        println("Save measurement results")
        save_Measurement(filename,res1[1],itrj,fh_q)
        save_Measurement(filename,res2[1],itrj,fh_e)

        itrj += 10
        filename = get_confname(itrj,fh_q)
        if filename == nothing
            isFile = false
        end
    end
    return
end



function main()
    β = 2.4
    L = 8
    
    NX = L
    NY = L
    NZ = L
    NT = L
    NC = 2
    #@time HMC_test_4D(NX,NY,NZ,NT,NC,β)
    @time Measurement_test_tHooft(NX,NY,NZ,NT,NC,β)
end
main()

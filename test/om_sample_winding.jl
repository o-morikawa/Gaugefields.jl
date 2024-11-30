#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop

using Plots

function UN_test_3D(NX,NY,NT,NC)

    Dim = 3
    L = NX

    n = 20
    println("Test random configuration: n=$n")

    eps = 0.001
    flow_number = 60000
    step = 100
    println("L=$L, eps=$eps,flow=$(eps*flow_number)")

    w = zeros(Float64, n, Int(flow_number/step)+1)
    s = zeros(Float64, n, Int(flow_number/step)+1)
    d = zeros(ComplexF64, n, Int(flow_number/step)+1)

    for i = 1:n

        #Random.seed!(123)
        t0 = Dates.DateTime(2024,1,1,16,10,7)
        t  = Dates.now()
        Random.seed!(Dates.value(t-t0))

        if i == 1
            U = Initialize_3D_UN_Gaugefields(
                NC,NX,NY,NT,
                condition = "cold",
                randomnumber="Random"
            )
            println(typeof(U))
        else
            U = Initialize_3D_UN_Gaugefields(
                NC,NX,NY,NT,
                condition = "hot",
                randomnumber="Random"
            )
            println(typeof(U))
        end

        temps = Temporalfields(U, num=9)
        println(typeof(temps))

        println(winding_UN_3D(U,temps))

        g = Gradientflow_3D(U, eps=eps)
        flownumber = flow_number

        j = 1
        W = winding_UN_3D(U,temps)
        S = calc_gdgaction_3D(U,temps)
        D = det_unitary(U)
        w[i,j] = W
        s[i,j] = S
        d[i,j] = D[1,1,1]

        for iflow = 1:flownumber
            flow!(U, g)
            if iflow%step==0
                j += 1
                W = winding_UN_3D(U,temps)
                S = calc_gdgaction_3D(U,temps)
                D = det_unitary(U)
                w[i,j] = W
                s[i,j] = S
                d[i,j] = D[1,1,1]
            end
        end
    end

    #println(w)

    flow = 0:(eps*step):(eps*flow_number)
    lt = length(flow)

    open("./wind.csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "w$i, ")
        end
        write(f, "w$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(w[i,it]), ")
            end
            write(f, "$(w[n,it])\n")
        end
    end
    plt = plot(flow, w[1,:])
    for i = 2:n
        plot!(plt, flow, w[i,:])
    end
    savefig("wind.png")
    
    open("./action.csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "s$i, ")
        end
        write(f, "s$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(s[i,it]), ")
            end
            write(f, "$(s[n,it])\n")
        end
    end
    plt = plot(flow, s[1,:])
    for i = 2:n
        plot!(plt, flow, s[i,:])
    end
    savefig("action.png")
    
    open("./det.csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "d$i, ")
        end
        write(f, "d$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(d[i,it]), ")
            end
            write(f, "$(d[n,it])\n")
        end
    end
    plt = plot(flow, real(d[1,:]))
    for i = 2:n
        plot!(plt, flow, real(d[i,:]))
    end
    savefig("det_re.png")
    plt = plot(flow, imag(d[1,:]))
    for i = 2:n
        plot!(plt, flow, imag(d[i,:]))
    end
    savefig("det_im.png")

end

function get_mass(i)
    if i == 1
        m = -1
    elseif i == 2
        m = 1
    elseif i == 3
        m = 3
    elseif i == 4
        m = 5
    elseif i == 5
        m = 7
    else
        error("Not supported for get_mass($i).")
    end
    return m
end

function UN_map_3D(NX,NY,NT,NC)

    Dim = 3
    L = NX

    println("Test mapping configuration")
    n = 3

    #=
    eps = 0.001
    flow_number = 40000
    step = 100
    =#
    eps = 0.01
    if L==10
        flow_number = 4000
    elseif L==20
        flow_number = 8000
    elseif L==30
        flow_number = 16000
    elseif L==40
        flow_number = 27000
    elseif L==50
        flow_number = 42000
    end
    step = 10

    println("L=$L, eps=$eps,flow=$(eps*flow_number)")

    w = zeros(Float64, n, Int(flow_number/step)+1)
    s = zeros(Float64, n, Int(flow_number/step)+1)
    d = zeros(ComplexF64, n, Int(flow_number/step)+1)

    for i = 1:n

        #Random.seed!(123)
        t0 = Dates.DateTime(2024,1,1,16,10,7)
        t  = Dates.now()
        Random.seed!(Dates.value(t-t0))

        U = Initialize_3D_UN_Gaugefields(
            NC,NX,NY,NT,
            condition = "test_map",
            m = get_mass(i),
        )
        println(typeof(U))

        temps = Temporalfields(U, num=9)
        println(typeof(temps))

        println(winding_UN_3D(U,temps))

        g = Gradientflow_TA_3D(U, eps=eps)
        flownumber = flow_number
 
        j = 1
        W = winding_UN_3D(U,temps)
        S = calc_gdgaction_3D(U,temps)
        D = det_unitary(U)
        w[i,j] = W
        s[i,j] = S
        d[i,j] = D[1,1,1]

        for iflow = 1:flownumber
            flow!(U, g)
            if iflow%step==0
                j += 1
                W = winding_UN_3D(U,temps)
                S = calc_gdgaction_3D(U,temps)
                D = det_unitary(U)
                w[i,j] = W
                s[i,j] = S
                d[i,j] = D[1,1,1]
            end
        end
    end

    #println(w)

    flow = 0:(eps*step):(eps*flow_number)
    lt = length(flow)

    open("./wind_map_L$(L).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "w$i, ")
        end
        write(f, "w$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(w[i,it]), ")
            end
            write(f, "$(w[n,it])\n")
        end
    end
    plt = plot(flow, w[1,:], label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, w[i,:], label="m=$(get_mass(i))")
    end
    savefig("wind_map_L$(L).png")
    
    open("./action_map_L$(L).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "s$i, ")
        end
        write(f, "s$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(s[i,it]), ")
            end
            write(f, "$(s[n,it])\n")
        end
    end
    plt = plot(flow, s[1,:], label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, s[i,:], label="m=$(get_mass(i))")
    end
    savefig("action_map_L$(L).png")
    
    open("./det_map_L$(L).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "d$i, ")
        end
        write(f, "d$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(d[i,it]), ")
            end
            write(f, "$(d[n,it])\n")
        end
    end
    plt = plot(flow, real(d[1,:]), label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, real(d[i,:]), label="m=$(get_mass(i))")
    end
    savefig("det_map_re_L$(L).png")
    plt = plot(flow, imag(d[1,:]), label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, imag(d[i,:]), label="m=$(get_mass(i))")
    end
    savefig("det_map_im_L$(L).png")

end
function UN_map_Random_3D(NX,NY,NT,NC)

    Dim = 3
    L = NX

    println("Test mapping configuration (Random noise)")
    n = 3

    #=
    eps = 0.001
    flow_number = 40000
    step = 100
    =#
    eps = 0.01
    if L==10
        flow_number = 4000
    elseif L==20
        flow_number = 8000
    elseif L==30
        flow_number = 16000
    elseif L==40
        flow_number = 27000
    elseif L==50
        flow_number = 42000
    end
    step = 10

    println("L=$L, eps=$eps,flow=$(eps*flow_number)")

    random_eps = 0.05
    println("Random noise: size=$(random_eps)")
    
    w = zeros(Float64, n, Int(flow_number/step)+1)
    s = zeros(Float64, n, Int(flow_number/step)+1)
    d = zeros(ComplexF64, n, Int(flow_number/step)+1)

    for i = 1:n

        #Random.seed!(123)
        t0 = Dates.DateTime(2024,1,1,16,10,7)
        t  = Dates.now()
        Random.seed!(Dates.value(t-t0))

        U = Initialize_3D_UN_Gaugefields(
            NC,NX,NY,NT,
            condition = "test_map_rand",
            m = get_mass(i),
            randomnumber="Random",
            reps = random_eps,
        )
        println(typeof(U))

        temps = Temporalfields(U, num=9)
        println(typeof(temps))

        println(winding_UN_3D(U,temps))

        g = Gradientflow_TA_3D(U, eps=eps)
        flownumber = flow_number
 
        j = 1
        W = winding_UN_3D(U,temps)
        S = calc_gdgaction_3D(U,temps)
        D = det_unitary(U)
        w[i,j] = W
        s[i,j] = S
        d[i,j] = D[1,1,1]

        for iflow = 1:flownumber
            flow!(U, g)
            if iflow%step==0
                j += 1
                W = winding_UN_3D(U,temps)
                S = calc_gdgaction_3D(U,temps)
                D = det_unitary(U)
                w[i,j] = W
                s[i,j] = S
                d[i,j] = D[1,1,1]
            end
        end
    end

    #println(w)

    flow = 0:(eps*step):(eps*flow_number)
    lt = length(flow)

    open("./wind_map_L$(L)_Rand$(random_eps).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "w$i, ")
        end
        write(f, "w$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(w[i,it]), ")
            end
            write(f, "$(w[n,it])\n")
        end
    end
    plt = plot(flow, w[1,:], label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, w[i,:], label="m=$(get_mass(i))")
    end
    savefig("wind_map_L$(L)_Rand$(random_eps).png")
    
    open("./action_map_L$(L)_Rand$(random_eps).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "s$i, ")
        end
        write(f, "s$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(s[i,it]), ")
            end
            write(f, "$(s[n,it])\n")
        end
    end
    plt = plot(flow, s[1,:], label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, s[i,:], label="m=$(get_mass(i))")
    end
    savefig("action_map_L$(L)_Rand$(random_eps).png")
    
    open("./det_map_L$(L)_Rand$(random_eps).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "d$i, ")
        end
        write(f, "d$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(d[i,it]), ")
            end
            write(f, "$(d[n,it])\n")
        end
    end
    plt = plot(flow, real(d[1,:]), label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, real(d[i,:]), label="m=$(get_mass(i))")
    end
    savefig("det_map_re_L$(L)_Rand$(random_eps).png")
    plt = plot(flow, imag(d[1,:]), label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, imag(d[i,:]), label="m=$(get_mass(i))")
    end
    savefig("det_map_im_L$(L)_Rand$(random_eps).png")

end
function UN_map_eta_3D(NX,NY,NT,NC)

    Dim = 3
    L = NX

    println("Test mapping configuration (improved action)")
    n = 3

    #=
    eps = 0.001
    flow_number = 40000
    step = 100
    =#
    eps = 0.01
    if L==10
        flow_number = 4000
    elseif L==20
        flow_number = 8000
    elseif L==30
        flow_number = 16000
    elseif L==40
        flow_number = 27000
    elseif L==50
        flow_number = 42000
    end
    step = 10

    println("L=$L, eps=$eps,flow=$(eps*flow_number)")

    eta = -10
    println("eta: $eta")
    
    w = zeros(Float64, n, Int(flow_number/step)+1)
    s = zeros(Float64, n, Int(flow_number/step)+1)
    d = zeros(ComplexF64, n, Int(flow_number/step)+1)

    for i = 1:n

        #Random.seed!(123)
        t0 = Dates.DateTime(2024,1,1,16,10,7)
        t  = Dates.now()
        Random.seed!(Dates.value(t-t0))

        U = Initialize_3D_UN_Gaugefields(
            NC,NX,NY,NT,
            condition = "test_map",
            m = get_mass(i),
        )
        println(typeof(U))

        temps = Temporalfields(U, num=9)
        println(typeof(temps))

        println(winding_UN_3D(U,temps))

        g = Gradientflow_TA_eta_3D(U, eta, eps=eps)
        flownumber = flow_number

        j = 1
        W = winding_UN_3D(U,temps)
        S = calc_gdgaction_3D(U,eta,temps)
        D = det_unitary(U)
        w[i,j] = W
        s[i,j] = S
        d[i,j] = D[1,1,1]

        for iflow = 1:flownumber
            flow!(U, g)
            if iflow%step==0
                j += 1
                W = winding_UN_3D(U,temps)
                S = calc_gdgaction_3D(U,eta,temps)
                D = det_unitary(U)
                w[i,j] = W
                s[i,j] = S
                d[i,j] = D[1,1,1]
            end
        end
    end

    #println(w)

    flow = 0:(eps*step):(eps*flow_number)
    lt = length(flow)

    open("./wind_map_L$(L)_eta$(eta).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "w$i, ")
        end
        write(f, "w$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(w[i,it]), ")
            end
            write(f, "$(w[n,it])\n")
        end
    end
    plt = plot(flow, w[1,:], label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, w[i,:], label="m=$(get_mass(i))")
    end
    savefig("wind_map_L$(L)_eta$(eta).png")
    
    open("./action_map_L$(L)_eta$(eta).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "s$i, ")
        end
        write(f, "s$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(s[i,it]), ")
            end
            write(f, "$(s[n,it])\n")
        end
    end
    plt = plot(flow, s[1,:], label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, s[i,:], label="m=$(get_mass(i))")
    end
    savefig("action_map_L$(L)_eta$(eta).png")
    
    open("./det_map_L$(L)_eta$(eta).csv", "w") do f
        write(f, "flowtime, ")
        for i = 1:(n-1)
            write(f, "d$i, ")
        end
        write(f, "d$n\n")
        for it = 1:lt
            write(f, "$(flow[it]), ")
            for i = 1:(n-1)
                write(f, "$(d[i,it]), ")
            end
            write(f, "$(d[n,it])\n")
        end
    end
    plt = plot(flow, real(d[1,:]), label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, real(d[i,:]), label="m=$(get_mass(i))")
    end
    savefig("det_map_re_L$(L)_eta$(eta).png")
    plt = plot(flow, imag(d[1,:]), label="m=$(get_mass(1))")
    for i = 2:n
        plot!(plt, flow, imag(d[i,:]), label="m=$(get_mass(i))")
    end
    savefig("det_map_im_L$(L)_eta$(eta).png")

end


function main()
    L = 10
    
    NX = L
    NY = L
    NT = L
    NC = 2
    #@time UN_test_3D(NX,NY,NT,NC)
    #@time UN_map_3D(NX,NY,NT,NC)
    #@time UN_map_Random_3D(NX,NY,NT,NC)
    @time UN_map_eta_3D(NX,NY,NT,NC)
end
main()

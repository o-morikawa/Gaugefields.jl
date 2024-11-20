#using LatticeQCD

using Random
using Dates
using Gaugefields
using LinearAlgebra
using Wilsonloop

using Plots

function calc_action_3D(U,temps_g)
    NV = U.NV

    temps, it_temps = get_temp(temps_g,2)
    temp1 = temps[1]
    temp2 = temps[2]

    S = 0.0
    substitute_U!(temp1, U)
    for μ=1:3
        temp2 = calculate_gdg(temp1, μ, cc=false)
        S += (-1/2) * tr(temp2, temp2)

        temp2 = calculate_gdg(temp1, μ, cc=true)
        S += (-1/2) * tr(temp2, temp2)
    end
    unused!(temps_g,it_temps)
    return real(S) / NV
end

function UN_test_3D(NX,NY,NT,NC,β)

    Dim = 3

    n = 5

    eps = 0.001
    flow_number = 10000
    step = 100
    
    w = zeros(Float64, n, Int(flow_number/step))
    s = zeros(Float64, n, Int(flow_number/step))

    for i = 1:n

        #Random.seed!(123)
        t0 = Dates.DateTime(2024,1,1,16,10,7)
        t  = Dates.now()
        Random.seed!(Dates.value(t-t0))

        if i == 0
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

        g = Gradientflow_3D(U, eps=eps)
        flownumber = flow_number
        j = 0
        for iflow = 1:flownumber
            flow!(U, g)
            if iflow%step==0
                j += 1
                W = winding_UN_3D(U,temps)
                S = calc_action_3D(U,temps)
                w[i,j] = W
                s[i,j] = S
            end
        end
    end

    #println(w)
    
    flow = (eps*step):(eps*step):(eps*flow_number)
    plt = plot(flow, w[1,:])
    for i = 2:n
        plot!(plt, flow, w[i,:])
    end
    savefig("wind.png")
    
    plt = plot(flow, s[1,:])
    for i = 2:n
        plot!(plt, flow, s[i,:])
    end
    savefig("action.png")

end


function main()
    β = 3.0
    NX = 16
    NY = 16
    NT = 16
    NC = 2
    @time UN_test_3D(NX,NY,NT,NC,β)
end
main()

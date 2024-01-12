using LinearAlgebra
# We use "stacked" signals to align with the Hankel formulation
# That is, if xₜ ∈ ℝⁿ then we typically use a column vector for a trajectory [x₀' ... xₜ₋₁']'

function H_matrix_original(u, L)
    H = zeros(L, length(u)-L+1)
    for i = 1:L
        H[i,:]=u[i:length(u)-L+i]
    end
    return H
end

function H_matrix(u::VecOrMat, L::Int)
    N = length(u)
    H = vcat([u[i:N-L+i]' for i in 1:L]...)
    return H
    # H = zeros(L, length(u)-L+1)
    # for i = 1:L
    #     H[i,:]=u[i:length(u)-L+i]
    # end
    # return H
end

function SS_signal(A::Matrix, B::Matrix, C::Matrix, D::Matrix)
    # Takes state-space matrices and returns an input/output response used for a Hankel representation
    @assert size(D) == (1,1)
    n = size(A,1)
    L = n + 1
    N = 4n+1

    input = vcat(ones(L+n), zeros(2n))
    A_powers = [A^k for k in 0:N-2]

    # output = Vector{Float64}(undef, N)
    output = vcat(D[1], [dot(C*sum(A_powers[1:k]),B) + D[1] for k in 1:L+n-1], [dot(C*sum(A_powers[k+1-L-n:k]),B) for k in L+n:N-1])
    # output[1] = D[1]
    # output[2:L+n] = [dot(C*sum(A_powers[1:k]),B) + D[1] for k in 1:L+n-1]
    # output[L+n+1:end] = [dot(C*sum(A_powers[k+1-L-n:k]),B) for k in L+n:N-1]
    return input, output
end

function PE_signal(L, num_data=100; m=1)
    # num_data = max(2*(m+1)*L + 1, num_data)
    num_data = max((m+1)*L, num_data)
    u = randn(num_data)
    # H = H_matrix(u,L)
    # while rank(H) != L
    #     u = randn(num_data)
    #     H = H_matrix(u,L)
    # end
    # return u, H
end

function is_trajectory(u, y, H_u, H_y,p=1)
    alpha = lsmr([H_u; H_y], [u; y], λ = 0)
    return norm([H_u; H_y]*alpha - [u; y], p)^p
end

function complete_prediction(u, y_init, H_u, H_y, H_y_nu)
    # alpha = [H_u; H_y_nu]\[u; y_init]
    # E = [H_u; H_y_nu]
    # A = [H_u[2:end,:]; reshape(-0.1*H_y[end,:], (1, length(H_y[end,:]))); H_y_nu[2:end,:]; reshape(H_y[end,:], (1, length(H_y[end,:])))]

    # println(norm.(eigvals(E'*A, E'E)))

    # alpha = lsmr([H_u; H_y_nu], [u; y_init], λ = 0.10) # TODO this doesn't return trivial solution when RHS is zero

    # α = qr([H_u; H_y_nu]'*[H_u; H_y_nu], Val(true)) \ [H_u; H_y_nu]'*[u; y_init]
    α = [H_u; H_y_nu]\[u; y_init]
    # α = [H_u; H_y_nu; 0.1*I(size(H_u,2))]\[u; y_init; zeros(size(H_u,2))]
    # y = H_y*alpha
    # println(typeof(H_y[end:end,:]*alpha))
    # println(typeof(α))
    y = *(H_y[end:end,:],α)

end

function get_rand_trajectory(u_init, y_init, L, nu, steps)
    # Takes a random input signal and predicts the system output using Hankel matrices
    # u_init and y_init need to be at least length (order) n to fix a unique trajectory
    # IMPORTANT: This example assumes u_init and y_init are the persistenly exciting data used to generate
    # the Hankel matrices in the first place

    H_u = H_matrix(u_init, L)
    H_y = H_matrix(y_init, L)
    H_y_nu = H_matrix(y_init[1:end-L+nu], nu)

    u = [u_init[1:nu]; randn(div(steps,2),1); ones(div(steps,2), 1)]
    y = y_init[1:nu]

    while length(y) < steps
        y_complete = complete_prediction(u[length(y)-nu+1:length(y)+L-nu], y[end-nu+1:end], H_u, H_y, H_y_nu)
        y = [y; y_complete[nu+1:end]]
        is_traj, = is_trajectory(u[length(y)-L+1:length(y)], y[end-L+1:end], H_u, H_y)
        # println(is_traj)
    end

    u = u[1:length(y)]
    return u, y
end

function get_closedloop_trajectory(pe_u, pe_y, u_init, y_init, r, L, steps, x_C=nothing, C=nothing)
    # Takes persistently exiciting data and L to generate approxpriate H matrices
    # u_init, y_init include the n steps in closed-loop, from which we simulate
    #   under reference r
    # IMPORTANT: L > n + 1 here
    # NOTE: pe_u, pe_y can be experimental open loop data, we just need to have some process data to intialize the closed loop trajectory

    nu = L - 1
    H_u = H_matrix(pe_u[1:end-L+nu], nu)
    H_y = H_matrix(pe_y[1:end-L+nu], nu)

    H_u_full = H_matrix(pe_u, L)
    H_y_full = H_matrix(pe_y, L)

    u = u_init[1:nu,:]
    y = y_init[1:nu,:]
    while length(y) - length(y_init) < steps
        y_complete = complete_prediction(u[end-nu+1:end], y[end-nu+1:end], H_u, H_y_full, H_y)
        y = [y; y_complete[end]]

        # Select an arbitrary function of y just to get output data
        if C == nothing
            u = [u; -0.01*(r-y[end]) + u[end]]
        else
            if x_C == nothing
                x_C = zeros(size(C.A,1), 1)
            end
            u_next = C.C*x_C + C.D*(r.-y[end])
            x_C = C.A*x_C + C.B*(r.-y[end])

            u = [u; u_next]
        end
    end
    return u, y, x_C
end

function Hankel_YK_controller(pe_u, pe_y, u_traj, y_traj, L, input, states_YK, Q=nothing, X=nothing, Y=nothing, C=nothing, D=nothing, N=nothing)
    # function YK_controller(input, states_YK, X=nothing, Y=nothing, N=nothing, D=nothing, Q=nothing)

        # Get output of Y^-1 dynamics and advance state
        states_YK[1] == nothing ? state_Y = zeros(size(Y.A,1), 1) : state_Y = states_YK[1]
        yinv = Y.C*state_Y + Y.D*input
        state_Y = Y.A*state_Y + Y.B*input

        # Get output of X dynamics and advance state
        states_YK[2] == nothing ? state_X = zeros(size(X.A,1), 1) : state_X = states_YK[2]
        x = X.C*state_X + X.D*yinv
        state_X = X.A*state_X + X.B*yinv

        # Get output of Q dynamics and advance state
        states_YK[3] == nothing ? state_Q = zeros(size(Q.A,1), 1) : state_Q = states_YK[3]
        q = Q.C*state_Q + Q.D*yinv
        state_Q = Q.A*state_Q + Q.B*yinv

        # Use the output of the Q parameter as input to simulate the nominal stable closed loop
        # TODO: What if u_traj and y_traj are updated based on the plant trajectory (not the internal Hankel trajectory)?
        u_Hankel, y_Hankel, state_C = get_closedloop_trajectory(pe_u, pe_y, u_traj, y_traj, q, L+1, 1, states_YK[4], C)
        u_traj = u_Hankel[2:end] # This is the simulation of D
        y_traj = y_Hankel[2:end] # This is the simulation of N

        # A gross way of fixing the one-step mismatch I was previously getting in the N dynamics
        #   Just recreate the H matrices and one-step predict N, but don't update the prediction of D
        #   (that waits until the next step, then repeat)
        H_u = H_matrix(pe_u[1:end-1], L)
        H_y = H_matrix(pe_y[1:end-1], L)
        H_u_full = H_matrix(pe_u, L+1)
        H_y_full = H_matrix(pe_y, L+1)
        y_complete = complete_prediction(u_traj[end-L+1:end], y_traj[end-L+1:end], H_u, H_y_full, H_y)

        d_Hankel = u_traj[end]
        n_Hankel = y_complete[end]

        # Get output of D dynamics and advance state
        #   D can be thought of as the stable closed-loop dynamics of the original controller
        #   where Q produces a 'ficticious' reference signal
        states_YK[5] == nothing ? state_D = zeros(size(D.A,1), 1) : state_D = states_YK[5]
        d_ss = D.C*state_D + D.D*q
        state_D = D.A*state_D + D.B*q

        # Advance the output of N to compare with next system tracking error
        #   N can be thought of as the stable closed-loop dynamics of the system output under the original controller
        #   where Q produces a 'ficticious' reference signal
        states_YK[6] == nothing ? state_N = zeros(size(N.A,1), 1) : state_N = states_YK[6]
        state_N = N.A*state_N + N.B*q
        n_ss = N.C*state_N + N.D*q

        # Produce next control action
        # u = d_ss.+ x
        u = d_Hankel.+ x
        states_YK = [state_Y, state_X, state_Q, state_C, state_D, state_N]

        return u, u_traj, y_traj, states_YK, n_ss, d_ss, n_Hankel, d_Hankel

end



function Hankel_YK_controller_stable(Qinputs, Qoutputs, state_Q=nothing, Q=nothing)
    # function YK_controller(input, states_YK, X=nothing, Y=nothing, N=nothing, D=nothing, Q=nothing)



        input_vec, u_vec = SS_signal(Q.A, Q.B, Q.C, Q.D)
        n_q = size(Q.A,1)
        H_input = H_matrix(input_vec, n_q+1)
        H_u = H_matrix(u_vec[1:end-1], n_q)
        H_u_full = H_matrix(u_vec, n_q+1)

        u = complete_prediction(Qinputs, Qoutputs, H_input, H_u_full, H_u)



        # u_traj = [u_traj[2:end]; u]

        # return u, state_Q
        return u


end

function YK_controller(input, states_YK, X=nothing, Y=nothing, N=nothing, D=nothing, Q=nothing)

    # Get output of Y^-1 dynamics and advance state
    states_YK[1] == nothing ? state_Y = zeros(size(Y.A,1), 1) : state_Y = states_YK[1]
    yinv = Y.C*state_Y + Y.D*input
    state_Y = Y.A*state_Y + Y.B*input

    # Get output of X dynamics and advance state
    states_YK[2] == nothing ? state_X = zeros(size(X.A,1), 1) : state_X = states_YK[2]
    x = X.C*state_X + X.D*yinv
    state_X = X.A*state_X + X.B*yinv

    # Get output of Q dynamics and advance state
    states_YK[3] == nothing ? state_Q = zeros(size(Q.A,1), 1) : state_Q = states_YK[3]
    q = Q.C*state_Q + Q.D*yinv
    state_Q = Q.A*state_Q + Q.B*yinv

    # Get output of D dynamics and advance state
    #   D can be thought of as the stable closed-loop dynamics of the original controller
    #   where Q produces a 'ficticious' reference signal
    states_YK[4] == nothing ? state_D = zeros(size(D.A,1), 1) : state_D = states_YK[4]
    d = D.C*state_D + D.D*q
    state_D = D.A*state_D + D.B*q

    # Produce next control action
    u = d.+ x

    # Advance the output of N to compare with next system tracking error
    #   N can be thought of as the stable closed-loop dynamics of the system output under the original controller
    #   where Q produces a 'ficticious' reference signal
    states_YK[5] == nothing ? state_N = zeros(size(N.A,1), 1) : state_N = states_YK[5]
    state_N = N.A*state_N + N.B*q
    n = N.C*state_N + N.D*q

    states_YK = [state_Y, state_X, state_Q, state_D, state_N]

    return u, n, states_YK

end



struct TankDynamicsParams{T}
    dt::T
    sampleInterval::T
    pumpTimeConstant::T
    flowInTimeConstant::T
    flowOutTimeConstant::T
    measurementTimeConstant::T
    pumpDeadTime::T
    flowInDeadTime::T
    levelDeadTime::T
    measurementDeadTime::T
    minPumpSpeed::T
    flowAtMinPumpSpeed::T
    flowAtMaxPumpSpeed::T
    pipeDiameter::T
    flowCoefficient::T
    gravitationalConstant::T
    tankDiameter::T
    tankHeight::T
    pump_delay_steps::Int64
    flow_in_delay_steps::Int64
    level_delay_steps::Int64
    measurement_delay_steps::Int64
    pump_filter_factor::T
    flow_in_filter_factor::T
    flow_out_filter_factor::T
    level_measurement_filter_factor::T
    T_reset::Int64
    kp_reset::T
    ki_reset::T
    kd_reset::T
    Tf_reset::T
    kp_flow::T
    ki_flow::T
    kd_flow::T
    Tf_flow::T
end

function TankDynamicsParams(;
    T=Float64,
    dt=0.5, pumpTimeConstant=0.1/60, flowInTimeConstant=0.1/60, flowOutTimeConstant=0.1/60,
    measurementTimeConstant=0.2/60, pumpDeadTime=0.1/60, flowInDeadTime=0.1/60, levelDeadTime=0.2/60,
    measurementDeadTime=0.2/60, minPumpSpeed=0, flowAtMinPumpSpeed=0, flowAtMaxPumpSpeed=80,
    pipeDiameter=0.25, flowCoefficient=0.61, tankDiameter=2.413, tankHeight=12.192,
    T_reset=500, kp_reset=4.0/2.0, ki_reset=0.067/2.0, kd_reset=0.04, Tf_reset=0.2, kp_flow=0.2, ki_flow=0.067, kd_flow=0.02, Tf_flow=0.1
    )

    sampleInterval = dt/60
    gravitationalConstant = 9.81*10/((1/60)^2)

    pump_delay_steps = ceil(pumpDeadTime / sampleInterval)
    flow_in_delay_steps = ceil(flowInDeadTime / sampleInterval)
    level_delay_steps = ceil(levelDeadTime / sampleInterval)
    measurement_delay_steps = ceil(measurementDeadTime / sampleInterval)

    pump_filter_factor = 1 - exp(-sampleInterval / pumpTimeConstant)
    flow_in_filter_factor = 1 - exp(-sampleInterval / flowInTimeConstant)
    flow_out_filter_factor = 1 - exp(-sampleInterval / flowOutTimeConstant)
    level_measurement_filter_factor = 1 - exp(-sampleInterval / measurementTimeConstant)

    TankDynamicsParams{T}(dt, sampleInterval, pumpTimeConstant, flowInTimeConstant, flowOutTimeConstant,
    measurementTimeConstant, pumpDeadTime, flowInDeadTime, levelDeadTime, measurementDeadTime,
    minPumpSpeed, flowAtMinPumpSpeed, flowAtMaxPumpSpeed, pipeDiameter, flowCoefficient, gravitationalConstant,
    tankDiameter, tankHeight, pump_delay_steps, flow_in_delay_steps, level_delay_steps, measurement_delay_steps,
    pump_filter_factor, flow_in_filter_factor, flow_out_filter_factor, level_measurement_filter_factor,
    T_reset, kp_reset, ki_reset, kd_reset, Tf_reset, kp_flow, ki_flow, kd_flow, Tf_flow
    )

end

mutable struct TankDynamics{T}
    tank_params::TankDynamicsParams{T}
    pump_sp::T
    pump_pv::T
    pump_setpoint_fifo::Vector{T}
    flow_sp::T
    flow_in::T
    flow_out::T
    level::T
    measured_level_state::T
    measured_level::T
    pump_flow_fifo::Vector{T}
    flow_in_fifo::Vector{T}
    flow_out_fifo::Vector{T}
    level_delay_fifo::Vector{T}
    error::T
    delta_error::T
    delta2_error::T
    u_prev::T
    error_prev::T
    delta_error_prev::T
    error_flow::T
    delta_error_flow::T
    delta2_error_flow::T
    u_prev_flow::T
    error_flow_prev::T
    delta_error_flow_prev::T
    flow_in_prev::T
    pump_pv_prev::T
    measured_level_prev::T
    level_sp::T
end

function TankDynamics(;
    T = Float64, pump_sp=0.0, pump_pv=0.0, flow_sp=0.0, flow_in=0.0, flow_out=0.0,
    level=0.0,  error=0.0, delta_error=0.0, delta2_error=0.0, u_prev=0.0, error_flow=0.0, delta_error_flow=0.0,
    delta2_error_flow=0.0, u_prev_flow=0.0, level_sp=50.0
    )
    tank_params=TankDynamicsParams(T=T)

    pump_setpoint_fifo = pump_sp*ones(tank_params.pump_delay_steps)
    measured_level_state = level
    measured_level = level

    pump_flow_fifo = flow_in*ones(tank_params.flow_in_delay_steps)
    flow_in_fifo = flow_in*ones(tank_params.flow_in_delay_steps)
    flow_out_fifo = flow_in*ones(tank_params.level_delay_steps)
    level_delay_fifo = flow_in*ones(tank_params.measurement_delay_steps)

    error_prev = error
    delta_error_prev = delta_error
    error_flow_prev = error_flow
    delta_error_flow_prev = delta_error_flow

    flow_in_prev = flow_in
    pump_pv_prev = pump_pv
    measured_level_prev = measured_level

    TankDynamics{T}(tank_params, pump_sp, pump_pv, pump_setpoint_fifo, flow_sp, flow_in, flow_out, level, measured_level_state,
    measured_level, pump_flow_fifo, flow_in_fifo, flow_out_fifo, level_delay_fifo, error, delta_error, delta2_error,
    u_prev, error_prev, delta_error_prev, error_flow, delta_error_flow, delta2_error_flow,
    u_prev_flow, error_flow_prev, delta_error_flow_prev, flow_in_prev, pump_pv_prev, measured_level_prev, level_sp)
end


function reset(env::TankDynamics{T}; level_sp=10.0) where {T}
    
    env.level_sp = level_sp
    step_response(env, steps=500)

    nothing
end

function PID(env::TankDynamics{T}; mode="flow", return_delta=false) where {T}

    if mode == "flow"
        env.error_flow = env.flow_sp - env.flow_in
        env.delta_error_flow = env.tank_params.Tf_flow*env.delta_error_flow + (1-env.tank_params.Tf_flow)*(env.flow_in - env.flow_in_prev) / env.tank_params.dt
        env.delta2_error_flow = (env.delta_error_flow - env.delta_error_flow_prev) / env.tank_params.dt
        env.u_prev_flow = env.pump_sp

        # env.error_flow_int += env.dt*env.error_flow
        # pump_sp = env.kp_flow*(env.error_flow) + env.ki_flow*env.error_flow_int

        pump_delta = env.tank_params.kp_flow*(env.error_flow - env.error_flow_prev) + env.tank_params.ki_flow*env.error_flow*env.tank_params.dt - env.tank_params.kd_flow*env.delta2_error_flow
        env.pump_sp = min(max(pump_delta + env.u_prev_flow,0.0),100.0)

        env.error_flow_prev = env.error_flow
        env.delta_error_flow_prev = env.delta_error_flow
        if return_delta
            return pump_delta
        else
            return env.pump_sp
        end
    else
        env.error = env.level_sp - env.measured_level
        env.delta_error = env.tank_params.Tf_reset*env.delta_error + (1 - env.tank_params.Tf_reset)*(env.measured_level - env.measured_level_prev) / (env.tank_params.dt)
        env.delta2_error = (env.delta_error - env.delta_error_prev) / env.tank_params.dt
        env.u_prev = env.flow_sp

        flow_delta = env.tank_params.kp_reset*(env.error - env.error_prev) + env.tank_params.ki_reset*env.error*env.tank_params.dt - env.tank_params.kd_reset*env.delta2_error
        env.flow_sp = min(max(flow_delta + env.u_prev,0.0),100.0)

        env.error_prev = env.error
        env.delta_error_prev = env.delta_error

        if return_delta
            return flow_delta
        else
            return env.flow_sp
        end
    end
end

function (env::TankDynamics{T})() where {T}

    env.pump_setpoint_fifo[2:end] = env.pump_setpoint_fifo[1:end-1]
    env.pump_setpoint_fifo[1] = env.pump_sp

    env.pump_pv = (1 - env.tank_params.pump_filter_factor) * env.pump_pv +
                    env.tank_params.pump_filter_factor * env.pump_setpoint_fifo[end]

    env.pump_flow_fifo[2:end] = env.pump_flow_fifo[1:end-1]
    env.pump_flow_fifo[1] = env.pump_pv

    pump_flow_equation = 0
    pump_flow_equation = max(0, env.tank_params.flowAtMinPumpSpeed + (
            (env.tank_params.flowAtMaxPumpSpeed - env.tank_params.flowAtMinPumpSpeed)/(100-env.tank_params.minPumpSpeed)) *
                         (env.pump_flow_fifo[end] - env.tank_params.minPumpSpeed))

    flow_in_noise = 0.3*randn() # Revisit this noise term after seeing the real system
    env.flow_in_prev = env.flow_in
    env.flow_in = (1 - env.tank_params.flow_in_filter_factor) * (env.flow_in+flow_in_noise) + (env.tank_params.flow_in_filter_factor * pump_flow_equation)
    env.flow_out = (1 - env.tank_params.flow_out_filter_factor) * env.flow_out + (env.tank_params.flow_out_filter_factor * π * ((env.tank_params.pipeDiameter/2)^2) *
                                                          env.tank_params.flowCoefficient * ((2*env.tank_params.gravitationalConstant*env.level)^0.5))

    env.flow_in_fifo[2:end] = env.flow_in_fifo[1:end-1]
    env.flow_in_fifo[1] = env.flow_in

    env.flow_out_fifo[2:end] = env.flow_out_fifo[1:end-1]
    env.flow_out_fifo[1] = env.flow_out

    env.level = env.level + ((env.flow_in_fifo[end] - env.flow_out_fifo[end]) /
                     (π*((env.tank_params.tankDiameter/2)^2)) * env.tank_params.sampleInterval)

    env.level = max(0, min(env.tank_params.tankHeight, env.level))

    env.level = float(env.level)

    env.level_delay_fifo[2:end] = env.level_delay_fifo[1:end-1]
    env.level_delay_fifo[1] = env.level

    env.measured_level_state = ((1 - env.tank_params.level_measurement_filter_factor) * env.measured_level_state +
                            env.tank_params.level_measurement_filter_factor * env.level_delay_fifo[end])

    # measurement_noise = 0  # make this something random
    measurement_noise = 0.015*randn()  # make this something random

    env.measured_level_prev = env.measured_level
    env.measured_level = 10.0*max(0.0, min(env.tank_params.tankHeight, env.measured_level_state + measurement_noise))

    return env.measured_level


end

function step(env::TankDynamics{T}, action) where {T}
    env.flow_sp = action
    PID(env)
    env()
end

function step_response(env::TankDynamics{T}; steps::Int64=300, level_control::Bool=true, flow_sp::Float64=0.0) where {T}

    output_vec = []
    input_vec = []
    pump_vec = []
    pump_sp_vec = []

    for _ in 1:steps

        level_control ? PID(env, mode="level") : env.flow_sp = flow_sp
        step(env, env.flow_sp)
        push!(output_vec, env.measured_level)
        push!(input_vec, env.flow_in)
        push!(pump_vec, env.pump_pv)
        push!(pump_sp_vec, env.pump_sp)
    end
    return output_vec, input_vec, pump_vec, pump_sp_vec
end

function tank_sim(env::TankDynamics{T}, input::Vector) where {T}
    output = Float64[]
    flow_in_vec = Float64[]
    for u in input
        push!(flow_in_vec, env.flow_in)
        y = step(env, u)
        push!(output, y)
    end
    return output, flow_in_vec
end

function tank_probe(env::TankDynamics{T}, delta::Vector) where {T}
    input = Float64[]
    output = Float64[]
    for u in delta
        PID(env, mode="level")
        env.flow_sp += u
        PID(env)
        env()
        push!(input, env.flow_sp)
        push!(output, env.measured_level)
    end
    return input, output
end
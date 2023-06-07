mutable struct Wiring
    units::Int
    adjacency_matrix::Matrix{Int}
    sensory_adjacency_matrix::Union{Nothing, Matrix{Int}}
    input_dim::Union{Nothing, Int}
    output_dim::Union{Nothing, Int}
end

function Base.getproperty(w::Wiring, v::Symbol)
    if v == :built
        return !isnothing(w.input_dim)
    elseif v == :synapse_count
        sum(abs.(w.adjacency_matrix))
    elseif v == :sensory_synapse_count
        sum(abs.(w.sensory_adjacency_matrix))
    else
        return getfield(w, v)
    end
end

function get_neurons_of_layer(w::Wiring, layer_id)
    collect(1:w.units)
end


function build!(w::Wiring, input_dim)
    if !(w.input_dim |> isnothing) && w.input_dim != input_dim
        throw(ErrorException(
            "Conflicting dimensions provided. Tried setting wiring.input_dim to $input_dim but wiring.input_dim is already $(w.input_dim)."))
    end
    if w.input_dim |> isnothing
        set_input_dim!(w, input_dim)
    end
end

function erev_initializer(w::Wiring)
    return copy(w.adjacency_matrix)
end

function sensory_erev_initializer(w::Wiring)
    copy(w.sensory_adjacency_matrix)
end

function set_input_dim!(w::Wiring, input_dim)
    w.input_dim = input_dim
    w.sensory_adjacency_matrix = zeros(Int, input_dim, w.units)
end

function set_output_dim!(w::Wiring, output_dim)
    w.output_dim = output_dim
end

function get_type_of_neuron(w::Wiring, neuron_id)
    return neuron_id < w.output_dim ? "motor" : "inter"
end

function add_synapse!(w::Wiring, src, dest, polarity)
    if src < 1 || src > w.units
        throw(ErrorException("Cannot add synapse originating in $src if cell only has $(w.units) units"))
    end
    if dest < 1 || dest > w.units
        throw(ErrorException("Cannot add synapse feeding into $dest if cell only has $(w.units) units"))
    end
    if polarity != 1 && polarity != -1
        throw(ErrorException("Cannot add synapse with polarity $polarity (expected -1 or +1)"))
    end
    w.adjacency_matrix[src, dest] = polarity
end

function add_sensory_synapse!(w::Wiring, src, dest, polarity)
    if w.input_dim |> isnothing
        throw(ErrorException("Cannot add sensory synapses before buil!() has been called"))
    end
    if src < 1 || src > w.input_dim
        throw(ErrorException("Cannot add sensory synapse originating in $src if input has only $(w.input_dim) features"))
    end
    if dest < 1 || dest > w.input_dim
        throw(ErrorException("Cannot add sensory synapse feeding into $dest if input has only $(w.input_dim) features"))
    end
    if polarity != 1 && polarity != -1
        throw(ErrorException("Cannot add sensory synapse with polarity $polarity (expected -1 or +1)"))
    end
    w.sensory_adjacency_matrix[src, dest] = polarity
end

function Wiring(units::Int)
    Wiring(
        units,
        zeros(Int, units, units),
        nothing,
        nothing,
        nothing,
    )
end








mutable struct FullyConnected
    wiring::Wiring
    rng::AbstractRNG
    self_connection::Bool
end

function FullyConnected(units; output_dim=nothing, erev_init_seed=1111, self_connections=true)
    w = Wiring(units)
    out_dim = isnothing(output_dim) ? units : output_dim
    set_output_dim!(w, out_dim)
    rng = MersenneTwister(erev_init_seed)
    rand()
    for src in 1:units
        for dest in 1:units
            if src == dest && !(self_connections)
                continue
            end
            polarity = rand(rng, [-1, 1, 1])
            add_synapse!(w, src, dest, polarity)
        end
    end
    FullyConnected(w, rng, self_connections)
end

function build!(fc::FullyConnected, input_shape)
    build!(fc.wiring, input_shape)
    for src in 1:fc.wiring.input_dim
        for dest in 1:fc.wiring.units
            polarity = rand(fc.rng, [-1, 1, 1])
            add_sensory_synapse!(fc.wiring, src, dest, polarity)
        end
    end
end



mutable struct NCP
    wiring::Wiring
    rng::AbstractRNG
    num_inter_neurons::Int
    num_command_neurons::Int
    num_motor_neurons::Int
    sensory_fanout::Int
    inter_fanout::Int
    recurrent_command_synapses::Int
    motor_fanin::Int
end

function NCP(
    inter_neurons,
    command_neurons,
    motor_neurons,
    sensory_fanout,
    inter_fanout,
    recurrent_command_synapses,
    motor_fanin;
    seed=22222,
)
    w = Wiring(inter_neurons + command_neurons + motor_neurons)
    set_output_dim!(w, motor_neurons)
    rng = MersenneTwister(seed)

    if motor_fanin > command_neurons
        throw(ErrorException(
            "Error: Motor fanin parameter is $motor_fanin but there are only $command_neurons command neurons"
        ))
    elseif sensory_fanout > inter_neurons
        throw(ErrorException(
            "Error: Sensory fanout parameter is $sensory_fanout but there are only $inter_neurons inter neurons"
        ))
    elseif inter_fanout > command_neurons
        throw(ErrorException(
            "Error: Inter fanout parameter is $inter_fanout but there are only $command_neurons command neurons"
        ))
    else
        NCP(
            w,
            rng,
            inter_neurons,
            command_neurons,
            motor_neurons,
            sensory_fanout,
            inter_fanout,
            recurrent_command_synapses,
            motor_fanin
        )
    end
end

function Base.getproperty(ncp::NCP, v::Symbol)
    if v == :motor_neurons
        return 1:ncp.num_motor_neurons
    elseif v == :command_neurons
        return (ncp.num_motor_neurons + 1) : (ncp.num_motor_neurons + ncp.num_command_neurons)
    elseif v == :inter_neurons
        (ncp.num_motor_neurons + ncp.num_motor_neurons + 1):(ncp.num_motor_neurons + ncp.num_motor_neurons + ncp.num_inter_neurons)
    elseif v == :num_layers
        3
    elseif v == :units
        ncp.wiring.units
    elseif v == :num_sensory_neurons
        ncp.wiring.input_dim
    elseif v == :sensory_neurons
        1:ncp.wiring.input_dim
    else
        return getfield(ncp, v)
    end
end

function get_neurons_of_layer(ncp::NCP, layer_id)
    if layer_id == 1
        return ncp.inter_neurons
    elseif layer_id == 2
        return ncp.command_neurons
    elseif layer_id == 3
        return ncp.motor_neurons
    else
        throw(ErrorException(
            "layer_id must be in {1, 2, 3} — got $layer_id"
        ))
    end
end

function get_type_of_neuron(ncp::NCP, neuron_id)
    if neuron_id <= ncp.num_motor_neurons
        return "motor"
    elseif neuron_id <= ncp.num_motor_neurons + ncp.num_command_neurons
        return "command"
    else
        return "inter"
    end
end

function build_sensory_to_inter_layer!(ncp::NCP)
    inters = ncp.inter_neurons
    unreachable_inter_neurons = Set(inters)
    for src in ncp.sensory_neurons
        for dest in sample(ncp.rng, inters, ncp.sensory_fanout; replace=false)
            if dest in unreachable_inter_neurons
                delete!(unreachable_inter_neurons, dest)
            end
            polarity = rand(ncp.rng, [-1, 1])
            add_sensory_synapse!(ncp.wiring, src, dest, polarity)
        end
    end

    mean_inter_neuron_fanin = ncp.num_sensory_neurons * ncp.sensory_fanout / ncp.num_inter_neurons
    mean_inter_neuron_fanin = clamp(mean_inter_neuron_fanin |> floor |> Int, 1, ncp.num_sensory_neurons)

    for dest in unreachable_inter_neurons
        for src in sample(ncp.rng, ncp.sensory_neurons, mean_inter_neuron_fanin; replace=false)
            polarity = rand(ncp.rng, [-1, 1])
            add_sensory_synapse!(ncp.wiring, src, dest, polarity)
        end
    end
end

function build_inter_to_command_layer!(ncp::NCP)
    unreachable_inter_neurons = Set(ncp.command_neurons)
    for src in ncp.inter_neurons
        for dest in sample(ncp.rng, ncp.command_neurons, ncp.inter_fanout; replace=false)
            if dest in unreachable_inter_neurons
                delete!(unreachable_inter_neurons, dest)
            end
            polarity = rand(ncp.rng, [-1, 1])
            add_synapse!(ncp.wiring, src, dest, polarity)
        end
    end

    mean_inter_neuron_fanin = ncp.num_inter_neurons * ncp.inter_fanout / ncp.num_command_neurons
    mean_inter_neuron_fanin = clamp(mean_inter_neuron_fanin |> floor |> Int, 1, ncp.num_command_neurons)

    for dest in unreachable_inter_neurons
        for src in sample(ncp.rng, ncp.inter_neurons, mean_inter_neuron_fanin; replace=false)
            polarity = rand(ncp.rng, [-1, 1])
            add_synapse!(ncp.wiring, src, dest, polarity)
        end
    end
end

function build_recurrent_command_layer!(ncp::NCP)
    for i in 1:ncp.recurrent_command_synapses
        src = rand(ncp.rng, ncp.command_neurons)
        dest = rand(ncp.rng, ncp.command_neurons)
        polarity = rand(ncp.rng, [-1, 1])
        add_synapse!(ncp.wiring, src, dest, polarity)
    end
end

function build_command_to_motor_layer!(ncp::NCP)
    unreachable_inter_neurons = Set(ncp.command_neurons)
    for dest in ncp.motor_neurons
        for src in sample(ncp.rng, ncp.command_neurons, ncp.motor_fanin; replace=false)
            if src in unreachable_inter_neurons
                delete!(unreachable_inter_neurons, src)
            end
            polarity = rand(ncp.rng, [-1, 1])
            add_synapse!(ncp.wiring, src, dest, polarity)
        end
    end

    mean_inter_neuron_fanin = ncp.num_motor_neurons * ncp.motor_fanin / ncp.num_command_neurons
    mean_inter_neuron_fanin = clamp(mean_inter_neuron_fanin |> floor |> Int, 1, ncp.num_motor_neurons)

    for src in unreachable_inter_neurons
        for dest in sample(ncp.rng, ncp.motor_neurons, mean_inter_neuron_fanin; replace=false)
            polarity = rand(ncp.rng, [-1, 1]) 
            add_synapse!(ncp.wiring, src, dest, polarity)
        end
    end
end

function build!(ncp::NCP, input_shape)
    build!(ncp.wiring, input_shape)
    
    build_sensory_to_inter_layer!(ncp)
    build_inter_to_command_layer!(ncp)
    build_recurrent_command_layer!(ncp)
    build_command_to_motor_layer!(ncp)
end


function AutoNCP(units, output_size; sparsity_level = 0.5, seed = 22222)
    if output_size >= units - 2
        throw(ErrorException("Output size must be less than the number of units - 2"))
    elseif sparsity_level < .1 || sparsity_level > 1.0
        throw(ErrorException("Sparsity level must be between 0.0 and 0.9"))
    end
    density_level = 1.0 - sparsity_level
    inter_and_command_neurons = units - output_size
    command_neurons = max(Int(floor(0.4 * inter_and_command_neurons)), 1)
    inter_neurons = inter_and_command_neurons - command_neurons

    sensory_fanout = max(Int(floor(inter_neurons * density_level)), 1)
    inter_fanout = max(Int(floor(command_neurons * density_level)), 1)
    recurrent_command_synapses = max(Int(floor(command_neurons * density_level * 2)), 1)
    motor_fanin = max(Int(floor(command_neurons * density_level)), 1)
    return NCP(
        inter_neurons,
        command_neurons,
        output_size,
        sensory_fanout,
        inter_fanout,
        recurrent_command_synapses,
        motor_fanin;
        seed = seed
    )
end
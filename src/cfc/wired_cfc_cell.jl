mutable struct WiredCfCCell
    wiring::Union{Wiring, NCP, FullyConnected}
    layers::Vector{CfCCell}
end

function WiredCfCCell(input_size, wiring::Union{Wiring, NCP, FullyConnected}; mode="default")
    if !(isnothing(input_size))
        build!(wiring, input_size)
    end
    @assert wiring.built

    layers = []
    in_features = wiring.input_dim
    for l in 1:wiring.num_layers
        hidden_units = get_neurons_of_layer(wiring, l)
        input_sparsity = if l == 1
            wiring.sensory_adjacency_matrix[:, hidden_units]
        else
            prev_layer_neurons = get_neurons_of_layer(wiring, l-1)
            sp = wiring.adjacency_matrix[:, hidden_units]
            sp[prev_layer_neurons, :]
        end
        input_sparsity = hcat(input_sparsity, ones(length(hidden_units), length(hidden_units)))

        rnn_cell = CfCCell(
            in_features,
            length(hidden_units);
            mode = mode,
            backbone_activation = "lecun_tanh",
            backbone_units = 0,
            backbone_layers = 0,
            backbone_dropout = 0,
            sparsity_mask = input_sparsity
        )
        push!(layers, rnn_cell)
        in_features = length(hidden_units)
    end
    WiredCfCCell(wiring, layers)
end

function Base.getproperty(cell::WiredCfCCell, v::Symbol)
    if v == :state_size
        return cell.wiring.units
    elseif v == :layer_sizes
        return [length(get_neurons_of_layer(cell.wiring, i)) for i in 1:cell.wiring.num_layers]
    elseif v == :num_layers
        return cell.wiring.num_layers
    elseif v == :sensory_size
        return cell.wiring.input_dim
    elseif v == :motor_size
        return cell.wiring.output_dim
    elseif v == :output_size
        return cell.wiring.output_dim
    elseif v == :synapse_count
        return sum(abs.(wiring.adjacency_matrix))
    elseif v == :sensory_synapse_count
        return sum(abs.(wiring.sensory_adjacency_matrix))
    else
        return getfield(cell, v)
    end
end
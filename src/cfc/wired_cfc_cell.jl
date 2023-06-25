mutable struct WiredCfCCell
    wiring::Union{Wiring,NCP,FullyConnected}
    layers::Vector{CfCCell}
end

function WiredCfCCell(input_size, wiring::Union{Wiring,NCP,FullyConnected}; mode="default")
    if !(isnothing(input_size))
        build!(wiring, input_size)
    end

    layers = []
    in_features = wiring.input_dim
    for l in 1:wiring.num_layers
        hidden_units = get_neurons_of_layer(wiring, l)
        input_sparsity = if l == 1
            wiring.sensory_adjacency_matrix[:, hidden_units]
        else
            prev_layer_neurons = get_neurons_of_layer(wiring, l - 1)
            sp = wiring.adjacency_matrix[:, hidden_units]
            sp[prev_layer_neurons, :]
        end
        input_sparsity = vcat(input_sparsity, ones(Float32, length(hidden_units), length(hidden_units)))

        rnn_cell = CfCCell(
            in_features,
            length(hidden_units);
            mode=mode,
            backbone_activation="lecun_tanh",
            backbone_units=0,
            backbone_layers=0,
            backbone_dropout=0,
            sparsity_mask=input_sparsity
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







# Python equivalent:
# def forward(self, input, hx, timespans):
#     h_state = torch.split(hx, self.layer_sizes, dim=1)

#     new_h_state = []
#     inputs = input
#     for i in range(self.num_layers):
#         h, _ = self._layers[i].forward(inputs, h_state[i], timespans)
#         inputs = h
#         new_h_state.append(h)

#     new_h_state = torch.cat(new_h_state, dim=1)
#     return h, new_h_state

# Here, h_state represents the hidden state of the previous time step

# equivalently in Julia,


function split(h, layer_sizes)
    # h is of shape (hidden_size, batch_size)
    # layer_sizes is a vector of the number of neurons in each layer
    # dims is the dimension along which to split
    # returns a vector of tensors of shape (layer_size, batch_size)
    # e.g. split(h, [3, 4, 5], dims=1) returns a vector of length 3
    # where each element is a tensor of shape (3, batch_size), (4, batch_size), (5, batch_size)
    # respectively
    start = 1
    out = []
    for layer_size in layer_sizes
        push!(out, h[start:start+layer_size-1, :])
        start += layer_size
    end
    return out
end

function (cell::WiredCfCCell)(h::AbstractArray, x::AbstractArray)
    # x is of shape (input_size, batch_size, sequence_length)
    # in julia, we can't split into those sizes with a split method, but we can do this:

    layers = cell.layers
    layer_sizes = cell.layer_sizes

    function forward(x, h; start=1, layer_index=1)
        s = layer_sizes[layer_index]
        layer = layers[layer_index]
        out = layer(h[start:start+s-1, :], x)
        if layer_index == length(layers)
            return out, out
        end
        next, y = forward(out, h; start=start+s, layer_index=layer_index+1)
        return vcat(out, next), y
    end

    h_state, y = forward(x, h)
    
    return h_state, Flux.reshape_cell_output(y, x)
end

Flux.@functor WiredCfCCell
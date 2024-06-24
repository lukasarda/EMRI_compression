import torch.nn as nn


### Conv2d
def conv2d_shape(h_in, w_in, pad, dil, ker, str):
    h_out = int((h_in + 2 * pad[0] - dil[0] * (ker[0] - 1) - 1) / str[0] + 1)
    w_out = int((w_in + 2 * pad[1] - dil[1] * (ker[1] - 1) - 1) / str[1] + 1)
    return h_out, w_out

### TransposeConv2d
def transposeconv2d_shape(h_in, w_in, pad, dil, ker, str, out_pad):
    h_out = (h_in - 1) * str[0] - 2 * pad[0] + dil[0] * (ker[0] - 1) + out_pad[0] + 1
    w_out = (w_in - 1) * str[1] - 2 * pad[1] + dil[1] * (ker[1] - 1) + out_pad[1] + 1
    return h_out, w_out

# Extract hyperparameters function
def extract_hyperparameters(model):
    def to_tuple(value):
        if isinstance(value, int):
            return (value, value)
        return value
    
    hyperparams = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
            hyperparams[name] = {
                'layer_type': type(layer).__name__,
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': to_tuple(layer.kernel_size),
                'stride': to_tuple(layer.stride),
                'padding': to_tuple(layer.padding),
                'dilation': to_tuple(layer.dilation),
                'bias': layer.bias is not None
            }
            if isinstance(layer, nn.ConvTranspose2d):
                hyperparams[name]['output_padding'] = to_tuple(layer.output_padding)
        elif isinstance(layer, nn.BatchNorm2d):
            hyperparams[name] = {
                'layer_type': type(layer).__name__,
                'num_features': layer.num_features
            }
        elif isinstance(layer, nn.MaxPool2d):
            hyperparams[name] = {
                'layer_type': type(layer).__name__,
                'kernel_size': to_tuple(layer.kernel_size),
                'stride': to_tuple(layer.stride),
                'padding': to_tuple(layer.padding)
            }
        elif isinstance(layer, nn.LeakyReLU) or isinstance(layer, nn.ReLU):
            hyperparams[name] = {
                'layer_type': type(layer).__name__,
                'inplace': layer.inplace
            }
        elif isinstance(layer, nn.Sigmoid):
            hyperparams[name] = {
                'layer_type': type(layer).__name__
            }
    
    # Sorting encoder and decoder layers separately
    encoder_layers = {name: params for name, params in hyperparams.items() if name.startswith('encoder')}
    decoder_layers = {name: params for name, params in hyperparams.items() if name.startswith('decoder')}
    
    sorted_encoder_layers = {k: encoder_layers[k] for k in sorted(encoder_layers, key=lambda x: int(x.split('.')[1]))}    
    sorted_decoder_layers = {k: decoder_layers[k] for k in sorted(decoder_layers, key=lambda x: int(x.split('.')[1]))}
    
    sorted_hyperparams = sorted_encoder_layers | sorted_decoder_layers
    
    return sorted_hyperparams

# Calculate output shapes
def calculate_output_shapes(h_in, w_in, c_in, hyperparams):
    shapes = {}
    current_shape = (h_in, w_in, c_in)
    
    for name, params in hyperparams.items():
        if params['layer_type'] == 'Conv2d':
            h_out, w_out = conv2d_shape(
                current_shape[0], current_shape[1],
                params['padding'], params['dilation'], params['kernel_size'], params['stride']
            )
            current_shape = (h_out, w_out, params['out_channels'])
        elif params['layer_type'] == 'ConvTranspose2d':
            h_out, w_out = transposeconv2d_shape(
                current_shape[0], current_shape[1],
                params['padding'], params['dilation'], params['kernel_size'], params['stride'],
                params['output_padding']
            )
            current_shape = (h_out, w_out, params['out_channels'])
        elif params['layer_type'] == 'MaxPool2d':
            h_out, w_out = conv2d_shape(
                current_shape[0], current_shape[1],
                params['padding'], (1, 1), params['kernel_size'], params['stride']
            )
            current_shape = (h_out, w_out, current_shape[2])
        
        shapes[name] = current_shape
    
    return shapes

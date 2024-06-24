import torch
import pprint

from calc_shape import calculate_output_shapes, extract_hyperparameters
from neural_net_class import CL_maxPool

def check_shapes(model):
    hyperparams = extract_hyperparameters(model(channel_mult=8))
    # Define initial shape
    initial_shape = (512, 1030, 2)  # Example input shape (height, width, channels)
    # Calculate and print output shapes
    output_shapes = calculate_output_shapes(initial_shape[0], initial_shape[1], initial_shape[2], hyperparams)
    pprint.pprint(output_shapes, sort_dicts=False)

# check_shapes(CL_maxPool)

def cn_enc_out_shape(channel_mult, input_shape):
    hyperparams = extract_hyperparameters(CL_maxPool(channel_mult= channel_mult, h_kernel= 1, w_kernel= 1))
    output_shapes = calculate_output_shapes(h_in= input_shape[0], w_in= input_shape[1], c_in= 2, hyperparams= hyperparams)
    
    encoder_entries = {k: v for k, v in output_shapes.items() if k.startswith('encoder')}
    decoder_entries = {k: v for k, v in output_shapes.items() if k.startswith('decoder')}

    h_cn_enc_out, w_cn_enc_out = list(encoder_entries.items())[-1][1][0], list(encoder_entries.items())[-1][1][1]

    h_kernel = list(encoder_entries.items())[0][1][0] - list(decoder_entries.items())[-1][1][0] + 1
    w_kernel = list(encoder_entries.items())[0][1][1] - list(decoder_entries.items())[-1][1][1] + 1

    return h_cn_enc_out, w_cn_enc_out, h_kernel, w_kernel

# cn_enc_out_shape(8, (512,1030))

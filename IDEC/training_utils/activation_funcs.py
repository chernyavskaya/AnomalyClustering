import torch.nn as nn

def get_activation_func(str_name):
    activations = {
    'elu': nn.ReLU(),
    'relu': nn.ELU(),
    'tanh': nn.Tanh()
    }
    parsed_name = str_name.split('_')
    if len(parsed_name)==1:
        return activations[parsed_name[0]]
    else:
        if 'leakyrelu' in  parsed_name[0]:
            return nn.LeakyReLU(negative_slope=float(parsed_name[1]))
        else :
            print('Activation function name is not recognized.')
            print('Acepted activation functions : relu, elu, leakyrelu_floatslope, tanh')
            return None

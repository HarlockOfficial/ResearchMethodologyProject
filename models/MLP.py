import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, input_dim:int, output_dim:int, hidden_dims:list[int], activation:str='ReLU', dropout:float=0.0):
        super(MLP, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(1, len(dims)):
            self.add_module(f'Linear_{i}', nn.Linear(dims[i-1], dims[i]))
            if i < len(dims)-1:
                self.add_module(f'Activation_{i}', getattr(nn, activation)())
                self.add_module(f'Dropout_{i}', nn.Dropout(dropout))
        self.add_module('Sigmoid', nn.Sigmoid())

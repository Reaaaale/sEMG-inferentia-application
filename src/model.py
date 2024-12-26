import torch
import lava

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 1.0,  
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
        }
        neuron_params_drop = {**neuron_params, 'dropout': lava.lib.dl.slayer.neuron.Dropout(p=0.01)}

        layers = []
        ofs = [32, 64, 32, None]  # 
        for i in range(len(ofs)):
            if i == len(ofs) - 1:
                layer = lava.lib.dl.slayer.block.cuba.Dense(
                    neuron_params,
                    ofs[i - 1],
                    ofs[i],
                    weight_norm=True
                )
            else:
                input_size = ofs[i - 1] if i > 0 else 11  
                layer = lava.lib.dl.slayer.block.cuba.Dense(
                    neuron_params_drop,
                    input_size,
                    ofs[i],
                    weight_norm=True,
                    delay=1  
                )
            layers.append(layer)
        self.blocks = torch.nn.ModuleList(layers)

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

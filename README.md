# Quantum neural network

This is a library for creating and training quantum neural networks with Qiskit.

## Training with entirely quantum networks

Images can be loaded into a quantum circuit using the data handlers in `quantum-neural-network/input`. The parameters
can then be trained by an external optimiser.

A working example of a single forward pass for a random input vector can be found in `run_simple_network.py`. This uses
a `vector_data_handler`
in which the data is inputted as single qubit rotations on a product state.

These networks can run by themselves or can be integrated as layers into a larger classical neural network.

## Embedding into classical neural networks

For running on current quantum computers, it may be beneficial to embed a QNN within a classical network. Rather than
inputting a whole image into the quantum circuit, the feature vector from the previous classical layer is passed in.

A working example of how to integrate our `QNet`
class with PyTorch can be found in `mnist_examples/pytorch_with_qnet.py`. The general structure is:

```python
from qnet import QNet
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CLASSICAL PYTORCH LAYERS

        self.qnet = QNet(n_qubits=2, encoding='vector', ansatz_type='farhi', layers=1,
                         activation_function_type='partial_measurement_1')

    def forward(self, x):
        # CLASSICAL PYTORCH LAYERS

        x = self.qnet(x)
        return x
```

where `QNet` returns a feature vector, the length of which is determiend by the input dimension and ansatz type.

## Config options

### Ansatzes

- `abbas` (https://arxiv.org/abs/2011.00027)
- `alternating_layer_tdcnot` (https://arxiv.org/abs/2002.04612)
- `farhi` (https://arxiv.org/abs/1802.06002)
- `sim_circ_13, sim_circ_13_half, sim_circ_14, sim_circ_14_half, sim_circ_15, sim_circ_19` (https://arxiv.org/abs/1905.10876)

### Quantum activation functions

- `parial_meas_x` where x is the number of qubits to be measured between each layer. `x` can also be 'half'.

### Data handlers

- `frqi` (https://link.springer.com/article/10.1007/s11128-010-0177-y)
- `havlicek` (https://arxiv.org/abs/1804.11326)
- `neqr` (https://link.springer.com/article/10.1007/s11128-013-0567-z)
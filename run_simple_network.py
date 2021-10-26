"""Single forward pass of quantum neural network."""
import logging
import sys

import numpy as np

sys.path.append('qnn')

from config import Config
from quantum_network_circuit import QuantumNetworkCircuit

logging.basicConfig(level=logging.INFO)
logging.getLogger('qiskit').setLevel(logging.WARN)


def sum_vector_cost_func(vector):
    return sum(vector)


n_input_data = 4

input_data = np.random.random(n_input_data)
config = Config(encoding='vector', ansatz_type='sim_circ_14', layers=1, sweeps_per_layer=1,
                activation_function_type='null', meas_method='all', backend_type='statevector_simulator')

qnn = QuantumNetworkCircuit(config, n_input_data, input_data)
parameter_values = np.random.random(len(qnn.qc.parameters))

print('Network with input from previous layer {}'.format(input_data))
print('Network initialised with weights {}'.format(parameter_values[n_input_data:]))

measurement_result = qnn.get_vector_from_results(qnn.evaluate_circuit(parameter_values))

qnn.qc.draw(output='mpl').show()

print('Cost function result = {}'.format(sum_vector_cost_func(measurement_result)))

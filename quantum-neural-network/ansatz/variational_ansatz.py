from abc import ABC, abstractmethod

from qiskit import QuantumRegister, QuantumCircuit


class VariationalAnsatz(ABC):

    def __init__(self, layers, sweeps_per_layer, activation_function):
        self.layers = layers
        self.sweeps_per_layer = sweeps_per_layer
        self.qr = None
        self.qc = None
        self.n_parameters_required = None
        self.activation_function = activation_function

        self.param_counter = 0

    # Base logic for creating an ansatz, can be overwritten by children
    def get_quantum_circuit(self, n_data_qubits):
        self.qr = QuantumRegister(n_data_qubits, name='qr')
        self.qc = QuantumCircuit(self.qr, name='Shifted circ')

        for layer_no in range(self.layers):
            for sweep in range(0, self.sweeps_per_layer):
                self.add_rotations(n_data_qubits)
                self.add_entangling_gates(n_data_qubits)
            if layer_no < self.layers - 1:
                self.apply_activation_function(n_data_qubits)
        return self.qc

    @abstractmethod
    def add_rotations(self, n_data_qubits):
        pass

    @abstractmethod
    def add_entangling_gates(self, n_data_qubits):
        pass

    def apply_activation_function(self, n_data_qubits):
        activation_function_circuit = self.activation_function.get_quantum_circuit(n_data_qubits)
        self.qc.extend(activation_function_circuit)
        return self.qc

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from ansatz.variational_ansatz import VariationalAnsatz


class SimCirc13(VariationalAnsatz):
    """
    A variational circuit ansatz. Based on circuit 13 in arXiv:1905.10876.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        super().__init__(layers, sweeps_per_layer, activation_function)

    def get_quantum_circuit(self, n_data_qubits):
        self.qr = QuantumRegister(n_data_qubits, name='qr')
        self.qc = QuantumCircuit(self.qr, name='Shifted circ')

        for layer_no in range(self.layers):
            for sweep in range(0, self.sweeps_per_layer):
                self.add_rotations(n_data_qubits)
                self.add_entangling_gates(n_data_qubits, block=1)
                self.add_rotations(n_data_qubits)
                self.add_entangling_gates(n_data_qubits, block=2)
            if layer_no < self.layers - 1:
                self.apply_activation_function(n_data_qubits)
        return self.qc

    def add_rotations(self, n_data_qubits):
        for i in range(0, n_data_qubits):
            param = Parameter("ansatz{}".format(str(self.param_counter)))
            self.qc.ry(param, self.qr[i])
            self.param_counter += 1
        return self.qc

    def add_entangling_gates(self, n_data_qubits, block=1):
        if block == 1:
            for i in reversed(range(n_data_qubits)):
                param = Parameter("ansatz{}".format(str(self.param_counter)))
                self.qc.crz(param, self.qr[i], self.qr[(i + 1) % n_data_qubits])
                self.param_counter += 1

        elif block == 2:
            for i in range(n_data_qubits):
                param = Parameter("ansatz{}".format(str(self.param_counter)))
                control_qubit = (i + n_data_qubits - 1) % n_data_qubits
                self.qc.crz(param, self.qr[control_qubit], self.qr[(control_qubit + 3) % n_data_qubits])
                self.param_counter += 1
        return self.qc

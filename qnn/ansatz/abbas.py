from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

from ansatz.variational_ansatz import VariationalAnsatz


class Abbas(VariationalAnsatz):
    """
    A variational circuit ansatz. Based on Figure 5 of arXiv:2011.00027 - which itself is based on circuit 15 in
    arXiv:1905.10876 but with all-to-all CNOT connectivity.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        super().__init__(layers, sweeps_per_layer, activation_function)

    def get_quantum_circuit(self, n_data_qubits):
        self.qr = QuantumRegister(n_data_qubits, name='qr')
        self.qc = QuantumCircuit(self.qr, name='Shifted circ')

        for layer_no in range(self.layers):
            if layer_no == 0:
                self.add_rotations(n_data_qubits)
            for sweep in range(0, self.sweeps_per_layer):
                self.add_entangling_gates(n_data_qubits)
                self.add_rotations(n_data_qubits)
            if layer_no < self.layers - 1:
                self.apply_activation_function(n_data_qubits)
        return self.qc

    def add_rotations(self, n_data_qubits):
        for i in range(0, n_data_qubits):
            param = Parameter("ansatz{}".format(str(self.param_counter)))
            self.qc.ry(param, self.qr[i])
            self.param_counter += 1
        return self.qc

    def add_entangling_gates(self, n_data_qubits):
        for i in range(n_data_qubits):
            for j in range(i + 1, n_data_qubits):
                self.qc.cx(self.qr[i], self.qr[j])
        return self.qc

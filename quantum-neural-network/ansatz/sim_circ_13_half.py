from qiskit.circuit import Parameter

from ansatz.variational_ansatz import VariationalAnsatz


class SimCirc13Half(VariationalAnsatz):
    """
    A variational circuit ansatz. Based on circuit 13 in arXiv:1905.10876 but modified to only use the first half of the
    circuit. Between each layer an activation function can be applied using appropriate nonlinear activation function.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        super().__init__(layers, sweeps_per_layer, activation_function)

    def add_rotations(self, n_data_qubits):
        for i in range(0, n_data_qubits):
            param = Parameter("ansatz{}".format(str(self.param_counter)))
            self.qc.ry(param, self.qr[i])
            self.param_counter += 1
        return self.qc

    def add_entangling_gates(self, n_data_qubits):
        for i in range(n_data_qubits):
            param = Parameter("ansatz{}".format(str(self.param_counter)))
            self.qc.crz(param, self.qr[i], self.qr[(i + 1) % n_data_qubits])
            self.param_counter += 1
        return self.qc

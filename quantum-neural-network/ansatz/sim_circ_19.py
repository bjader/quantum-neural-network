from qiskit.circuit import Parameter

from ansatz.variational_ansatz import VariationalAnsatz


class SimCirc19(VariationalAnsatz):
    """
    A variational circuit ansatz. Prepares quantum circuit object for variational circuit 19 in arXiv:1905.10876.
    Between each layer an activation function can be applied using appropriate nonlinear activation function.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        super().__init__(layers, sweeps_per_layer, activation_function)

    def add_rotations(self, n_data_qubits):
        for i in range(0, n_data_qubits):
            param = Parameter("ansatz{}".format(str(self.param_counter)))
            self.qc.rx(param, self.qr[i])
            self.param_counter += 1

        for i in range(0, n_data_qubits):
            param = Parameter("ansatz{}".format(str(self.param_counter)))
            self.qc.rz(param, self.qr[i])
            self.param_counter += 1

        return self.qc

    def add_entangling_gates(self, n_data_qubits):
        param = Parameter("ansatz{}".format(str(self.param_counter)))
        self.qc.crx(param, self.qr[n_data_qubits - 1], self.qr[0])
        self.param_counter += 1
        for i in reversed(range(1, n_data_qubits)):
            param = Parameter("ansatz{}".format(str(self.param_counter)))
            self.qc.crx(param, self.qr[i - 1], self.qr[i])
            self.param_counter += 1
        return self.qc

from qiskit.circuit import Parameter

from ansatz.variational_ansatz import VariationalAnsatz


class AlternatingLayerTDCnotAnsatz(VariationalAnsatz):
    """
    A variational circuit ansatz. Prepares quantum circuit object for variational circuit consisting of thinly
    dressed C-NOT gates applied to pairs of qubits in an alternating layer pattern. A single sweep consists of two
    applications of thinly-dressed gates between nearest neighbour qubits, with alternating pairs between the two
    applications. The thinly dressed C-NOT gate is define in arXiv:2002.04612, we use single qubit Y rotations
    preceding and single qubit X rotations following the C-NOT gate.
    """

    def __init__(self, layers, sweeps_per_layer, activation_function):
        super().__init__(layers, sweeps_per_layer, activation_function)

    def add_entangling_gates(self, n_data_qubits):

        for i in range(n_data_qubits - 1)[::2]:
            ctrl, tgt = i, ((i + 1) % self.qc.num_qubits)
            self.build_tdcnot(ctrl, tgt)

        for i in range(n_data_qubits)[1::2]:
            ctrl, tgt = i, ((i + 1) % self.qc.num_qubits)
            self.build_tdcnot(ctrl, tgt)

        return self.qc

    def build_tdcnot(self, ctrl, tgt):
        params = [Parameter("ansatz{}".format(str(self.param_counter + j))) for j in range(4)]
        self.qc.ry(params[0], self.qr[ctrl])
        self.qc.ry(params[1], self.qr[tgt])
        self.qc.cx(ctrl, tgt)
        self.qc.rz(params[2], self.qr[ctrl])
        self.qc.rz(params[3], self.qr[tgt])
        self.param_counter += 4

    def add_rotations(self, n_data_qubits):
        pass

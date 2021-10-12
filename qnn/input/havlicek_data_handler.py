import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter

from input.data_handler import DataHandler


class HavlicekDataHandler(DataHandler):
    """
    Data encoding based on Havlicek et al. Nature 567, pp209–212 (2019). For quantum circuit diagram see Fig. 4 in
    arXiv:2011.00027.
    """

    def __init__(self):
        super().__init__()

    def get_quantum_circuit(self, input_data):
        self.qr = QuantumRegister(len(input_data))
        self.qc = QuantumCircuit(self.qr)
        num_qubits = len(input_data)
        param_list = []
        for index in range(num_qubits):
            self.qc.h(self.qr[index])

            param = Parameter("input{}".format(str(index)))
            param_list.append(param)

            self.qc.rz(param, self.qr[index])

        for i in range(num_qubits - 1):
            for j in range(i + 1, num_qubits):
                param_i = param_list[i]
                param_j = param_list[j]
                self.qc.rzz((param_i - np.pi / 2) * (param_j - np.pi / 2), i, j)

        return self.qc

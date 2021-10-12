from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter

from input.data_handler import DataHandler


class VectorDataHandler(DataHandler):
    def get_quantum_circuit(self, input_data):
        self.qr = QuantumRegister(len(input_data))
        self.qc = QuantumCircuit(self.qr)

        for index, _ in enumerate(input_data):
            param = Parameter("input{}".format(str(index)))
            self.qc.rx(param, index)

        return self.qc

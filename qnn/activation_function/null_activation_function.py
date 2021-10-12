from qiskit import QuantumRegister, QuantumCircuit

from activation_function.activation_function import ActivationFunction


class NullActivationFunction(ActivationFunction):
    """
    Creates activation function circuit corresponding to identity operation i.e. no activation function
    """

    def __init__(self):
        super().__init__()

    def get_quantum_circuit(self, n_qubits):
        self.qr = QuantumRegister(n_qubits, name='qr')
        self.qc = QuantumCircuit(self.qr)

        return self.qc

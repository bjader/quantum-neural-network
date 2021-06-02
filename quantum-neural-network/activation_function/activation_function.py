from abc import ABC, abstractmethod


class ActivationFunction(ABC):

    def __init__(self):
        self.qr = None
        self.cr = None
        self.qc = None

    @abstractmethod
    def get_quantum_circuit(self, n_qubits):
        """
        Returns a quantum circuit for implementing the nonlinear action
        """
        pass

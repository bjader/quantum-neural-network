import logging
from math import floor

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from activation_function.activation_function import ActivationFunction


class PartialMeasActivationFunction(ActivationFunction):
    """
    Creates activation function circuit in which a subset of qubits are measured and set to measurement value. Qubits to
     be to be measured will be equally spaced in circuit.
    """

    def __init__(self, n_measurements):
        super().__init__()
        self.n_measurements = n_measurements

    def get_quantum_circuit(self, n_qubits):

        if self.n_measurements == 'half':
            self.n_measurements = floor(n_qubits / 2)

        if self.n_measurements > n_qubits:
            raise ValueError('Activation function was asked to measure more qubits than exist in the circuit.')

        if n_qubits % self.n_measurements != 0:
            logging.warning(
                f'In acivation function, number of qubits ({n_qubits}) is not multiple of number of measurements '
                f'({self.n_measurements}), measurements will not be equally spaced in circuit.')

        self.qr = QuantumRegister(n_qubits, name='qr')
        self.cr = ClassicalRegister(self.n_measurements, name='activation_cr')
        self.qc = QuantumCircuit(self.qr, self.cr, name='Partial measurement')

        step = floor(n_qubits / self.n_measurements)

        for i, qubit in enumerate([step * j for j in range(0, self.n_measurements)]):
            self.qc.measure(self.qr[qubit], self.cr[i])

        return self.qc

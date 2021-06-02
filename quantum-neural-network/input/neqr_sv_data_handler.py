import numpy as np
from qiskit import QuantumCircuit

from input.data_handler import DataHandler


class NEQRSVDataHandler(DataHandler):
    """
    A statevector data handler. This will represent the input data as a state vector and then ask Qiskit
    to find a circuit which initializes this state.
    """

    def __init__(self):
        super().__init__()

    def get_quantum_circuit(self, input_data):

        encoded_data = self.encode_using_neqr(input_data)

        state = self.create_input_state(encoded_data)

        qc = QuantumCircuit((len(state) - 1).bit_length())
        qc.initialize(state, qc.qregs)

        return qc

    def create_input_state(self, bitstrings):
        state = self.convert_bitstring_to_statevector(bitstrings[0])

        for bitstring in bitstrings[1:]:
            state += self.convert_bitstring_to_statevector(bitstring)

        normalized_state = state / np.linalg.norm(state)

        return normalized_state

    def convert_bitstring_to_statevector(self, bitstring):
        state = np.array([1])

        for b in bitstring:
            ket = self.convert_bit_to_bloch(b)
            state = np.kron(state, ket)

        return state

    def convert_bit_to_bloch(self, digit):
        if digit == 0:
            return np.array([1, 0])

        elif digit == 1:
            return np.array([0, 1])

        else:
            raise ValueError("Cannot convert non binary digit")

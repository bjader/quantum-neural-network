import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from input.data_handler import DataHandler


class FRQIBennettDataHandler(DataHandler):
    """
    A Bennett data handler for FRQI encoding. Prepares the input data using our hand-crafted scheme rather than Qiskit's initialize
    function. This has the benefit of being scalable since we don't need to store the statevector.

    The scheme is as follows:
    1. Prepare the location qubits into all computational basis states (hadamard each qubit H^⊗n)
    2. Prepare the pixels. This is achieved by:
        a. Pick a location basis. Apply X gates so that every qubit is in the |1> state.
        b. Apply N-control-Y-rotation gates to the colour qubits to rotate by angle encoding colour.
            The control should be across all location qubits.
        c. Apply X gates to reverse step a. such that the real location state is restored.
        d. Cycle to the next location basis and repeat.
    """

    def __init__(self):
        super().__init__()

    def get_quantum_circuit(self, input_data):
        self.n_location_qubits = (len(input_data) - 1).bit_length()
        self.n_colour_qubits = 1
        self.n_ancilla_qubits = max(self.n_location_qubits - 2, 0)

        self.qr = QuantumRegister(self.n_location_qubits + self.n_ancilla_qubits + self.n_colour_qubits)
        self.qc = QuantumCircuit(self.qr, name='FRQI')

        self.location_register = self.qr[:self.n_location_qubits]
        self.colour_register = self.qr[self.n_location_qubits:self.n_location_qubits + self.n_colour_qubits]
        self.ancilla_register = self.qr[
                                self.n_location_qubits + self.n_colour_qubits:self.n_location_qubits + self.n_colour_qubits + self.n_ancilla_qubits:]

        location_strings, angle_data = self.encode_using_frqi(input_data)

        self.qc.h(self.location_register)

        self.prepare_colour_pixels(angle_data, location_strings)

        self.qc.barrier()

        return self.qc

    def prepare_colour_pixels(self, angle_data, location_strings):
        for i, location_string in enumerate(location_strings):
            theta = angle_data[i] * np.pi
            if theta != 0:
                location_qubits_flipped = self.make_location_qubits_ones(location_string)

                self.qc.mcry(theta, self.location_register, self.colour_register[0], self.ancilla_register,
                             mode='basic')

                self.reverse_location_flips(location_qubits_flipped)

    def make_location_qubits_ones(self, location_string):  # Repeated for FRQI and NEQR - move to DataHandler Class

        location_qubits_to_flip = np.where(location_string == 0)[0]

        for qubit in location_qubits_to_flip:
            self.qc.x(self.location_register[int(qubit)])

        return location_qubits_to_flip

    def reverse_location_flips(self, qubits_to_flip):

        for qubit in qubits_to_flip:
            self.qc.x(qubit)

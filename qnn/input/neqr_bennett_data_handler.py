import logging

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from input.data_handler import DataHandler


class NEQRBennettDataHandler(DataHandler):
    """
    A Bennett data handler. Prepares the input data using our hand-crafted scheme rather than Qiskit's initialize
    function. This has the benefit of being scalable since we don't need to store the statevector.

    The scheme is as follows:
    1. Prepare the location qubits into all computational basis states (hadamard each qubit H^⊗n)
    2. Prepare the colour qubits into a base colour who's bitstring has the most popular bit at each position. This
        ensures the minimum number of gates applied at step 3.
    3. Prepare the pixels. This is achieved by:
        a. Pick a location basis. Apply X gates so that every qubit is in the |1> state.
        b. Apply N-control-X gates to the colour qubits that differ between the base colour and desired pixel colour.
            The control should be across all location qubits.
        c. Apply X gates to reverse step a. such that the real location state is restored.
        d. Cycle to the next location basis and repeat.
    """

    def __init__(self):
        super().__init__()

    def get_quantum_circuit(self, input_data):
        self.n_location_qubits = (len(input_data) - 1).bit_length()
        self.n_ancilla_qubits = max(self.n_location_qubits - 2, 0)
        self.n_colour_qubits = len(input_data[0])

        self.qr = QuantumRegister(self.n_location_qubits + self.n_ancilla_qubits + self.n_colour_qubits)
        self.qc = QuantumCircuit(self.qr, name='NEQR')

        self.colour_register = self.qr[self.n_location_qubits:self.n_location_qubits + self.n_colour_qubits]
        self.ancilla_register = self.qr[
                                self.n_location_qubits + self.n_colour_qubits:self.n_location_qubits + self.n_colour_qubits + self.n_ancilla_qubits:]

        encoded_data = self.encode_using_neqr(input_data)

        # Prepare the location qubits in all computational basis
        self.qc.h(self.location_register)

        base_colour = self.prepare_base_colour(encoded_data)

        self.prepare_colour_pixels(encoded_data, base_colour)

        self.qc.barrier()

        return self.qc

    def prepare_base_colour(self, encoded_data):

        bit_frequency = np.zeros(self.n_colour_qubits)

        for bit_string in encoded_data:
            bit_frequency += bit_string[-self.n_colour_qubits:]

        mode_bit_string = np.array(
            [int(np.floor(((2 * val) - 1) / len(encoded_data))) if val != 0 else 0 for val in bit_frequency])

        logging.info(
            "Over all pixels, the modal bit at each position creates the base colour {}".format(mode_bit_string))

        for colour_qubit_to_flip in np.where(mode_bit_string == 1)[0]:
            self.qc.x(self.colour_register[colour_qubit_to_flip])

        return mode_bit_string

    def prepare_colour_pixels(self, encoded_data, base_colour):

        for bitstring in encoded_data:
            location_string = bitstring[:self.n_location_qubits]
            location_qubits_flipped = self.make_location_qubits_ones(location_string)

            target_colour = bitstring[self.n_location_qubits:]
            self.change_base_colour_to_pixel_colour(target_colour, base_colour)

            self.reverse_location_flips(location_qubits_flipped)

    def make_location_qubits_ones(self, location_string):

        location_qubits_to_flip = np.where(location_string == 0)[0]

        for qubit in location_qubits_to_flip:
            self.qc.x(self.location_register[qubit])

        return location_qubits_to_flip

    def change_base_colour_to_pixel_colour(self, target_colour, base_colour):

        colour_qubits_to_flip = np.where(base_colour != target_colour)[0]

        for qubit in colour_qubits_to_flip:
            self.qc.mct(self.location_register, self.colour_register[qubit],
                        self.ancilla_register, mode='basic')

    def reverse_location_flips(self, qubits_to_flip):

        for qubit in qubits_to_flip:
            self.qc.x(qubit)

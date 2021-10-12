from abc import ABC, abstractmethod
from itertools import product

import numpy as np


class DataHandler(ABC):

    def __init__(self):

        self.n_location_qubits = None
        self.n_ancilla_qubits = None
        self.n_colour_qubits = None

        self.qr = None
        self.qc = None

        self.location_register = None
        self.colour_register = None
        self.ancilla_register = None

    @abstractmethod
    def get_quantum_circuit(self, input_data):
        """
        Returns a quantum circuit which initializes the qubits to the input data
        :return:
        """
        pass

    def encode_using_neqr(self, input_data):
        """
        Novel enhanced quantum representation (NEQR) encoding, where a pixel's location and value data is encoded in one
        computational basis state. To do this we append the location of pixels at the front of the bitstring.
        """
        n_location_bits = (len(input_data) - 1).bit_length()

        location_strings = [i for i in product([0, 1], repeat=n_location_bits)]

        encoded_data = []
        for index, location_string in enumerate(location_strings):
            encoded_data.append(np.concatenate([location_string, input_data[index]]))

        return encoded_data

    def encode_using_frqi(self, input_data):

        n_location_bits = (len(input_data) - 1).bit_length()

        location_strings = np.array([i for i in product([0, 1], repeat=n_location_bits)])

        colour_ints = np.array([])
        normalisation = 2 ** len(input_data[0])
        for colour_bit_string in input_data:
            out = 0
            for bit in colour_bit_string:
                out = (out << 1) | bit
            colour_ints = np.append(colour_ints, out)
        angle_data = colour_ints / normalisation

        return location_strings, angle_data

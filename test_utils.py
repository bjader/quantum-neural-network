import logging

import numpy as np


def create_random_bitstring(num_bits):  # repeated code from random_data_test
    return np.random.choice([0, 1], size=num_bits)


def create_random_bitstrings(num_strings, num_bits):  # repeated code from random_data_test
    bitstrings = []
    for i in range(num_strings):
        bitstrings.append(create_random_bitstring(num_bits))

    logging.info("Random pixels created with values {}".format(bitstrings))
    return bitstrings

import sys
from unittest import TestCase

import numpy as np

sys.path.append('quantum-neural-network')

from config import Config
from gradient_calculator import calculate_gradient_list
from quantum_network_circuit import QuantumNetworkCircuit


class TestGradientCalculator(TestCase):
    def test_when_calculate_gradient_list_then_happy_path(self):
        config = Config('vector')
        qnn = QuantumNetworkCircuit(config, 2, [1, 1])
        calculate_gradient_list(qnn, [1, 1])

    def test_when_calculate_gradient_list_using_parameter_shifterence_then_happy_path(self):
        config = Config('vector')
        qnn = QuantumNetworkCircuit(config, 2, [1, 1])
        calculate_gradient_list(qnn, [1, 1], method='parameter shift', eps=0.1)

    def test_when_calculate_gradient_list_then_non_zero_gradient(self):
        config = Config('vector', ansatz_type='sim_circ_13_half', layers=1, backend_type='statevector_simulator')
        qnn = QuantumNetworkCircuit(config, 4, [1, 1, 1, 1])
        gradient_list = calculate_gradient_list(qnn, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertTrue(any([abs(grad) > 1e-10 for grad in gradient_list.flatten()]),
                        'For a normal circuit, at least one gradient should not be zero')

    def test_when_calculate_gradient_list_with_parameter_shift_then_non_zero_gradient(self):
        config = Config('vector', ansatz_type='sim_circ_13_half', layers=1, backend_type='statevector_simulator')
        qnn = QuantumNetworkCircuit(config, 4, [1, 1, 1, 1])
        gradient_list = calculate_gradient_list(qnn, [1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9], method='parameter shift',
                                                eps=0.01)
        self.assertTrue(any([abs(grad) > 1e-5 for grad in gradient_list.flatten()]),
                        'For a normal circuit, at least one gradient should not be zero')

    def test_given_input_zero_and_no_ansatz_when_calculate_gradient_then_zero_gradient(self):
        config = Config('vector', backend_type='statevector_simulator')
        input = [0, 0]
        qnn = QuantumNetworkCircuit(config, 2, input)
        gradient_list = calculate_gradient_list(qnn, input)
        self.assertTrue((np.zeros_like(gradient_list) == gradient_list).all(), 'Circuit should have zero gradient')

    def test_given_input_zero_and_no_ansatz_when_calculate_gradient_with_parameter_shift_then_zero_gradient(self):
        config = Config('vector', backend_type='statevector_simulator')
        input = [0, 0]
        qnn = QuantumNetworkCircuit(config, 2, input)
        gradient_list = calculate_gradient_list(qnn, input, method='parameter shift', eps=0.01)
        self.assertTrue((np.zeros_like(gradient_list) == gradient_list).all(), 'Circuit should have zero gradient')

    def test_given_input_pi_and_no_ansatz_when_calculate_gradient_then_zero_gradient(self):
        config = Config('vector', backend_type='statevector_simulator')
        input = [np.pi, np.pi]
        qnn = QuantumNetworkCircuit(config, 2, input)
        gradient_list = calculate_gradient_list(qnn, input)
        self.assertTrue((np.isclose(gradient_list, 0).all()), 'Circuit should have zero gradient')

    def test_given_non_zero_input_and_no_ansatz_when_calculate_gradient_then_expected_gradient(self):
        config = Config('vector', backend_type='statevector_simulator')
        input = [1]
        qnn = QuantumNetworkCircuit(config, 1, input)
        gradient_list = calculate_gradient_list(qnn, input)
        self.assertAlmostEqual(gradient_list[0][0], -np.sin(1), delta=1e-6)

    def test_given_non_zero_input_and_no_ansatz_when_calculate_gradient_with_qasm_simulator_then_expected_gradient(
            self):
        config = Config('vector', backend_type='qasm_simulator')
        input = [1]
        qnn = QuantumNetworkCircuit(config, 1, input)
        gradient_list = calculate_gradient_list(qnn, input, shots=10000)
        self.assertAlmostEqual(gradient_list[0][0], -np.sin(1), delta=1e-2)

    def test_given_non_zero_input_and_no_ansatz_when_calculate_gradient_with_parameter_shift_then_expected_gradient(
            self):
        config = Config('vector', backend_type='statevector_simulator')
        input = [1]
        qnn = QuantumNetworkCircuit(config, 1, input)
        gradient_list = calculate_gradient_list(qnn, input, method='parameter shift', eps=0.0001)
        self.assertAlmostEqual(gradient_list[0][0], -np.sin(1), delta=1e-6)

    def test_given_non_zero_input_and_no_ansatz_when_calculate_gradient_with_parameter_shift_and_qasm_simulator_then_expected_gradient(
            self):
        config = Config('vector', backend_type='qasm_simulator')
        input = [1]
        qnn = QuantumNetworkCircuit(config, 1, input)
        gradient_list = calculate_gradient_list(qnn, input, method='parameter shift', eps=0.0001, shots=10000)
        self.assertAlmostEqual(gradient_list[0][0], -np.sin(1), delta=1e-2)

    def test_given_product_circuit_when_calculate_gradient_then_offdiagonal_gradient_zero(self):
        config = Config('vector', backend_type='statevector_simulator')
        input = [1, 2, 3, 4]
        qnn = QuantumNetworkCircuit(config, 4, input)
        gradient_list = calculate_gradient_list(qnn, input)

        for i in range(len(gradient_list)):
            for j in range(len(gradient_list[0])):
                if i != j:
                    self.assertAlmostEqual(gradient_list[i][j], 0, delta=1e-10,
                                           msg='For a circuit without entangling gates, each measured qubit should have'
                                               ' zero gradient with respect to rotations on a different qubit')

    def test_given_sim_circ_13_type_ansatz_when_calculate_gradient_then_final_crz_layer_has_zero_gradient(self):
        config = Config('vector', ansatz_type='sim_circ_13', layers=1, backend_type='statevector_simulator')
        qnn = QuantumNetworkCircuit(config, 4, [1, 1, 1, 1])
        gradient_list = calculate_gradient_list(qnn,
                                                [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.assertTrue(all([abs(grad) < 1e-10 for grad in gradient_list[-4:].flatten()]),
                        'For sim_cir_13, final crz layer should have zero gradient wrt measurement.')

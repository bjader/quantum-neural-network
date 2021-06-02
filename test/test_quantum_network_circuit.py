import copy
import sys

sys.path.append('quantum-neural-network')
from unittest import TestCase

from qiskit import QuantumCircuit, Aer, execute

from config import Config
from quantum_network_circuit import QuantumNetworkCircuit

default_config = Config('vector', ansatz_type='sim_circ_13_half', layers=1)


class TestQuantumNetworkCircuit(TestCase):

    def test_when_create_qnn_then_happy_path(self):
        for ansatz in ['sim_circ_13', 'sim_circ_13_half', 'farhi', 'alternating_layer_tdcnot', 'sim_circ_15',
                       'sim_circ_19', 'abbas', 'null']:
            for af in ['null', None, 'partial_measurement_half', 'partial_measurement_2']:
                for encoding in ['vector', 'havlicek']:
                    QuantumNetworkCircuit(Config(encoding, ansatz_type=ansatz, activation_function_type=af), 4,
                                          [1, 1, 1, 1])

    def test_when_create_qnn_then_input_parameters_ordered(self):
        qnn = QuantumNetworkCircuit(copy.deepcopy(default_config), 12, input_data=[i for i in range(12)])
        expected = ['input{}'.format(i) for i in range(12)]
        self.assertEqual(expected, [p.name for p in qnn.input_circuit_parameters])

    def test_when_create_qnn_then_ansatz_parameters_ordered(self):
        qnn = QuantumNetworkCircuit(copy.deepcopy(default_config), input_qubits=6)
        expected = ['ansatz{}'.format(i) for i in range(12)]
        for i in range(len(qnn.ansatz_circuit_parameters)):
            self.assertEqual(expected[i], qnn.ansatz_circuit_parameters[i].name)

    def test_when_construct_network_then_not_empty(self):
        qnn = QuantumNetworkCircuit(copy.deepcopy(default_config), 2)

        qnn.construct_network([1, 1])

        self.assertTrue(qnn.qc.depth() > 0)

    def test_when_bind_circuits_then_no_parameterised_gates_left(self):
        qnn = QuantumNetworkCircuit(copy.deepcopy(default_config), 2, [1, 1])
        params = [1, 1, 2, 2, 2, 2]
        bound_qc = qnn.bind_circuit(params)

        self.assertEqual(bound_qc.num_parameters, 0)

    def test_evaluate_circuit(self):
        pass

    def test_given_qasm_simulator_when_hadamard_circuit_then_correct_values(self):
        qc = QuantumCircuit(5)
        qc.h(qc.qregs[0])
        qc.measure_all()
        result = execute(qc, backend=Aer.get_backend('qasm_simulator'), shots=10000).result()

        vector = QuantumNetworkCircuit.get_vector_from_results(result)

        self.assertTrue(len(vector) == 5)
        for i in range(5):
            self.assertAlmostEqual(0.0, vector[i], delta=0.5)

    def test_given_statevector_simulator_when_hadamard_circuit_then_correct_values(self):
        qc = QuantumCircuit(5)
        qc.h(qc.qregs[0])
        result = execute(qc, backend=Aer.get_backend('statevector_simulator'), shots=1).result()

        vector = QuantumNetworkCircuit.get_vector_from_results(result)

        self.assertTrue(len(vector) == 5)
        for i in range(5):
            self.assertAlmostEqual(0.0, vector[i], delta=1e-10)

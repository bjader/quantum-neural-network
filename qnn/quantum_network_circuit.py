import logging
from math import log

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, transpile
from qiskit.aqua.utils import get_subsystems_counts
from qiskit.quantum_info import Pauli, Statevector

from config import Config

logging.basicConfig(level=logging.INFO)
logging.getLogger('qiskit').setLevel(logging.WARN)


class QuantumNetworkCircuit:
    """
    A quantum neural network. Combines state preparation circuit and variational ansatz to produce quantum neural
    network circuit.
    """

    def __init__(self, config: Config, input_qubits, input_data=None):

        self.config = config
        self.input_qubits = input_qubits
        self.input_data = input_data
        self.input_circuit_parameters = None

        self.ansatz_circuit = self._create_ansatz_circuit(input_qubits)

        self.ansatz_circuit_parameters = sorted(list(self.ansatz_circuit.parameters),
                                                key=lambda p: int(''.join(filter(str.isdigit, p.name))))

        self.qr = QuantumRegister(self.ansatz_circuit.num_qubits, name='qr')
        self.cr = ClassicalRegister(len(self.qr), name='cr')
        self.qc = QuantumCircuit(self.qr, self.cr)

        self.backend = config.backend

        if input_data is not None:
            self.construct_network(input_data)

        self.statevectors = []
        self.gradients = []
        self.transpiled = False

    def create_input_circuit(self):
        return self.config.data_handler.get_quantum_circuit(self.input_data)

    def _create_ansatz_circuit(self, input_qubits):
        return self.config.ansatz.get_quantum_circuit(input_qubits)

    def construct_network(self, input_data):
        self.input_data = input_data
        input_circuit = self.config.data_handler.get_quantum_circuit(input_data)
        self.input_circuit_parameters = sorted(list(input_circuit.parameters),
                                               key=lambda p: int(''.join(filter(str.isdigit, p.name))))

        self.qc.append(input_circuit, self.qr[:input_circuit.num_qubits])
        self.qc = self.qc.combine(self.ansatz_circuit)

        if self.backend.name() is not 'statevector_simulator':
            if self.config.meas_method == 'ancilla':
                self.qc.measure(self.qr[-1], self.cr[0])

            elif self.config.meas_method == 'all':
                self.qc.measure(self.qr, self.cr)

        logging.info("QNN created with {} trainable parameters.".format(len(self.ansatz_circuit_parameters)))
        self.config.log_info()

    def bind_circuit(self, parameter_values):
        """
        Assigns all parameterized gates to values
        :param parameter_values: List of parameter values for circuit. Input parameters should come before ansatz
        parameters.
        """
        if self.input_circuit_parameters is None:
            raise NotImplementedError(
                "No input data was specified before binding. Please call construct_network() first.")
        combined_parameter_list = self.input_circuit_parameters + self.ansatz_circuit_parameters
        if len(parameter_values) != len(combined_parameter_list):
            raise ValueError('Parameter_values must be of length {}'.format(len(combined_parameter_list)))

        binding_dict = {}
        for i, value in enumerate(parameter_values):
            binding_dict[combined_parameter_list[i]] = value

        bound_qc = self.qc.bind_parameters(binding_dict)
        return bound_qc

    def evaluate_circuit(self, parameter_list, shots=100):
        # if self.transpiled is False:
        #     self.qc = transpile(self.qc, optimization_level=0, basis_gates=['cx', 'u1', 'u2', 'u3'])
        #     self.transpiled = True
        circuit = self.bind_circuit(parameter_list)
        job = execute(circuit, backend=self.backend, shots=shots)
        return job.result()

    @staticmethod
    def get_vector_from_results(results, circuit_id=0):
        """
        Calculates the expectation value of individual qubits for a set of observed bitstrings. Assumes counts
        corresponding to classical  register used for final measurement is final element in job counts array in
        order to exclude classical registers used for activation function measurements (if present).
        :param results: Qiskit results object.
        :param circuit_id: For results of multiple circuits, integer labelling which circuit result to use.
        :return: A vector, where the ith element is the expectation value of the ith qubit
        """

        if results.backend_name == 'statevector_simulator':
            state = results.get_statevector(circuit_id)

            n = int(log(len(state), 2))
            vector = [Statevector(state).expectation_value(Pauli.pauli_single(n, i, 'Z')).real for i in range(n)]
            return vector

        else:
            counts = results.get_counts(circuit_id)
            all_register_counts = get_subsystems_counts(counts)
            output_register_counts = all_register_counts[-1]
            num_measurements = len(next(iter(output_register_counts)))
            vector = np.zeros(num_measurements)

            for counts, frequency in output_register_counts.items():
                for i in range(num_measurements):
                    if counts[i] == '0':
                        vector[i] += frequency
                    elif counts[i] == '1':
                        vector[i] -= frequency
                    else:
                        raise ValueError("Measurement returned unrecognised value")

            return vector / (sum(output_register_counts.values()))

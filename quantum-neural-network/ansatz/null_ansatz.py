from qiskit import QuantumCircuit, QuantumRegister

from ansatz.variational_ansatz import VariationalAnsatz


class NullAnsatz(VariationalAnsatz):
    def add_rotations(self, n_data_qubits):
        pass

    def add_entangling_gates(self, n_data_qubits):
        pass

    def get_quantum_circuit(self, n_data_qubits):
        self.qr = QuantumRegister(n_data_qubits, name='qr')
        self.qc = QuantumCircuit(self.qr, name='Shifted circ')
        return self.qc

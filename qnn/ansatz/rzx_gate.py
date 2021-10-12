import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.extensions import CXGate, U3Gate, U1Gate, U2Gate


class RZXGate(Gate):
    """Two-qubit XZ-rotation gate. Modified XX-rotation gate from Qiskit.

    This gate corresponds to the rotation U(θ) = exp(-1j * θ * Z⊗X / 2)
    """

    def __init__(self, theta):
        """Create new rzx gate."""
        super().__init__("rzx", 2, [theta])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""

        definition = []
        q = QuantumRegister(2, "q")
        theta = self.params[0]
        rule = [
            (U3Gate(np.pi / 2, theta, 0), [q[0]], []),
            (CXGate(), [q[0], q[1]], []),
            (U1Gate(-theta), [q[1]], []),
            (CXGate(), [q[0], q[1]], []),
            (U2Gate(-np.pi, np.pi - theta), [q[0]], []),
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return RZXGate(-self.params[0])


def rzx(self, theta, qubit1, qubit2):
    """Apply RZX to circuit."""
    return self.append(RZXGate(theta), [qubit1, qubit2], [])


# Add to QuantumCircuit class
QuantumCircuit.rzx = rzx

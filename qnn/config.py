import logging

from qiskit import IBMQ, Aer
from qiskit.providers import QiskitBackendNotFoundError
from qiskit.providers.ibmq import IBMQAccountError
from qiskit.providers.ibmq.api.exceptions import AuthenticationLicenseError

from activation_function.activation_function import ActivationFunction
from activation_function.activation_function_factory import ActivationFunctionFactory
from ansatz.variational_ansatz import VariationalAnsatz
from ansatz.variational_ansatz_factory import VariationalAnsatzFactory
from input.data_handler import DataHandler
from input.data_handler_factory import DataHandlerFactory


class Config:
    def __init__(self, encoding, data_handler_method=None, ansatz_type=None, layers=3, sweeps_per_layer=1,
                 activation_function_type=None, meas_method='all', grad_method='parameter shift',
                 backend_type='qasm_simulator'):
        """
        :param encoding:
        :param data_handler_method:
        :param ansatz_type:
        :param layers: Number of layers in variational ansatz
        :param sweeps_per_layer: Number of sweeps of parameterised gates within a single layer
        :param activation_function_type: Valid types are null, 'partial_measurement_half or 'partial_measurement_X'
        where X is an integer giving the number of measurements for the activation function.
        :param meas_method: Valid methods are 'all' or 'ancilla'
        :param grad_method:
        """
        self.encoding = encoding
        self.data_handler_method = data_handler_method
        self.ansatz_type = ansatz_type
        self.layers = layers
        self.sweeps_per_layer = sweeps_per_layer
        self.activation_function_type = activation_function_type
        self.meas_method = meas_method
        self.grad_method = grad_method
        self.backend = self.get_backend(backend_type)
        self.data_handler: DataHandler = DataHandlerFactory(encoding, data_handler_method).get()
        self.activation_function: ActivationFunction = ActivationFunctionFactory(activation_function_type).get()
        self.ansatz: VariationalAnsatz = VariationalAnsatzFactory(ansatz_type, layers, sweeps_per_layer,
                                                                  self.activation_function).get()

    def log_info(self):
        logging.info(f"QNN configuration:\nencoding = {self.encoding}" +
                     f"\ndata handler = {self.data_handler_method}" +
                     f"\nansatz = {self.ansatz_type}" +
                     f"\nnumber of layers = {self.layers}" +
                     f"\nnumber of sweeps per layer = {self.sweeps_per_layer}" +
                     f"\nactivation function = {self.activation_function_type}" +
                     f"\noutput measurement = {self.meas_method}" +
                     f"\ngradient method = {self.grad_method}" +
                     f"\nsimulation backend = {self.backend}")

    def get_backend(self, backend_name):
        backend = None
        if backend_name not in ['qasm_simulator', 'statevector_simulator']:
            try:
                IBMQ.load_account()
                oxford_provider = IBMQ.get_provider(hub='ibm-q-oxford')
                backend = oxford_provider.get_backend(backend_name)
            except (IBMQAccountError, AuthenticationLicenseError):
                logging.warning("Unable to connect to IBMQ servers.")
                pass
            except QiskitBackendNotFoundError:
                logging.debug('{} is not a valid online backend, trying local simulators.'.format(backend_name))
                pass
        if backend is None:
            backend = Aer.get_backend(backend_name)
        return backend

from activation_function.null_activation_function import NullActivationFunction
from activation_function.partial_meas_activation_function import PartialMeasActivationFunction


class ActivationFunctionFactory:

    def __init__(self, activation_function_type):
        self.activation_function_type = activation_function_type

    def get(self):
        """
        Returns appropriate activation function object.
        """

        if self.activation_function_type == 'null' or self.activation_function_type == None:
            return NullActivationFunction()
        elif self.activation_function_type == 'partial_measurement_half':
            return PartialMeasActivationFunction("half")

        elif self.activation_function_type[:len('partial_measurement_')] == 'partial_measurement_':
            n_measurements = int(self.activation_function_type[len('partial_measurement_'):])
            return PartialMeasActivationFunction(n_measurements)

        else:
            raise ValueError("Invalid activation function type.")

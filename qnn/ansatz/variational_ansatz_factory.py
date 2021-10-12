from ansatz.abbas import Abbas
from ansatz.alternating_layer_tdcnot_ansatz import AlternatingLayerTDCnotAnsatz
from ansatz.farhi_ansatz import FarhiAnsatz
from ansatz.null_ansatz import NullAnsatz
from ansatz.sim_circ_13 import SimCirc13
from ansatz.sim_circ_13_half import SimCirc13Half
from ansatz.sim_circ_14 import SimCirc14
from ansatz.sim_circ_14_half import SimCirc14Half
from ansatz.sim_circ_15 import SimCirc15
from ansatz.sim_circ_19 import SimCirc19


class VariationalAnsatzFactory:

    def __init__(self, ansatz_type, layers, sweeps_per_layer, activation_function):
        self.ansatz_type = ansatz_type
        self.layers = layers
        self.sweeps_per_layer = sweeps_per_layer
        self.activation_function = activation_function

    def get(self):
        """
        Returns the appropriate ansatz circuit
        """

        if self.ansatz_type == 'farhi':
            return FarhiAnsatz(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'alternating_layer_tdcnot':
            return AlternatingLayerTDCnotAnsatz(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'sim_circ_13_half':
            return SimCirc13Half(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'sim_circ_13':
            return SimCirc13(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'sim_circ_14_half':
            return SimCirc14Half(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'sim_circ_14':
            return SimCirc14(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'sim_circ_15':
            return SimCirc15(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'sim_circ_19':
            return SimCirc19(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type == 'abbas':
            return Abbas(self.layers, self.sweeps_per_layer, self.activation_function)

        elif self.ansatz_type is None or 'null':
            return NullAnsatz(self.layers, self.sweeps_per_layer, self.activation_function)

        else:
            raise ValueError("Invalid ansatz type: {}".format(self.ansatz_type))

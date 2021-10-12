from input.frqi_bennett_data_handler import FRQIBennettDataHandler
from input.havlicek_data_handler import HavlicekDataHandler
from input.neqr_bennett_data_handler import NEQRBennettDataHandler
from input.neqr_sv_data_handler import NEQRSVDataHandler
from input.vector_data_handler import VectorDataHandler


class DataHandlerFactory:

    def __init__(self, encoding, method):
        self.encoding = encoding
        self.method = method

    def get(self):
        """
        Returns the appropriate data handler
        """
        if self.encoding == 'vector':
            return VectorDataHandler()
        if self.encoding == 'havlicek':
            return HavlicekDataHandler()
        if self.method is None:
            raise ValueError("This encoding requires a data handler method.")
        elif self.encoding == 'frqi':
            if self.method == 'bennett':
                return FRQIBennettDataHandler()
        elif self.encoding == 'neqr':
            if self.method == 'statevector':
                return NEQRSVDataHandler()
            elif self.method == 'bennett':
                return NEQRBennettDataHandler()
        else:
            raise ValueError("Invalid string used for encoding or data handler method.")

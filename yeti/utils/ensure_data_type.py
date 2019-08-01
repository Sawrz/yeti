import numpy as np

from yeti.systems.atom import Atom
from yeti.systems.residue import Residue


class EnsureDataTypes(object):
    def __init__(self, exception_class):
        self.exception_class = exception_class

    def __check_type__(self, parameter, parameter_name, data_type):
        if type(parameter) != data_type:
            msg = 'Wrong data type for parameter "{name}". Desired type is {data_type}'.format(
                name=parameter_name, data_type=data_type.__name__)
            raise self.exception_class(msg)

    def __check_numpy_dimensions__(self, parameter, parameter_name, desired_shape):
        actual_shape = parameter.shape
        msg = 'Wrong shape for parameter "{name}". Desired shape: {des_shape}.'.format(name=parameter_name,
                                                                                       des_shape=desired_shape)

        if len(actual_shape) != len(desired_shape):
            self.exception_class(msg)

        for actual_dim, desired_dim in zip(actual_shape, desired_shape):
            if desired_dim is None:
                continue
            else:
                if actual_dim != desired_dim:
                    raise self.exception_class(msg)

    def ensure_integer(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=int)

    def ensure_string(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=str)

    def ensure_atom(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=Atom)

    def ensure_residue(self, parameter, parameter_name):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=Residue)

    def ensure_numpy_array(self, parameter, parameter_name, shape):
        self.__check_type__(parameter=parameter, parameter_name=parameter_name, data_type=np.ndarray)
        self.__check_numpy_dimensions__(parameter=parameter, parameter_name=parameter_name, desired_shape=shape)

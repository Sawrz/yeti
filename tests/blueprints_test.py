import unittest


class BlueprintTestCase(unittest.TestCase):
    pass


class BlueprintExceptionsTestCase(unittest.TestCase):
    @staticmethod
    def create_data_type_exception_messages(parameter_name, data_type_name):
        return 'Wrong data type for parameter "{name}". Desired type is {data_type}'.format(name=parameter_name,
                                                                                            data_type=data_type_name)

    @staticmethod
    def create_array_shape_exception_messages(parameter_name, desired_shape):
        return 'Wrong shape for parameter "{name}". Desired shape: {des_shape}.'.format(name=parameter_name,
                                                                                        des_shape=desired_shape)

    @staticmethod
    def create_array_dtype_exception_messages(parameter_name, dtype_name):
        return 'Wrong dtype for ndarray "{name}". Desired dtype is {data_type}'.format(name=parameter_name,
                                                                                       data_type=dtype_name)



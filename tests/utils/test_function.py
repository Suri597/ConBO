import pickle
import numpy as np
import pickle
import unittest

from partxv2.sampling import lhs_sampling, uniform_sampling
from partxv2.utils import Fn


class TestFunction(unittest.TestCase):
    def test1_uniform_sampling(self):
        region_support = np.array([[-1, 1], [-3, -2]])
        tf_dim = 2
        num_samples = 10
        seed = 10001
        rng = np.random.default_rng(seed)
        samples_in_unif = uniform_sampling(num_samples, region_support, tf_dim, rng)

        def test_function(X):
            return X[0] ** 2 + X[1] ** 2

        decFunction = Fn(test_function)
        for x in samples_in_unif:
            decFunction(x)

        assert decFunction.count == num_samples
        np.testing.assert_array_equal(
            np.stack(np.array(decFunction.point_history, dtype=object)[:, 1]),
            samples_in_unif,
        )

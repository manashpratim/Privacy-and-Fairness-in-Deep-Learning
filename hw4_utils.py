import importlib
import unittest
from collections import namedtuple
from datetime import datetime
import inspect
import gzip
import pickle as pkl
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import warnings
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

Splits = namedtuple("Splits", ["train", "valid", "test"])
Domains = namedtuple("Domains", ["X", "Y"])


def accuracy(model, x, y):
    return np.mean(
        np.argmax(
            model.predict(x, verbose=1), axis=1
        ) == y
    )


def find_file(filename):
    for d in ['.', '..', '../..', '/autograder/source/tests']:
        try:
            path = os.path.join(d, filename)
            assert (os.path.exists(path) and os.access(path, os.R_OK))
            break
        except Exception:
            continue

    assert(os.path.exists(path) and os.access(path, os.R_OK))

    return path


def load_mnist():
    path = find_file("mnist.pkl.gz")

    with gzip.open(path, 'rb') as f:
        (train_X, train_y), \
        (val_X, val_y), \
        (test_X, test_y) = pkl.load(f, encoding='latin1')

    return Splits(
        Domains(train_X, train_y),
        Domains(val_X, val_y),
        Domains(test_X, test_y)
    )


def timeit(thunk):
    start = datetime.now()

    ret = thunk()

    return ret, datetime.now() - start


def seed(seedh=0):
    np.random.seed(seedh)
    tf.compat.v1.set_random_seed(seedh)


def exercise(andrew_username=None, seed=None):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    cname = inspect.getouterframes(inspect.currentframe(), 2)[1][3]

    seedh = hash(hash(andrew_username) + hash(cname) + hash(seed)) % 0xffffffff

    seed(seedh)

    print(f"=== EXERCISE {cname} {andrew_username} {seed:x} {seedh:x} ===")


class Dataset(object):
    """Dataset Dataset object of features and labels
        The label of the data is optinal

        Arguments:
            X {np.ndarray} -- features

        Keyword Arguments:
            y {np.ndarray} -- labels (default: {None})
            batch_size {int} -- size of mini-batch (default: {16})
            shuffle {bool} -- if True, shuffle the data (default: {False})
    """
    def __init__(self, X, y=None, batch_size=16, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.size = X.shape[0] // batch_size
        if self.X.shape[0] % batch_size:
            self.size += 1
        if shuffle:
            index = np.random.permutation(X.shape[0])
            self.X = self.X[index]
            if self.y is not None:
                self.y = self.y[index]

    def __getitem__(self, i):
        if i == self.size - 1:
            features = self.X[i * self.batch_size:]
            if self.y is not None:
                labels = self.y[i * self.batch_size:]
        else:
            features = self.X[i * self.batch_size:(i + 1) * self.batch_size]
            if self.y is not None:
                labels = self.y[i * self.batch_size:(i + 1) * self.batch_size]

        if self.y is not None:
            return (features, labels)
        return features

    def __len__(self):
        return self.size


class Dataloader(object):
    """Dataloader A batch generator that iterates the whole dataset.
        The label of the data is optinal

        Arguments:
            X {np.ndarray} -- dataset

        Keyword Arguments:
            y {np.ndarray} -- labels (default: {None})
            batch_size {int} -- The size of each mini batch (default: {16})
            shuffle {bool} -- If True, shuffle the dataset everytime before
               iteration
    """
    def __init__(self, X, y=None, batch_size=16, shuffle=False):
        """__Constructor__

        Arguments:
            X {np.array} -- Input dataset

        Keyword Arguments:
            y {np.array} -- labels. If None, do not iterate  (default: {None})
            batch_size {int} -- batch size (default: {16})
            shuffle {bool} -- If True, shuffle the dataset (default: {False})
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.dataset = Dataset(
            self.X,
            y=self.y,
            batch_size=self.batch_size,
            shuffle=self.shuffle)

        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.dataset):
            batch = self.dataset[self.n]
            self.n += 1
            return batch
        else:
            raise StopIteration


def batched_map(X, Y=None, f=None, batch_size=16, tqdm=tqdm, **kwargs):
    if Y is not None:
        dataloader = Dataloader(X, Y, batch_size=batch_size)
    else:
        dataloader = Dataloader(X, batch_size=batch_size)

    parts = []

    for batch in tqdm(
        dataloader,
        unit="batches",
        total=len(data[0])//batch_size,
        **kwargs
    ):
        if Y is not None:
            parts.append(*f(*batch))

        else:
            parts.append(*f(batch))

    return [np.hstack(part) for part in parts]


def norms(
    a,
    ords=[0, 1, 2, float('inf')],
    ordstrs=[0, 1, 2, "\\infty"]
) -> str:
    ret = [np.linalg.norm(a, ord=o, axis=1).mean() for o in ords]
    ret_str = ", ".join(
            [
             f"$L_{o}$={d:0.4f}"
             for o, d in zip(ordstrs, ret)
            ]
        )

    return ret, ret_str


class TestCase(unittest.TestCase):
    def setUp(self):
        if self.module_name:
            self.solution = importlib.import_module(self.module_name)

        unittest.TestCase.setUp(self)

    @staticmethod
    def assertArrayEqual(solution, submission, msg='Arrays not equal.'):
        return np.testing.assert_array_almost_equal(
            solution, submission,
            decimal=6,
            err_msg=msg
        )

    def assertShapeEqual(self, solution, submission, msg='Shapes not equal.'):
        return self.assertTupleEqual(
            solution, submission,
            msg=msg
        )

    def assertNonEmpty(self, l, msg=None):
        if msg is None:
            msg = "Value is empty."
        return self.assertTrue(l, msg=msg)

    def assertHasAttribute(self, obj, attr, msg=None):
        if msg is None:
            msg = f"Attribute {attr} not found."
        return self.assertTrue(
            hasattr(obj, attr),
            msg=msg
        )


def read_file(filename):
    if not (os.path.exists(filename) and os.access(filename, os.R_OK)):
        return None

    with open(filename, 'r') as f:
        return "".join(list(f))


def run_tests(testclass):
    tests = unittest.TestLoader().loadTestsFromTestCase(testclass)

    f = open(os.devnull, "w")

    runner = unittest.TextTestRunner(stream=f,
                                     descriptions=False,
                                     verbosity=0,
                                     buffer=False)

    for test in tests:
        print(f"=== BEGIN {test} ===")
        res = runner.run(test)

        bads = [m[1] for m in [*res.errors, *res.failures]]

        if len(bads) == 0:
            print("OK")
        else:
            print("FAIL")

        for bad in bads:
            print(bad)

        print(f"=== END {test} ===")

    f.close()



# https://github.com/tensorflow/cleverhans/blob/f1d233688e5aea473c93d7afdc04910292f2c2b3/cleverhans/utils_tf.py#L358
def clip_eta(eta, ord, eps):
  """
  Helper function to clip the perturbation to epsilon norm ball.
  :param eta: A tensor with the current perturbation.
  :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
  :param eps: Epsilon, bound of the perturbation.
  """

  # Clipping perturbation eta to self.ord norm ball
  if ord not in [np.inf, 1, 2]:
    raise ValueError('ord must be np.inf, 1, or 2.')
  reduc_ind = list(range(1, len(eta.get_shape())))
  avoid_zero_div = 1e-12
  if ord == np.inf:
    eta = clip_by_value(eta, -eps, eps)
  elif ord == 1:
    # Implements a projection algorithm onto the l1-ball from
    # (Duchi et al. 2008) that runs in time O(d*log(d)) where d is the
    # input dimension.
    # Paper link (Duchi et al. 2008): https://dl.acm.org/citation.cfm?id=1390191

    eps = tf.cast(eps, eta.dtype)

    dim = tf.reduce_prod(tf.shape(eta)[1:])
    eta_flat = tf.reshape(eta, (-1, dim))
    abs_eta = tf.abs(eta_flat)

    if 'sort' in dir(tf):
      mu = -tf.sort(-abs_eta, axis=-1)
    else:
      # `tf.sort` is only available in TF 1.13 onwards
      mu = tf.nn.top_k(abs_eta, k=dim, sorted=True)[0]
    cumsums = tf.cumsum(mu, axis=-1)
    js = tf.cast(tf.divide(1, tf.range(1, dim + 1)), eta.dtype)
    t = tf.cast(tf.greater(mu - js * (cumsums - eps), 0), eta.dtype)

    rho = tf.argmax(t * cumsums, axis=-1)
    rho_val = tf.reduce_max(t * cumsums, axis=-1)
    theta = tf.divide(rho_val - eps, tf.cast(1 + rho, eta.dtype))

    eta_sgn = tf.sign(eta_flat)
    eta_proj = eta_sgn * tf.maximum(abs_eta - theta[:, tf.newaxis], 0)
    eta_proj = tf.reshape(eta_proj, tf.shape(eta))

    norm = tf.reduce_sum(tf.abs(eta), reduc_ind)
    eta = tf.where(tf.greater(norm, eps), eta_proj, eta)

  elif ord == 2:
    # avoid_zero_div must go inside sqrt to avoid a divide by zero
    # in the gradient through this operation
    norm = tf.sqrt(tf.maximum(avoid_zero_div,
                              tf.reduce_sum(tf.square(eta),
                                            reduc_ind,
                                            keepdims=True)))
    # We must *clip* to within the norm ball, not *normalize* onto the
    # surface of the ball
    factor = tf.minimum(1., tf.div(eps, norm))
    eta = eta * factor
  return eta

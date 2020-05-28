# hw4_mnist.py

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import \
    Conv2D, Activation, MaxPooling2D, Flatten, Dense

import hw4_utils
from hw4_utils import Splits


def load_data():
    """Load data and reshape to 28x28x1 (1 channel). Training, validation, test
       data is then splits.train, splits.valid, splits.test . Within a split,
       inputs are split.X and outputs are split.Y.
    """

    data = Splits(*[
        split._replace(
            X=np.reshape(split.X, (split.X.shape[0], 28, 28, 1))
        )
        for split in hw4_utils.load_mnist()])

    return data


class WithSession(object):
    def __init__(self, session: tf.Session = None):
        if session is None:
            session = tf.keras.backend.get_session()

        self.set_session(session)

    def set_session(self, session: tf.Session):
        self.session = session
        tf.keras.backend.set_session(session)

    def reset_session(self):
        tf.reset_default_graph()
        self.set_session(tf.keras.backend.get_session())

    def initialize_variables(self):
        self.session.run(tf.global_variables_initializer())


class HW4Model(WithSession):
    def __init__(self, session: tf.Session = None):
        WithSession.__init__(self, session=session)

        # Instantiations of this class should provide these attributes:
        self.X: Input = None
        self.Ytrue: Input = None

        self.tensors: dict = None

        self.logits: Model = None
        self.probits: Model = None
        self.preds: Model = None

        self.f_logits: K.Function = None
        self.f_probits: K.Function = None
        self.f_preds: K.Function = None

        self.model: Model = None


class MNISTModel(HW4Model):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)

        self.num_classes: int = 10

        # note to make this model usable, one must first call either build() or load()

    def build(self):
        # Running this will reset the model's parameters
        layers = []

        layers.append(Conv2D(
            filters=16,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            activation='relu'
        ))

        layers.append(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        layers.append(Conv2D(
            filters=32,
            kernel_size=(4, 4),
            padding='same',
            activation='relu'
        ))

        layers.append(MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2)
        ))

        layers.append(Flatten())

        layers.append(
            Dense(64, activation='relu')
        )

        # layers[-2]
        layers.append(Dense(
            units=self.num_classes,
            name="logits"
        ))

        # layers[-1]
        layers.append(Activation('softmax'))

        self.layers = layers

        self._define_ops()

    def _define_ops(self):
        self.X = Input(shape=(28, 28, 1))
        self.Ytrue = Input(shape=(self.num_classes), dtype=tf.int32)

        self.tensors = self.forward(self.X, self.Ytrue)

        self.model = Model(self.X, self.tensors['probits'])

        # models can be symbolically composed
        self.logits = Model(self.X, self.tensors['logits'])
        self.probits = Model(self.X, self.tensors['probits'])
        self.preds = Model(self.X, self.tensors['preds'])
     

        # functions evaluate to concrete outputs
        self.f_logits = K.function(self.X, self.tensors['logits'])
        self.f_probits = K.function(self.X, self.tensors['probits'])
        self.f_preds = K.function(self.X, self.tensors['preds'])


    def forward(self, X, Ytrue=None):
        # Define various tensors that make up the model and potentially its
        # loss (if Ytrue is given).

        _logits: tf.Tensor = None
        _probits: tf.Tensor = None
        _preds: tf.Tensor = None
        _loss: tf.Tensor = None

        c = X
        parts = []
        for l in self.layers:
            c = l(c)
            parts.append(c)

        _logits = parts[-2]
        _probits = parts[-1]

        _preds = tf.argmax(_probits, axis=1)

        if Ytrue is not None:
            # Same as the loss specified in train below.
            _loss = K.mean(K.sparse_categorical_crossentropy(
                self.Ytrue,
                _probits
            ))

        return {
            'logits': _logits,
            'probits': _probits,
            'preds': _preds,
            'loss': _loss,
        }

    def train(self, X, Y, epochs=1, batch_size=16, **kwargs):
        self.model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy']
        )

        self.model.fit(
            X, Y,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )

    def load(self, batch_size=16, filename=None):
        if filename is None:
            filename = f"model.MNIST.h5"

        f = hw4_utils.find_file(filename)

        self.build()

        model = Sequential(self.layers)
        model.build(input_shape=(batch_size, 28, 28, 1))
        model.load_weights(f)
        self.layers = model.layers

        self._define_ops()

    def save(self, filename=None):
        if filename is None:
            filename = f"model.MNIST.h5"

        model = Sequential(self.layers)
        model.save(filename)

    def _prepare_model(self, seed=0):
        data = load_data()

        hw4_utils.seed(hash(seed))

        self.build()
        self._define_ops()

        self.train(
            data.train.X,
            data.train.Y,
            validation_data=data.valid,
            epochs=1
        )

        self.save()

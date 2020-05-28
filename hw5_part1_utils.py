import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import regularizers
from numpy.random import permutation

from tqdm import tqdm

import numpy as np

# YOU DO NOT NEED TO MODIFY THIS SCRIPT. ######################################


class DataSplit(object):
    '''Utility class for splitting shadow data into 'in' and 'out' sets.'''

    def __init__(self, labels, seed):
        '''Constructor

        :labels - the labels of the data we would like to split. This allows us
           to ensure we get a good representation of all classes in the split.

        :seed - random seed for creating the split.

        The resulting object has two (numpy array) lists, in_idx and out_idx,
        which contain the indices of the 'in' set and 'out' set respectively.
        '''

        C = labels.max() + 1

        np.random.seed(seed)

        idx = np.arange(len(labels))

        all_in_idx = []
        all_out_idx = []
        for c in range(C):
            n = len(labels[labels == c])
            split_indices = permutation(n)
            in_idx = split_indices[:n // 2]
            
            
            #in_idx = split_indices[:int(0.7*n)]
            out_idx = split_indices[n // 2:]
            #out_idx = split_indices[int(0.7*n):]
            #print(len(in_idx))
            #print(len(out_idx))

            all_in_idx.append(idx[labels == c][in_idx])
            all_out_idx.append(idx[labels == c][out_idx])

        self.in_idx = np.concatenate(all_in_idx)
        self.out_idx = np.concatenate(all_out_idx)


class TargetModel(object):
    def __init__(
            self,
            dataset_name,
            epochs,
            batch_size,
            noload=False
    ):

        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size

        self.noload = noload

        # Set in subclass.
        self.model_name = None

        # Set in init().
        self.input_shape = None
        self.num_classes = None
        self.model = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def init(
        self,
        train_data, train_labels,
        verbose=0,
        valid_data=None, valid_labels=None
    ):

        self.input_shape = train_data.shape[1:]
        self.num_classes = train_labels.max() + 1

        try:
            if not self.noload:
                self.model = load_model(self.__get_name())
            else:
                raise Exception()

        except Exception:
            tqdm.write('Training target model...')
            np.random.seed(0)
            tf.set_random_seed(0)

            self.model = self._get_architecture()

            extra = dict()
            if valid_data is not None and valid_labels is not None:
                extra['validation_data'] = (
                    valid_data,
                    to_categorical(valid_labels)
                )

            self.model.fit(
                train_data,
                to_categorical(train_labels),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0,
                **extra
            )

            self.model.save(self.__get_name())

        return self

    def __get_name(self):
        return '{}.{}.epochs-{}.batch_size-{}.h5'.format(
            self.model_name, self.dataset_name, self.epochs, self.batch_size)


    # Uncomment for WE-5. The input and output layers are same for all models.
    """def _get_architecture_shadow(self):
        #raise NotImplementedError('abstract function')
        l_in = Input(self.input_shape)
        
        =====================================================================   MLP Model ============================================
        #l_inter = Flatten()(l_in)
        #layers = [64,32,16]
        #for num_units in layers:
        #    l_inter = Dense(num_units, activation='relu')(l_inter)
        ===============================================================================================================================
        
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  CONV1 Model ++++++++++++++++++++++++++++++++++++++++++++
        
        x = tf.keras.layers.Conv2D(32, (3, 3), 1, activation='relu')(l_in)
        x = tf.keras.layers.MaxPool2D((2, 2),2)(x)
        x = tf.keras.layers.Flatten()(x)
        l_inter = tf.keras.layers.Dense(64, activation='relu')(x)
        ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        _____________________________________________________________________  CONV2 Model  ____________________________________________

        x = tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu')(l_in)
        x = tf.keras.layers.Conv2D(32, (3, 3), 1, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        l_inter = tf.keras.layers.Dense(64, activation='relu')(x)
        _________________________________________________________________________________________________________________________________

        l_out = Dense(10, activation='softmax')(l_inter)

        m = Model(l_in, l_out)

        m.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return m"""
    
    def _get_architecture(self):            #Comment it for WE5 
            raise NotImplementedError('abstract function')

    def train_shadow_model(
        self,
        shadow_in_data, shadow_in_labels,
        verbose=0, seed=0,
        valid_in_data=None, valid_in_labels=None
    ):

        np.random.seed(seed)
        tf.set_random_seed(seed)

        extra = dict()
        if valid_in_data is not None and valid_in_labels is not None:
            extra['validation_data'] = (
                valid_in_data,
                to_categorical(valid_in_labels)
            )

        shadow_model = self._get_architecture()                 #Comment for WE5
        #shadow_model = self._get_architecture_shadow()         #Uncomment for WE5
        #print('Shadow Model')
        #print(shadow_model.summary())

        shadow_model.fit(
            shadow_in_data,
            to_categorical(shadow_in_labels),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=verbose,
            **extra
        )

        return shadow_model


class CIFARModel(TargetModel):
    def __init__(
        self,
        epochs,
        batch_size,
        layers=[128, 64, 32],
        dataset_name="cifar",
        *argv, **kwargs
    ):

        super(CIFARModel, self).__init__(
            dataset_name, epochs, batch_size, *argv, **kwargs
        )

        self.model_name = 'cifar_' + "_".join(map(str, layers))
        self.layers = layers

    def _get_architecture(self):

        l_in = Input(self.input_shape)
        l_inter = Flatten()(l_in)

        #for num_units in self.layers:
        l_inter = Dense(128, activation='relu')(l_inter)
        l_inter = Dense(64, activation='relu')(l_inter)
        l_inter = Dense(32, activation='relu',kernel_regularizer= regularizers.l2(0.1),bias_regularizer = regularizers.l2(0.1))(l_inter)  #Uncomment for WE4
        #l_inter = Dense(32, activation='relu')(l_inter)
        l_inter = tf.keras.layers.Dropout(0.5)(l_inter)        #Uncomment for WE4 
        l_out = Dense(10, activation='softmax')(l_inter)

        m = Model(l_in, l_out)

        m.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return m


class CIFARData(object):
    def __init__(self):
        # Get CIFAR splits with train and test having the same size and shadow
        # having twice that.

        train, test = tf.keras.datasets.cifar10.load_data()

        def norme(d):
            d = d - d.max() / 2.0
            d = d / d.std()
            return d

        def res(d):
            return d.reshape(d.shape[0])

        self.train = norme(train[0][0:10000])
        self.test = norme(test[0])
        self.shadow = norme(train[0][10000:30000])

        self.labels_train = res(train[1][0:10000])
        self.labels_test = res(test[1])
        self.labels_shadow = res(train[1][10000:30000])

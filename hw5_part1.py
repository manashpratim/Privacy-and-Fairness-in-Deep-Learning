import hw5_part1_utils

from typing import Tuple
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tqdm import tqdm



def synthesize_attack_data(
    target_model: hw5_part1_utils.TargetModel,
    shadow_data: np.ndarray,
    shadow_labels: np.ndarray,
    num_shadow_models: int = 4
):
    """Synthesize attack data.

    Arguments:

        target_model {TargetModel} -- an instance of the TargetModel class;
          behaves as a keras model but additionally has a train_shadow_model
          function, which takes a subset of the shadow data and labels and
          returns a model with identical architecture and hyperparameters to
          the original target model, but that is trained on the given shadow
          data.

        shadow_data {np.ndarray} -- data available to the attack to train
          shadow models. If the target model's training set is size N x D,
          shadow_data is 2N x D.

        shadow_labels {np.ndarray} -- the corresponding labels to the
          shadow_data, given as a numpy array of 2N integers in the range 0 to
          C where C is the number of classes.

        num_shadow_models {int} -- the number of shadow models to use when
          constructing the attack model's dataset.

    Returns: three np.ndarrays; let M = 2N * num_shadow_models

        attack_data {np.ndarray} [M, 2C] -- shadow data label probability and
           label one-hot

        attack_classes {np.ndarray} [M, 1 of {0,1,...,C}] -- shadow data labels

        attack_labels {np.ndarray} [M, 1 of {0,1}] -- attack data labels
           (training membership)

    """

    C = shadow_labels.max() + 1

    attack_data: np.ndarray = None
    attack_classes: np.ndarray = None
    attack_labels: np.ndarray = None

    # SOLUTION
    #raise NotImplementedError('You need to implement this.')
    # END OF SOLUTION

    
    one_hot = to_categorical(shadow_labels, num_classes= C)

    pred0 = []
    pred1 = []
    one_hot_00 = []
    one_hot_11 = []
    labels0 = []
    labels1 = []

    from tqdm import tqdm
    outer = tqdm(total=num_shadow_models, desc='Training Shadow Models', position=0)
    
    for i in range(num_shadow_models):
        outer.update(1)
        
        #Splitting the dataset into Sin and Sout
        in_idx= hw5_part1_utils.DataSplit(shadow_labels, seed = i).in_idx
        out_idx = hw5_part1_utils.DataSplit(shadow_labels, seed = i).out_idx
        s0 = shadow_data[out_idx]
        y0 = shadow_labels[out_idx]
        one_hot_0 = one_hot[out_idx]
        s1 = shadow_data[in_idx]
        y1 = shadow_labels[in_idx]
        one_hot_1 = one_hot[in_idx]

        model=target_model.train_shadow_model(s1, y1,seed=i)        #training the different models on the S_in set
        pred1 = pred1 + list(model.predict(s1))                     #predicting the probits for S_in
        pred0 = pred0 + list(model.predict(s0))                     #predicting the probits for S_out
        one_hot_00 = one_hot_00 + list(one_hot_0)
        one_hot_11 = one_hot_11 + list(one_hot_1)
        labels0 = labels0 + list(y0)
        labels1 = labels1 + list(y1)
        
    pred0 = np.array(pred0)
    pred1 = np.array(pred1)
    one_hot_00 = np.array(one_hot_00)
    one_hot_11 = np.array(one_hot_11)
    labels0 = np.array(labels0)
    labels1 = np.array(labels1)

    attack_data0 = np.concatenate((pred0,one_hot_00),axis=1)
    attack_data1 = np.concatenate((pred1,one_hot_11),axis=1)
    attack_data =  np.concatenate((attack_data0,attack_data1))

    attack_classes = np.concatenate((labels0,labels1))

    attack_labels = np.zeros(attack_classes.shape)
    attack_labels[int(attack_classes.shape[0]//2):] = 1

    #Shuffling the new datasets
    idx = np.random.permutation(attack_labels.shape[0])
    attack_data = attack_data[idx]
    attack_classes = attack_classes[idx]
    attack_labels = attack_labels[idx]


    return attack_data, attack_classes, attack_labels.astype(int)


def build_attack_models(
    target_model: hw5_part1_utils.TargetModel,
    shadow_data: np.ndarray,
    shadow_labels: np.ndarray,
    num_shadow_models: int = 4
):
    """Build attacker models.

    Arguments:

        target_model {TargetModel} -- an instance of the TargetModel class;
          behaves as a keras model but additionally has a train_shadow_model
          function, which takes a subset of the shadow data and labels and
          returns a model with identical architecture and hyperparameters to
          the original target model, but that is trained on the given shadow
          data.

        shadow_data {np.ndarray} -- data available to the attack to train
          shadow models. If the arget model's training set is size N x D,
          shadow_data is 2N x D.

        shadow_labels {np.ndarray} -- the corresponding labels to the
          shadow_data, given as a numpy array of 2N integers in the range 0 to
          C where C is the number of classes.

        num_shadow_models {int} -- the number of shadow models to use when
          constructing the attack model's dataset.

    Returns:

        {tuple} -- a tuple of C keras models, where the c^th model predicts the
        probability that an instance of class c was a training set member.

    """

    attack_data, attack_classes, attack_labels = \
        synthesize_attack_data(
            target_model,
            shadow_data,
            shadow_labels,
            num_shadow_models=4
        )

    # to return
    attack_models: Tuple[Model] = None

    C = shadow_labels.max() + 1

    # SOLUTION
    #raise NotImplementedError('You need to implement this.')
    # END OF SOLUTION
    
    models = []
    
    outer = tqdm(total=C, desc='Training', position=0)
    
    for i in range(C):

        outer.update(1)
        l_in = Input(int(2*C))
        l_inter = Dense(2056, activation='relu')(l_in)
        l_inter = Dense(1024, activation='relu')(l_inter)
        l_inter = tf.keras.layers.Dropout(0.3)(l_inter)
        l_inter = Dense(512, activation='relu')(l_inter)
        l_inter = tf.keras.layers.Dropout(0.5)(l_inter)

        l_out = Dense(1, activation='sigmoid')(l_inter)

        m = Model(l_in, l_out)

        m.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
            )

        m.fit(
                attack_data[attack_classes==i],
                attack_labels[attack_classes==i],
                epochs=32,
                batch_size=2048,
                verbose=0
        )
        
        models.append(m)
        
    attack_models = tuple(models)
    return attack_models


def evaluate_membership(attack_models, y_pred, y):
    """Evaluate the attacker about the membership inference

    Arguments:

        attack_model {tuple} -- a tuple of C keras models, where C is the
          number of classes.

        y_pred {np.ndarray} -- an N x C numpy array with the predictions of the
          model on the N instances we are performing the inference attack on.

        y {np.ndarray} -- the true labels for each of the instances given as a
          numpy array of N integers.

    Returns:

        {np.ndarray} -- an array of N floats in the range [0,1] representing
          the estimated probability that each of the N given instances is a
          training set member.

    """

    # To return
    preds: np.ndarray = None

    # SOLUTION
    #raise NotImplementedError('You need to implement this.')
    one_hot = to_categorical(y, num_classes= len(attack_models))

    attack_data = np.concatenate((y_pred,one_hot),axis=1)
    preds = np.zeros((y.shape[0]))
    
    for i in range(attack_data.shape[0]):

        preds[i] = attack_models[y[i]].predict(attack_data[i].reshape(1,-1))[0][0]

    # END OF SOLUTION

    return preds



if __name__ == '__main__':
    # Load the dataset.
    data = hw5_part1_utils.CIFARData()

    # Make a target model for the dataset.
    target_model = \
        hw5_part1_utils.CIFARModel(
            epochs=48,
            batch_size=2048,
            noload=True, # prevents loading an existing pre-trained target
                         # model
        ).init(
            data.train, data.labels_train,
            # data.test, data.labels_test # validation data
        )

    tqdm.write('Building attack model...')
    attack_models = build_attack_models(
        target_model,
        data.shadow,
        data.labels_shadow
    )

    tqdm.write('Evaluating attack model...')
    y_pred_in = target_model.predict(data.train)
    y_pred_out = target_model.predict(data.test)

    tqdm.write('  Train Accuracy: {:.4f}'.format(
        (y_pred_in.argmax(axis=1) == data.labels_train).mean()))
    tqdm.write('  Test Accuracy:  {:.4f}'.format(
        (y_pred_out.argmax(axis=1) == data.labels_test).mean()))

    in_preds = evaluate_membership(
        attack_models,
        y_pred_in,
        data.labels_train
    )
    out_preds = evaluate_membership(
        attack_models,
        y_pred_out,
        data.labels_test
    )

    wrongs_in = y_pred_in.argmax(axis=1) != data.labels_train
    wrongs_out = y_pred_out.argmax(axis=1) != data.labels_test

    true_positives = (in_preds > 0.5).mean()
    true_negatives = (out_preds < 0.5).mean()
    attack_acc = (true_positives + true_negatives) / 2.

    attack_precision = (in_preds > 0.5).sum() / (
        (in_preds > 0.5).sum() + (out_preds > 0.5).sum()
    )

    # Compare to a baseline that merely guesses correct classified instances
    # are in and incorrectly classified instances are out.
    baseline_true_positives = \
        (y_pred_in.argmax(axis=1) == data.labels_train).mean()
    baseline_true_negatives = \
        (y_pred_out.argmax(axis=1) != data.labels_test).mean()
    baseline_attack_acc = \
        (baseline_true_positives + baseline_true_negatives) / 2.

    baseline_precision = \
        (y_pred_in.argmax(axis=1) == data.labels_train).sum() / (
            (y_pred_in.argmax(axis=1) == data.labels_train).sum() +
            (y_pred_out.argmax(axis=1) == data.labels_test).sum()
        )

    tqdm.write(
      f"\nTrue positive rate: {true_positives:0.4f}, " +
      f"true negative rate: {true_negatives:0.4f}"
    )
    tqdm.write(
      f"Shadow Attack Accuracy: {attack_acc:0.4f}, precision: {attack_precision:0.4f} " +
      f"(baseline: {baseline_attack_acc:0.4f}, {baseline_precision:0.4f})"
    )

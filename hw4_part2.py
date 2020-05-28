# hw4_part2.py

import tqdm
import numpy as np
import tensorflow as tf

import hw4_utils
from hw4_utils import Splits
from tensorflow.keras import Model

from hw4_mnist import HW4Model, MNISTModel
from hw4_part1 import Attacker, PGDAttacker


class FineTunable(object):
    def __init__(self, finetune: bool = False):
        self.finetune = finetune

    def defend(self) -> None:
        if not self.finetune:
            # If we are not finetuning, we are training from scratch.
            self.model.build() # resets the model


class Defender(object):
    def __init__(
        self,
        attacker: Attacker,
        model: HW4Model,
        batch_size: int = 16,
        epochs: int = 2,
    ) -> None:

        self.attacker = attacker
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

    def defend(self) -> None:
        pass


class AugmentDefender(Defender, FineTunable):
    def __init__(
        self,
        finetune: bool = False,
        *argv, **kwargs
    ) -> None:
        """
            finetune: bool -- finetune the existing model instead of training
              from scratch
        """
        Defender.__init__(self, *argv, **kwargs)
        FineTunable.__init__(self, finetune)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        augment_ratio: float = 0.1,
    ):
        """Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

          augment_ratio: float -- how much adversarial data to use as a ratio
            of training data

        returns Xadv, Yadv (the adversarial instances generated in defense),
        self.model should be defended
          Xadv: np.ndarray [N*augment_ratio, 28, 28, 1]
          Yadv: np.ndarray [N*augment_ratio]

        """

        Xadv = None # the adversarial instances generated in the process of
                    # defense
        Yadv = None # and their (correct) class

        # >>> Your code here <<<

        num_samples = int(X.shape[0]*augment_ratio)
        
        ind = np.random.permutation(num_samples)
        adver_x =  X[ind]
        Yadv=  Y[ind]
        Xadv = np.zeros(adver_x.shape)
        for i in range(0,adver_x.shape[0],self.batch_size):
            p = self.attacker.attack_batch(adver_x[i:i+self.batch_size], Yadv[i:i+self.batch_size])
            Xadv[i:i+len(p)] = p

        # Generate | X | * augment_ratio adversarial examples,

        FineTunable.defend(self)

        # Resets model if not finetuning. If not finetuning,
        # make sure you generate the adversarial examples
        # before you call this.
        if not self.finetune: 
            new_X = np.vstack((X,Xadv))
            new_Y = np.vstack((Y.reshape(-1,1),Yadv.reshape(-1,1))).flatten()
            idx1 = np.random.permutation(new_X.shape[0])
            new_X = new_X[idx1]
            new_Y = new_Y[idx1]
        else:
            new_X = Xadv
            new_Y= Yadv 

        self.model.train(new_X,new_Y,batch_size=self.batch_size,epochs=self.epochs)

        # >>> End of your code <<<

        return Xadv, Yadv


class PreMadryDefender(Defender, FineTunable):
    def __init__(self, finetune: bool = False, *argv, **kwargs) -> None:
        Defender.__init__(self, *argv, **kwargs)
        FineTunable.__init__(self, finetune)

    def defend(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ):
        """
        Defend by augmenting the training data.

        inputs:
          X: np.ndarray - input images [N, 28, 28, 1]
          Y: np.ndarray - ground truth classes [N]

        returns Xadv, Yadv (the adversarial instances generated in defense),
        self.model should be defended
          Xadv: np.ndarray [N*epochs, 28, 28, 1]
          Yadv: np.ndarray [N*epochs]
        """

        FineTunable.defend(self) # resets model if not finetuning

        Xadv = None # the adversarial instances generated in the process of
                    # defense
        Yadv = None # and their (correct) class

        # >>> Your code here <<<

        # For each input batch, generate adversarial examples and train on them
        # instead of original data

        Xadv= []
        Yadv = []

  
        self.model.model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy']
        )


       
        for i in range(self.epochs):
            for j in range(0,X.shape[0],self.batch_size):
                p = self.attacker.attack_batch(X[j:j+self.batch_size], Y[j:j+self.batch_size])
                #self.model.train(p,Y[i:i+self.batch_size],batch_size=self.batch_size,epochs=1)
                self.model.model.train_on_batch(p, Y[j:j+self.batch_size])
                Xadv = Xadv + list(p)
                Yadv = Yadv + list(Y[j:j+self.batch_size])

        Xadv = np.array(Xadv)
        Yadv = np.array(Yadv)

        # >>> End of your code <<<

        return Xadv, Yadv



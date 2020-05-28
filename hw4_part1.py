from tqdm import tqdm

import numpy as np
import tensorflow as tf
from typing import List

from abc import ABC

import hw4_utils
from hw4_mnist import HW4Model, WithSession


class Attacker(ABC, WithSession):
    def __init__(
        self,
        model: HW4Model,
        target: int = None,
        learning_rate: float = 1.0,
        learning_rate_decay: float = 0.9,
        num_steps: int = 10,
        batch_size: int = 16,
    ) -> None:

        """
        Base attacker class.

        inputs:
          model: HW4Model -- model to attack
          target: None or int -- None for evasion attack, target class for
             targetted class attack.
          learning_rate: float -- learning rate for gradient descent
          learning_rate_decay: float -- decay learning rate by this every step
          num_steps: int -- number of steps to run before stopping
          batch_size: int -- number of instances in a batch of training
        """
        super().__init__(session=model.session)

        assert hasattr(model, "f_preds"), \
            "Attacker requires model to feature f_preds attribute."
        assert hasattr(model, "probits"), \
            "Attacker requires model to feature probits attribute."
        assert hasattr(model, "logits"), \
            "Attacker requires model to feature logits attribute."

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.target = target

        self.model = model
        self.batch_shape = tf.TensorShape([batch_size, 28, 28, 1])

        # variable initializations if needed
        self.variables = []

        self._define_ops()

    def __str__(self) -> str:
        return f"lr={self.learning_rate},lrd={self.learning_rate_decay}," + \
            f"steps={self.num_steps},t={self.target}"

    def initialize_variables(self) -> None:
        """Run any variable initializations in self.variables.

        """

        self.session.run(tf.variables_initializer(self.variables))

    def _define_ops(self) -> None:

        self.X = tf.placeholder(
            tf.float32, shape=self.batch_shape, name="X_placeholder"
        )
        self.Y = tf.placeholder(
            tf.uint8, shape=self.batch_size, name="Y_placeholder"
        )

        # variable versions the above
        self.X_var = tf.Variable(
            tf.zeros(self.batch_shape, dtype=tf.float32),
            name="X_var"
        )
        self.Y_var = tf.Variable(
            tf.zeros(self.batch_shape[0], dtype=tf.uint8),
            name="Y_var"
        )
        self.Yi = tf.one_hot(
            self.Y_var,
            self.model.num_classes,
            name="Yi_var"
        )

        self.lr = tf.Variable(self.learning_rate, name="lr")

        # operation to intialize variables from placeholders
        self.init_inputs = [
            tf.assign(self.X_var, self.X),
            tf.assign(self.Y_var, self.Y),
            tf.assign(self.lr, self.learning_rate)
        ]

    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        display_progress: bool = False
    ) -> (float, List[float]):

        """Evaluate the attacker on the given instances. Return success rate and mean
        distortion over successful attacks. Mean distortion is returned in
        terms of L0,L1,L2,Linf averaged over instances.

        """

        print(f'Target = {self.target} evaluation')

        Xadv = self.attack(X, Y, display_progress=display_progress)
        pred_adv = self.model.f_preds(Xadv)

        if self.target is None:
            success = pred_adv != Y
        else:
            success = (Y != self.target)*(pred_adv == self.target)

        delta = (Xadv - X)[success]
        delta = delta.reshape(len(delta), 28 * 28)
        distortions, distortions_str = hw4_utils.norms(delta)

        num_successes = success.sum()
        success_rate = num_successes / len(X)

        print('Success rate: ', success_rate)
        print('Mean Distortion: ', distortions_str)

        return success_rate, distortions

    def attack(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        display_progress: bool = False
    ) -> np.ndarray:
        """Try to attack all the provided instances. Return the best adversarial
        instance attainable in the amount of time steps that were allowed.

        inputs:
          X: np.ndarray -- images to be attacked (numpy array) [N,28,28,1]
          Y: np.ndarray -- class numbers of the correct class (for evasion)
          display_progress: bool -- if in a notebook, display loss graph while
             training

        returns:
          np.ndarray (same shape as X) -- Best attacks for each input instance.

        """

        self.initialize_variables()

        data = hw4_utils.Dataloader(X, Y, batch_size=self.batch_size)

        ret = []

        for X_batch, Y_batch in tqdm(
            data, unit="batch",
            leave=False,
            total=len(X)//self.batch_size
        ):
            ret.append(
                self.attack_batch(
                    X_batch, Y_batch,
                    display_progress=display_progress
                )
            )

        return np.vstack(ret)

    def attack_batch(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        display_progress: bool = False,
    ) -> np.ndarray:
        """
        Attack a batch of images.

        inputs:
          X: np.ndarray -- images to be attacked (numpy array) shaped
             [batch_size,28,28,1]
          Y: np.ndarray -- class numbers of the correct class (for evasion)
          display_progress: bool -- if in a notebook, display loss graph while
             training

        returns:
          np.ndarray (same shape as X) -- perturbed images
          float -- L1 distortions (per attacked image)
        """

        self.initialize_variables()
        for op in self.init_inputs:
            # run ops in order:
            self.session.run(
                op, feed_dict={self.X: X,
                               self.Y: Y}
            )

        best_loss = 100000 * np.ones(Y.shape) # large number
        best_Xadv = np.copy(X)

        if display_progress:
            from IPython.display import clear_output, display
            import matplotlib.pyplot as plt

            losses = []

        for _ in tqdm(range(self.num_steps), unit="step", leave=False):
            for s in self.step:
                self.session.run(s)

            loss = self.session.run([self.loss])[0]
            Xadv = self.session.run([self.output])[0]

            if display_progress:
                losses.append(loss)
                fig, ax = plt.subplots()

                clear_output(wait=True)
                for i in range(X.shape[0]):
                    ax.plot([aloss[i] for aloss in losses])

                display(fig)

            improved = loss < best_loss
            best_loss[improved] = loss[improved]
            best_Xadv[improved] = Xadv[improved]

        return best_Xadv


class PGDAttacker(Attacker):
    def __init__(self,
                 c: float = 1.0,
                 step_mode: str = "project",
                 *argv, **kwargs) -> None:
        """Projected Gradient Descent attacker.

        inputs:
           c: float -- hyper parameter for PGD attack

           step_mode: str -- "clip" for clipping after each step or "project"
             for projection

        """

        self.c = c
        self.step_mode = step_mode
        super().__init__(*argv, **kwargs)

    def __str__(self) -> str:
        return f"PGD(c={self.c},sm={self.step_mode},{super().__str__()})"

    def _define_ops(self) -> None:
        super()._define_ops()

        # Loss to be optimized by attacker.
        self.loss: tf.Tensor = None

        # A single step of the attack.
        self.step: List[tf.Tensor] = None

        # The output perturbed image.
        self.output: tf.Tensor = None

        # >>> Your code here <<<

        self.Xadv_var = tf.Variable(tf.zeros(self.batch_shape), name="Xadv")

        self.init_inputs.append(tf.assign(self.Xadv_var, self.X_var * 0.9))
        # gives us some room to search around white pixels

        # perturbation itself
        self.delta = self.X_var - self.Xadv_var

        probits = self.model.probits(self.Xadv_var)

        if self.target is None: # untargetted attack
            # Many different here, see Carlini paper, section V.A for some.
            correct_class = tf.log(tf.reduce_sum(probits * self.Yi, axis=1))
            incorrect_classes = tf.log(
                tf.reduce_max(probits * (1.0 - self.Yi), axis=1)
            )

            self.loss = correct_class - incorrect_classes

        else: # targetted attack
            # Many different here, see Carlini paper, section V.A for some.
            target_onehot = tf.one_hot(
                np.repeat(self.target, self.batch_size),
                self.model.num_classes
            )
            other_classes = tf.log(tf.reduce_max(
                (1.0 - target_onehot) * probits,
                axis=1
            ))
            target_class = tf.log(tf.reduce_sum(
                target_onehot * probits,
                axis=1
            ))

            self.loss = other_classes - target_class

        grad = tf.gradients(ys=self.loss, xs=self.Xadv_var)[0]
        Xadv_updated = tf.assign(
            self.Xadv_var,
            self.Xadv_var - self.lr * grad
        )

        delta_updated = self.X_var - Xadv_updated

        if self.step_mode == "project":
            # delta projected to self.c * 28 * 28 norm ball around 0.
            delta_project = hw4_utils.clip_eta(delta_updated,
                                               1,
                                               self.c*28.0*28.0)

        elif self.step_mode == "clip":
            # All pixel values clipped to [-c,c],
            delta_project = tf.clip_by_value(delta_updated, -self.c, self.c)

        else:
            raise Exception(f"unknown post-step mode {self.step_mode}")

        Xadv_projected = tf.assign(Xadv_updated, self.X_var + delta_project)

        # To be executed each step.
        self.step = [Xadv_projected,
                     tf.assign(self.lr, self.lr * self.learning_rate_decay)]

        # Clip output to valid [0,1] pixel values.
        self.output = tf.clip_by_value(Xadv_projected, 0, 1)

        # >>> End of your code <<<


class CWL2Attacker(Attacker):
    def __init__(self,
                 c: float = 1.0,
                 k: float = 0.0,
                 *argv, **kwargs):
        """
        inputs:
          c: float -- hyper parameter for CWL2 attack
          k: float -- confidence hyper parameter for CWL2 attack
          inputs of Attacker -- (see Attacker class)
        """
        self.c = c
        self.k = k
        super().__init__(*argv, **kwargs)

    def __str__(self) -> str:
        return f"CWL2 (c={self.c},k={self.k},{super().__str__()})"

    def _define_ops(self):
        super()._define_ops()

        # Loss to be optimized by attacker.
        self.loss: tf.Tensor = None

        # A single step of the attack. Will be run in order.
        self.step: List[tf.Tensor] = None

        # The output perturbed image.
        self.output: tf.Tensor = None

        # >>> Your code here <<<
        self.w = tf.Variable(tf.zeros(self.batch_shape), name="w")


        # multiplied by 0.9 and added 0.1 to take care of -1 and 1
        #self.init_inputs.append(tf.assign(self.w, tf.atanh(2*(self.X_var*0.9)-1+0.1)))
        self.init_inputs.append(tf.assign(self.w, tf.atanh(2*(self.X_var*0.9)-1)))

        self.newimg = 0.5*(tf.tanh(self.w)+1)

        logits = self.model.logits(self.newimg)

        self.l2dist = tf.reduce_sum(tf.square(self.newimg - self.X_var),[1,2,3])


        if self.target is None: # untargetted attack
            # Many different here, see Carlini paper, section V.A for some.
            correct_class = tf.reduce_sum(logits* self.Yi, axis=1)
            incorrect_classes =  tf.reduce_max(logits * (1.0 - self.Yi), axis=1)
    
            self.loss2 = tf.maximum(0.0, correct_class-incorrect_classes)


        else: # targetted attack
            # Many different here, see Carlini paper, section V.A for some.
            target_onehot = tf.one_hot(
                np.repeat(self.target, self.batch_size),
                self.model.num_classes
            )
            other_classes = tf.reduce_max(
                (1.0 - target_onehot) * logits,
                axis=1
            )
            target_class = tf.reduce_sum(
                target_onehot * logits,
                axis=1
            )

            self.loss2 = tf.maximum(0.0, other_classes-target_class)

        self.loss = self.l2dist + self.c*self.loss2
  
        grad2 = tf.gradients(ys=self.loss, xs=self.w)[0]
              
        w_star = tf.assign(self.w, self.w - self.lr * grad2)


        self.step = [w_star, tf.assign(self.lr, self.lr * self.learning_rate_decay)]

        self.output = 0.5*(tf.tanh(w_star)+1)


        # >>> End of your code <<<

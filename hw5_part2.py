import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
np.random.seed(42)
import random
random.seed(12345)
from tensorflow.data import Dataset
from tensorflow.errors import OutOfRangeError
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

import tensorflow.keras.backend as K
tf.set_random_seed(1234)
from tqdm.notebook import tqdm

sess = tf.Session()         #Without these two lines, the code will throw environment error due to session mismatch
K.set_session(sess)

#================================================================NOTE========================================================================================

# In this script, I have trained the adversary and classifier simulanteously.

# I have tried different combinations like training adversary first for all batches in an epoch and then the classifier (Please refer to script hw5_part22.py), 
# updating the parameters of adversary and then computing gradient of adversary loss wrt classifier weights and updating classifier parameters for a batch etc.
# These approaches gave inferior results compared to the 1st approach.

# My adversary model is a combination of the classifier and the adversary.
# That is I am training only 1 model that includes both the adversary and the classifier. 
# I am using symbolic tensors to get the layers of classifier model.
# So, updating weights for the adversary model will update the parameters of the classifier.

# I have set seeds so that the results that I obtained can be replicated...
# For Dem Parity, running for 41 epochs with a batch size of 1024 gave good results.. 
# For Eq Op, running for 38 epochs with a batch size of 1024 gave good results.. 
#==============================================================================================================================================================
class AdversarialFairModel(object):
    def __init__(self, classifier):
        # YOU DO NOT NEED TO MODIFY THIS CODE.
        self.classifier = classifier

    def predict(self, X):
        # YOU DO NOT NEED TO MODIFY THIS CODE.
        return self.classifier.predict(X)

    def _get_adversary_architecture(self):
        """Create a Model for the adversary."""

        # For Dem Parity Only
        out = Dense(1, activation='sigmoid')(self.classifier.output)
         
        return Model(self.classifier.input,out)

        #raise NotImplementedError('You need to implement this.')

    def projection_weights(self,q,k):           #Function to compute Weight Projection (proj term in the Paper)
    
        if len(q.shape) > 1:

            l = np.dot(q.flatten().reshape(1,-1),k.flatten().reshape(-1,1))/(np.square(np.linalg.norm(k.flatten()))+np.finfo(np.float32).tiny)
            return l*k

        else:

            a = q.reshape(1,-1)
            b = k.reshape(-1,1)
            
            return ((np.dot(a, b) / (np.square(np.linalg.norm(b))+np.finfo(np.float32).tiny))*b).flatten()         

    def train_dem_parity(
        self,
        X, y, z,
        epochs=32,
        batch_size=1024
    ):
        """ Train a model with (positive class) demographic parity.
        Inputs:

          X: np.ndarray [N, F] -- Instances over F features.

          y: np.ndarray [N, 1] -- Target class.

          z: np.ndarray [N, 1] -- Group membership.

          Returns nothing but updates self.classifier
        """

        #raise NotImplementedError('You need to implement this.')

        # SOLUTION
        # END OF SOLUTION
        #K.clear_session()
        #sess = tf.Session()
        #K.set_session(sess)

        adversary = self._get_adversary_architecture()                              #getting the adversary model

        # Defining Tensors Operations
        Y = tf.placeholder(tf.float32, shape=[None, 1])                             # placeholder for true labels
        Z = tf.placeholder(tf.float32, shape=[None, 1])                             # placeholder for protected attribute

        class_params = adversary.trainable_weights[:-2]                             #parameters of the classifier
        adv_params = adversary.trainable_weights[-2:]                               #parameters of the adversary
        outputs = [layer.output for layer in adversary.layers]                      #getting the symbolic tensors of all layers

        l_p = K.mean(K.binary_crossentropy(Y,outputs[-2], from_logits=False))       #classifier loss
        loss_p = K.function([adversary.input, Y], l_p)

        l_a = K.mean(K.binary_crossentropy(Z,outputs[-1], from_logits=False))       #adversary loss
        loss_a = K.function([adversary.input, Z], l_a)

        grads_adv = tf.gradients(ys=l_a, xs=adv_params)                             #Adversary gradients
        grads_class = tf.gradients(ys=l_p, xs=class_params)                         #Classifier gradients
        grads_class_adv = tf.gradients(ys=l_a, xs=class_params)                     #classifier gradients wrt adversary loss

        gradients_adv = K.function([adversary.input, Z], grads_adv)
        gradients_class = K.function([adversary.input, Y], grads_class)
        gradients_class_adv = K.function([adversary.input, Z], grads_class_adv)

        num = len(X)//batch_size

        #sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            outer = tqdm(total=num, desc='Train epochs', position=0)
    
            learning_rate = 1/(epoch+1)
            alpha = np.sqrt(epoch+1)
    
            loss_class = 0
            loss_adv = 0
            c = 0
        
            for b in range(0, len(X), batch_size):
                outer.update(1)
                
                c = c + 1
                l1 = loss_p([X[b:b+batch_size], y[b:b+batch_size]])                                                 #classifier loss
                l2 = loss_a([X[b:b+batch_size], z[b:b+batch_size]])                                                 #adversary loss
                clas = gradients_class([X[b:b+batch_size], y[b:b+batch_size]])                                      #classifier gradients
                adv = gradients_adv([X[b:b+batch_size], z[b:b+batch_size]])                                         #adversary gradients
                clasadv = gradients_class_adv([X[b:b+batch_size], z[b:b+batch_size]])                               #classifier gradient wrt adversary loss
    
                for i in range(len(adversary.trainable_weights)):
    
                    if i>7:
                        sess.run(tf.assign_sub(adversary.trainable_weights[i], learning_rate*adv[i-8]))            #adversary weight update
                    
                    else:
                
                        k = self.projection_weights(clas[i],clasadv[i])                                            
                        grad = clas[i] - k - alpha*clasadv[i]                                                      
                        sess.run(tf.assign_sub(adversary.trainable_weights[i], learning_rate*grad))                #classifier weight update
                
                loss_class += l1
                loss_adv += l2
                del l1,l2,clas,adv,clasadv,k,grad
            
            y_pred = (self.classifier.predict(X) > 0.5) * 1
            acc1 = (y_pred == y).mean()

            y_pred1 = (adversary.predict(X) > 0.5) * 1
            acc2 = (y_pred1 == z).mean()
          
            print('Epoch: ',epoch+1)
            print('Demographic Parity: ',evaluate_dem_parity(y_pred, y, z))
            print('Equality of Opportunity: ',evaluate_eq_op(y_pred, y, z))    
            print('Classification Loss: ',loss_class/c)
            print('Adversarial Loss: ',loss_adv/c)
            print('Classification Accuracy: ',acc1)
            print('Adversary Accuracy: ',acc2)
            del y_pred,y_pred1

    def train_eq_op(
        self,
        X, y, z,
        epochs=32, batch_size=1
    ):
        """ Train a model with (positive class) equality of opportunity debiasing.

        Inputs:

          X: np.ndarray [N, F] -- Instances over F features.

          y: np.ndarray [N, 1 of {0,1}] -- Target class.

          z: np.ndarray [N, 1 of {0,1}] -- Group membership.

        Returns nothing but updates self.classifier
        """

        #raise NotImplementedError('You need to implement this.')

        # SOLUTION
        # END OF SOLUTION

        # Model
        inp = Input(1)                  # for giving y as input to the adversary
        next_layer = tf.keras.layers.Concatenate(axis=1)([self.classifier.output, inp])
        out = Dense(1, activation='sigmoid')(next_layer)
         
        adversary = Model([self.classifier.input,inp],out)

        # The following part is same as dem_parity (Only difference is now y is given as input to the adversary)
        # Defining Tensors Operations
        
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        Z = tf.placeholder(tf.float32, shape=[None, 1])

        class_params = adversary.trainable_weights[:-2]
        adv_params = adversary.trainable_weights[-2:]
        outputs = [layer.output for layer in adversary.layers]

        l_p = K.mean(K.binary_crossentropy(Y,outputs[4], from_logits=False))
        loss_p = K.function([adversary.input, Y], l_p)

        l_a = K.mean(K.binary_crossentropy(Z,outputs[-1], from_logits=False))

        loss_a = K.function([adversary.input,tf.concat((outputs[4],adversary.layers[5].input),-1), Z], l_a)

        grads_adv = tf.gradients(ys=l_a, xs=adv_params)
        grads_class = tf.gradients(ys=l_p, xs=class_params)
        grads_class_adv = tf.gradients(ys=l_a, xs=class_params)

        gradients_adv = K.function([adversary.input,tf.concat((outputs[4],adversary.layers[5].input),-1), Z], grads_adv)
        gradients_class = K.function([adversary.input, Y], grads_class)
        gradients_class_adv = K.function([adversary.input,tf.concat((outputs[4],adversary.layers[5].input),-1), Z], grads_class_adv)

        num = len(X)//batch_size

        #sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
    
            learning_rate = 1/(epoch+1)
            alpha = np.sqrt(epoch+1)
            c = 0
            loss_class = 0
            loss_adv = 0
            

            outer = tqdm(total=num, desc='Train epochs', position=0)
            for b in range(0, len(X), batch_size):
                outer.update(1)
                c = c+1
                
                # Notations same as dem parity trainer 
                l1 = loss_p([X[b:b+batch_size], y[b:b+batch_size], y[b:b+batch_size]])
                l2 = loss_a([X[b:b+batch_size], z[b:b+batch_size], z[b:b+batch_size], z[b:b+batch_size]])
                clas = gradients_class([X[b:b+batch_size], y[b:b+batch_size],y[b:b+batch_size]])
                adv = gradients_adv([X[b:b+batch_size],y[b:b+batch_size], z[b:b+batch_size], z[b:b+batch_size]])
                clasadv = gradients_class_adv([X[b:b+batch_size],y[b:b+batch_size],z[b:b+batch_size],z[b:b+batch_size]])  
                
                for i in range(len(adversary.trainable_weights)):
            
                    if i>7:
                        sess.run(tf.assign_sub(adversary.trainable_weights[i], learning_rate*adv[i-8]))

                    else:
                        k = self.projection_weights(clas[i],clasadv[i]) 
                        grad = clas[i] - k - alpha*clasadv[i]
                        sess.run(tf.assign_sub(adversary.trainable_weights[i], learning_rate*grad))
            
                loss_class += l1
                loss_adv += l2
            
                del l1,l2,clas,adv,clasadv,k,grad
            y_pred = (self.classifier.predict(X) > 0.5) * 1
            acc1 = (y_pred == y).mean()

            y_pred1 = (adversary.predict([X,y]) > 0.5) * 1
            acc2 = (y_pred1 == z).mean()
    
            print('Epoch: ',epoch+1)
            print('Demographic Parity: ',evaluate_dem_parity(y_pred, y, z))
            print('Equality of Opportunity: ',evaluate_eq_op(y_pred, y, z))           
            print('Classification Loss: ',loss_class/c)
            print('Adversarial Loss: ',loss_adv/c)
            print('Classifier Accuracy: ',acc1)
            print('Adversary Accuracy: ',acc2)
            del y_pred,y_pred1


def evaluate_dem_parity(y_pred, y, z):
    """Compute demographic parity statistics.

        Inputs:
           y_pred: np.ndarray [N, 1 of {0,1}] -- Predicted class.

           y: np.ndarray [N, 1 of {0,1}] -- Target class.

           z: np.ndarray [N, 1 of {0,1}] -- Group membership.

        Returns tuple of positive outcome probabilities for the two groups.

    """
    #raise NotImplementedError('You need to implement this.')
    
    pred0 =  y_pred[z==0]
    prob0 = len(pred0[pred0==1])/len(pred0)
    
    pred1 =  y_pred[z==1]
    prob1 = len(pred1[pred1==1])/len(pred1)

    return (prob0,prob1)
   

def evaluate_eq_op(y_pred, y, z):
    """Compute equality of opportunity statistics.

        Inputs:
           y_pred: np.ndarray [N, 1 of {0,1}] -- Predicted class.

           y: np.ndarray [N, 1 of {0,1}] -- Target class.

           z: np.ndarray [N, 1 of {0,1}] -- Group membership.

        Returns tuple of positive outcome probabilities for the two groups,
        conditioned on positive ground truth.

    """

    #raise NotImplementedError('You need to implement this.')

    # SOLUTION

    # END OF SOLUTION
    c=0
    d=0
    e=0
    f=0
    for i in range(y_pred.shape[0]):
        if y[i] ==1 and z[i] == 0 and y_pred[i] == 1:
            c = c+1
        if y[i] ==1 and z[i] == 0 and y_pred[i] == 0:
            d = d+1
        if y[i] ==1 and z[i] == 1 and y_pred[i] == 1:
            e = e +1
        if y[i] ==1 and z[i] == 1 and y_pred[i] == 0:
            f = f + 1

    return (c/(c+d),e/(e+f))


if __name__ == '__main__':
    def norme(d):
        d = d.astype('float32')
        d -= d.min(axis=0)
        d -= d.max(axis=0) * 0.5
        d /= d.std(axis=0)
        return d

    temp = np.load("adult.npz")
    X, y, z = norme(temp['X'])[0:10000], \
        temp['y'].astype('float32')[0:10000], \
        temp['z'].astype('float32')[0:10000]

    baseline_accuracy = max(y.mean(), 1-y.mean())

    def make_adult_classifier():
        c_in = Input((X.shape[1],))
        c_inter = Dense(32, activation='relu')(c_in)
        c_inter = Dense(32, activation='relu')(c_inter)
        c_inter = Dense(32, activation='relu')(c_inter)
        c_out = Dense(1, activation='sigmoid')(c_inter)

        return Model(c_in, c_out)

    # Train original model.
    c_orig = make_adult_classifier()
    c_orig.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )
    c_orig.fit(X, y, epochs=2, batch_size=512)

    y_pred_orig = (c_orig.predict(X) > 0.5) * 1

    tqdm.write(
        'Original Model\n'
        '--------------\n'
        '\n'
        'Accuracy: {:.4f} (baseline: {:0.4f})\n'
        '\n'
        '                    Group 0\tGroup 1\n'
        'Demographic Parity: {:.4f}\t{:.4f}\n'
        'Equal Opportunity:  {:.4f}\t{:.4f}\n'
        .format(*(
          ((y_pred_orig == y).mean(), baseline_accuracy) +
          evaluate_dem_parity(y_pred_orig, y, z) +
          evaluate_eq_op(y_pred_orig, y, z))
        )
    )

    # Train model with demographic parity.
    c_dem_par = AdversarialFairModel(make_adult_classifier())
    c_dem_par.train_dem_parity(X, y, z, epochs=2, batch_size=512)

    y_pred_dem_par = (c_dem_par.predict(X) > 0.5) * 1

    print(
        'Demographic Parity\n'
        '------------------\n'
        '\n'
        'Accuracy: {:.4f} (baseline: {:0.4f})\n'
        '\n'
        '                    Group 0\tGroup 1\n'
        'Demographic Parity: {:.4f}\t{:.4f}\n'
        'Equal Opportunity:  {:.4f}\t{:.4f}\n'
        .format(*(
          ((y_pred_dem_par == y).mean(), baseline_accuracy) +
          evaluate_dem_parity(y_pred_dem_par, y, z)+
          evaluate_eq_op(y_pred_dem_par, y, z))
        )
    )

    # Train model with equality of opportunity.
    c_eq_op = AdversarialFairModel(make_adult_classifier())
    c_eq_op.train_eq_op(X, y, z, epochs=2, batch_size=512)

    y_pred_eq_op = c_eq_op.predict(X)

    print(
        'Equal Opportunity\n'
        '-----------------\n'
        '\n'
        'Accuracy: {:.4f} (baseline: {:0.4f})\n'
        '\n'
        '                    Group 0\tGroup 1\n'
        'Demographic Parity: {:.4f}\t{:.4f}\n'
        'Equal Opportunity:  {:.4f}\t{:.4f}\n'
        .format(*(
          ((y_pred_eq_op == y).mean(), baseline_accuracy) +
          evaluate_dem_parity(y_pred_eq_op, y, z) +
          evaluate_eq_op(y_pred_eq_op, y, z))
        )
    )

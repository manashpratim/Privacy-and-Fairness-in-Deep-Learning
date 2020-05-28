import gensim.models
import numpy as np
import json
from tqdm import tqdm


class WordEmbeddingDebiaser:

    def __init__(
        self,
        embedding_file_path,
        definitional_file_path='./data/definitional_pairs.json',
        equalize_file_path='./data/equalize_pairs.json',
        gender_specific_file_path='./data/gender_specific_full.json'
    ):

        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            embedding_file_path, binary=True
        )

        # collect first 300000 words
        self.words = sorted([w for w in self.model.vocab],
                            key=lambda w: self.model.vocab[w].index)[:300000]

        # all vectors in an array (same order as self.words)
        self.vecs = np.array([self.model[w] for w in self.words])
        tqdm.write('vectors loaded')
        # should take 2-5 min depending on your machine

        self.n, self.d = self.vecs.shape

        # word to index dictionary
        self.w2i = {w: i for i, w in enumerate(self.words)}

        # Some relevant words sets required for debiasing
        with open(definitional_file_path, "r") as f:
            self.definition_pairs = json.load(f)

        with open(equalize_file_path, "r") as f:
            self.equalize_pairs = json.load(f)

        with open(gender_specific_file_path, "r") as f:
            self.gender_specific_words = json.load(f)
        self._normalize()

    def accuracy(self):
        accuracy = self.model.accuracy('./data/questions-words.txt')
    
        sum_corr = len(accuracy[-1]['correct'])
        sum_incorr = len(accuracy[-1]['incorrect'])
        total = sum_corr + sum_incorr
        percent = lambda a: a / total * 100
        print('Total sentences: {}, Correct: {:.2f}%, Incorrect: {:.2f}%'.format(total, percent(sum_corr), percent(sum_incorr)))

    def _normalize(self):
        """
        normalize self.vecs
        """
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]

    def _drop(self, u, v):
        """
        remove a direction v from u
        """
        return u - v * u.dot(v) / v.dot(v)

    def w2v(self, word):
        """
        for a word, return its corresponding vector
        """
        return self.vecs[self.w2i[word]]

    def debias(self):
        self.gender_direction = self.identify_gender_subspace()
        self.neutralize()
        self.equalize()

    def identify_gender_subspace(self):
        """Using self.definitional_pairs to identify a gender axis (1 dimensional).

          Output: a gender direction using definitonal pairs

        ****Note****

         no other unimported packages listed above are allowed, please use
         numpy.linalg.svd for PCA

        """

        array = []
        for i in  self.definition_pairs:
            a = self.w2v(i[0])              #getting the word vectors
            b = self.w2v(i[1])
            mu_i = (a+b)/2                    #finding center 
            a = a - mu_i                      #centering
            b = b - mu_i 
            array = array + list(a.reshape(1,-1))   #appending the vectors to a list
            array = array + list(b.reshape(1,-1))
        
        array = np.array(array)
        
        u,s,v = np.linalg.svd(array, full_matrices=False)     #SVD    
        
        gender_direction = v[0]
        print(gender_direction.shape)
        return gender_direction

    def neutralize(self):
        """Performing the neutralizing step: projecting all gender neurtal words away
        from the gender direction

        No output, please adjust self.vecs

        """

        for i in range(len(self.words)):                                               #For all words in the embedding
            if self.words[i] not in  self.gender_specific_words:                       #If word is not in the set of gender specific
                self.vecs[i] = self._drop(self.vecs[i],self.gender_direction)          #neutralize that vector
        self._normalize()                                                              # done as suggested in the instructions

    def equalize(self):
        """Performing the equalizing step: make sure all equalized pairs are
        equaldistant to the gender direction.

        No output, please adjust self.vecs

        """
        #raise NotImplementedError('You need to implement this.')
        for i in self.equalize_pairs:
            a = self.w2v(i[0])                                                          #Getting the word vectors for the pair
            b = self.w2v(i[1])
            mu = (a+b)/2                                                                # finding the mean
            nu = self._drop(mu, self.gender_direction)                                  # neutralizing the vector
            nu_sqrt = np.sqrt(1 - np.square(np.linalg.norm(nu)))                        
            
            if np.dot((a-b).reshape(1,-1),self.gender_direction.reshape(-1,1)) <0:      #to preserve the sign convention during update.           
                                                                 
                self.vecs[self.words.index(i[0])] = nu - nu_sqrt*self.gender_direction
                self.vecs[self.words.index(i[1])] = nu + nu_sqrt*self.gender_direction 
            else:
                self.vecs[self.words.index(i[0])] = nu + nu_sqrt*self.gender_direction
                self.vecs[self.words.index(i[1])] = nu - nu_sqrt*self.gender_direction          

        self._normalize()

       
    def compute_analogy(self, w3, w1='woman', w2='man'):
        """input: w3, w1, w2, satifying the analogy w1: w2 :: w3 : w4

        output: w4(a word string) which is the solution to the analogy (w4 is
          constrained to be different from w1, w2 and w3)

        """
        diff = self.w2v(w2) - self.w2v(w1)
        vec = diff / np.linalg.norm(diff) + self.w2v(w3)
        vec = vec / np.linalg.norm(vec)
        if w3 == self.words[np.argsort(vec.dot(self.vecs.T))[-1]]:
            return self.words[np.argsort(vec.dot(self.vecs.T))[-2]]
        return self.words[np.argmax(vec.dot(self.vecs.T))]


if __name__ == '__main__':

    # Original Embedding

    we = WordEmbeddingDebiaser('./data/GoogleNews-vectors-negative300.bin')

    print('=' * 50)
    print('Original Embeddings')
    # she-he analogy evaluation
    w3s1 = [
        'her', 'herself', 'spokeswoman', 'daughter', 'mother', 'niece',
        'chairwoman', 'Mary', 'sister', 'actress'
    ]
    w3s2 = [
        'nurse', 'dancer', 'feminist', 'baking', 'volleyball', 'softball',
        'salon', 'blond', 'cute', 'beautiful'
    ]

    w4s1 = [we.compute_analogy(w3) for w3 in w3s1]
    w4s2 = [we.compute_analogy(w3) for w3 in w3s2]

    print('Appropriate Analogies')
    for w3, w4 in zip(w3s1, w4s1):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    print('Potentially Biased Analogies')
    for w3, w4 in zip(w3s2, w4s2):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    we.debias()

    print('=' * 50)
    print('Debiased  Embeddings')
    # she-he analogy evaluation
    w4s1 = [we.compute_analogy(w3) for w3 in w3s1]
    w4s2 = [we.compute_analogy(w3) for w3 in w3s2]

    print('Appropriate Analogies')
    for w3, w4 in zip(w3s1, w4s1):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    print('Potentially Biased Analogies')
    for w3, w4 in zip(w3s2, w4s2):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

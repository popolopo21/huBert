import keras
from pprint import pprint
import numpy as np

#Callback function to validate the training result after each epoch
class ValidationCallback(keras.callbacks.Callback):

    def __init__(self, sample_tokens,vocab,mask_token_id, top_k=5):
        self.sample_tokens = sample_tokens
        self.vocab = vocab
        self.mask_token_id = mask_token_id
        self.k = top_k
    
    #Decode the tokens (Should change it to the tokenizer's decode func)
    def decode(self, tokens):
        return " ".join([self.vocab[t] for t in tokens if t != 0])

    def convert_ids_to_tokens(self, id):
        return self.vocab[id]

    #On every epoch end do a validation
    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.sample_tokens) #do a prediction

        masked_index = np.where(self.sample_tokens == self.mask_token_id) #returns a 2 element array
        masked_index = masked_index[1] #the second element is the right index
        mask_prediction = prediction[0][masked_index] #prediction on the index

        top_indices = mask_prediction[0].argsort()[-self.k :][::-1] #sort the tokens with the highest prob.
        values = mask_prediction[0][top_indices] #probability for the token p(top_indices[i])=values[i]

        for i in range(len(top_indices)):
            p = top_indices[i] #token
            v = values[i]      #probability
            tokens = np.copy(self.sample_tokens[0])
            tokens[masked_index[0]] = p
            result = {
                "input_text": self.decode(self.sample_tokens[0]),
                "prediction": self.decode(tokens),
                "probability": v,
                "predicted mask token": self.convert_ids_to_tokens(p),
            }
            pprint(result)
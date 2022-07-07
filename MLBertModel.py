import keras
from keras import layers
import numpy as np
import tensorflow as tf
from MLModel import MLModel
from tensorflow.keras.optimizers import Adam

class MLBertModel:
    def __init__(self, config):
        self.config = config

    #This is the bert module. Q,K,V are matrixes and from them can the self-attention model calculate the outputs
    def bert_module(self,query, key, value, i):
        # Multi headed self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.config.NUM_HEAD,
            key_dim=self.config.EMBED_DIM // self.config.NUM_HEAD,
            name="encoder_{}/multiheadattention".format(i),
        )(query, key, value)
        attention_output = layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
            attention_output
        ) # The dropout can be changed
        attention_output = layers.LayerNormalization(
            epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
        )(query + attention_output) #The epsilon is added bcs in the calculation we dont want / with 0 so we add it.

        # Feed-forward layer
        ffn = keras.Sequential(
            [
                layers.Dense(self.config.FF_DIM, activation="relu"),
                layers.Dense(self.config.EMBED_DIM),
            ],
            name="encoder_{}/ffn".format(i),
        )
        ffn_output = ffn(attention_output)
        ffn_output = layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
            ffn_output
        )# The dropout can be changed
        sequence_output = layers.LayerNormalization(
            epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
        )(attention_output + ffn_output) #The epsilon can be changed
        return sequence_output

    #Bert uses position embeddings also.
    @staticmethod
    def get_pos_encoding_matrix(max_len, d_emb):
        pos_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
                if pos != 0
                else np.zeros(d_emb)
                for pos in range(max_len)
            ]
        )
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc

    def create_masked_language_bert_model(self):
        inputs = layers.Input((self.config.MAX_LEN,), dtype=tf.int64)

        #Word embedding layer
        word_embeddings = layers.Embedding(
            self.config.VOCAB_SIZE, self.config.EMBED_DIM, name="word_embedding"
        )(inputs)
        #Position Embedding layer
        position_embeddings = layers.Embedding(
            input_dim = self.config.MAX_LEN,
            output_dim = self.config.EMBED_DIM,
            weights= [self.get_pos_encoding_matrix(self.config.MAX_LEN, self.config.EMBED_DIM)],
            name="position_embedding",
        )(tf.range(start=0, limit= self.config.MAX_LEN, delta=1))
        embeddings = word_embeddings + position_embeddings # embeddings are created from word embeddings + pos embeddings

        encoder_output = embeddings
        for i in range(self.config.NUM_LAYERS): #The number of encoder layers in transformer
            encoder_output = self.bert_module(encoder_output, encoder_output, encoder_output, i)

        mlm_output = layers.Dense(self.config.VOCAB_SIZE, name="mlm_cls", activation="softmax")(
            encoder_output
        ) #Output of the model, from this can we generate our needed words or what we want
        mlm_model = MLModel(inputs, mlm_output, name="masked_bert_model")

        optimizer = Adam(learning_rate=self.config.LR)
        mlm_model.compile(optimizer=optimizer)
        return mlm_model
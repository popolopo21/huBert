# %%
import numpy as np
import tensorflow as tf
from BertConfig import BertConfig
import torch

class DataForMLM():
    
    def __init__(self, mask_token_id, special_tokens_ids_max, config):
        self.mask_token_id = mask_token_id
        self.special_tokens_ids_max = special_tokens_ids_max
        self.config = BertConfig

    def get_masked_input_and_labels(self,encoded_texts, vocab_size):

        #Same sized tensor as ethe encoded texts with values between 0 and 1
        rand = torch.rand(encoded_texts.shape)
        # 15% BERT masking
        inp_mask = (rand < 0.15)
        # Do not mask special tokens
        inp_mask[encoded_texts <= 5] = False
        # Set targets to -1 by default, it means ignore
        labels = -1 * np.ones(encoded_texts.shape, dtype=np.int64)
        # Set labels for masked tokens
        labels[inp_mask] = encoded_texts[inp_mask]

        # Prepare input
        encoded_texts_masked = np.copy(encoded_texts)
        # Set input to [MASK] which is the last token for the 90% of tokens
                # # # This means leaving 10% unchanged
        inp_mask_2mask = inp_mask & (torch.rand(encoded_texts.shape)< 0.90)
        encoded_texts_masked[inp_mask_2mask] = 5  # mask token is the last in the dict
        
        print(encoded_texts_masked)
        
        inp_mask_2random = inp_mask_2mask & (torch.rand(encoded_texts.shape) < 1 / 9)
        encoded_texts_masked[inp_mask_2random] = torch.randint(
            5, vocab_size, (1,)
            )

        # Prepare sample_weights to pass to .fit() method
        sample_weights = np.ones(labels.shape)
        sample_weights[labels == -1] = 0

        # y_labels would be same as encoded_texts i.e input tokens
        y_labels = np.copy(encoded_texts)

        mlm_ds = mlm_ds = tf.data.Dataset.from_tensor_slices(
        (encoded_texts_masked, y_labels, sample_weights))

        mlm_ds = mlm_ds.shuffle(1000).batch(self.config.BATCH_SIZE)

        return mlm_ds

# %%

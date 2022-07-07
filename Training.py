
from pathlib import Path
from DataLoader import Data
from transformers import BertTokenizer
from BertConfig import BertConfig
from DataForMLM import DataForMLM
from TokenizerConfig import TokenizerConfig
from ValidationCallback import ValidationCallback
import numpy as np
from MLBertModel import MLBertModel
import torch
import keras
from MLModel import MLModel

#Create the dataset from txt files

paths = [str(x) for x in Path('/content/drive/MyDrive/oscar_hu/Cleaned/').glob('**/*.txt')]
data = Data()
df = data.get_data_from_text_files(paths)

#Load the trained tokenizer

tokenizer = BertTokenizer.from_pretrained('./WordPieceTokenizer/vocab.txt')

#Encode the texts
encoded_texts = tokenizer(
    df.text.values.tolist(),
    max_length=BertConfig.MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors="tf")

#save mask id
mask_id = tokenizer.convert_tokens_to_ids('[MASK]')

#create the dataset for the model
mlm_ds = DataForMLM(mask_token_id=mask_id,special_tokens_ids_max=mask_id,config = BertConfig)
mlm_ds = mlm_ds.get_masked_input_and_labels(encoded_texts,TokenizerConfig.vocab_size)


#sample
sample = tokenizer(["Tegnap este [MASK] éreztem magam!"],max_length=256, padding='max_length', truncation=True, return_tensors="tf")
sample_tokens = sample.input_ids.numpy()
print(sample_tokens)
#we need to change key and values for the MaskedTextGenerator
vocab = dict((value, key) for key, value in tokenizer.get_vocab().items())
generator_callback = ValidationCallback(sample_tokens,vocab, mask_id)

#create the model
bert = MLBertModel(BertConfig)
bert_masked = bert.create_masked_language_bert_model()
bert_masked.summary()


#Training
bert_masked.fit(mlm_ds, epochs=5, callbacks=[generator_callback])

bert_masked.save('./Model')


#loading and testing
mlm_model = keras.models.load_model(
    "./Model/bert_mlm_hu.h5", custom_objects={"MaskedLanguageModel": MLModel}
)

prediction = mlm_model.predict(sample_tokens) #do a prediction

masked_index = np.where(sample_tokens == mask_id) #returns a 2 element array
masked_index = masked_index[1] #the second element is the right index
mask_prediction = prediction[0][masked_index]
print(mask_prediction)
top_indic = mask_prediction[0].argsort()[-1:] #sort the tokens with the highest prob.
print(top_indic)
values = mask_prediction[0][top_indic]
print(values)
print(tokenizer.convert_ids_to_tokens(top_indic))
# 

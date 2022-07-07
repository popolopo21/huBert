# maafidavid-smarttranslator

DataCleaning:
    - Download the data and place them in txt files (Data_DownLoader.py) -> DownloadedTextFiles
    - Preprocessing the text(cleaning) (Cleaning_Methods.py) -> CleanedTextFiles
    - Split the texts into sentences and load them into a dataframe  (DataLoader.py)

Tokenizer:
    - Bert uses WordPiece Tokenizer, which is a subword tokenizer between the word and the character tokenizer.(F.e.: 'playing' -> 'play' '##ing') (WordPieceTokenizer.py)
    - It can be configured in TokenizerConfig.py

DataPreprocessing:
    - We have to create a datastructure what can be fed into bert. Our model needs at least 2, but 3 inputs(features, labels, sample weights) (DataForMLM)
    - features = After we tokenize a text(encoded_text) we have to manipulate 15% of them. First we select randomly the 15% of the encoded text and we mask the 15%'s 90%. The remaining 10% will be changed to random tokens. These random tokens make the model more flexible.
    -labels = original encoded_text, this makes it supervised learning.
    -sample weights = probability of each token in the input.

Bert:
    -Its config is in BertConfig.py
    -Bert's inputs are embeddings(word_emb+pos_emb)
    -The embedding are going through encoders. The encoder has a multi head self-attention layer and a feed forward neural network layer. After each layer we have to normalize the vectors or matrixes.(MLBertModel.py)
    -In the last layer we have a softmax function to calculate the tokens probability for the mask token. Here just the masked tokens are calculated, all the other tokens are unchanged. (MLBertModel.py)

Training:
    - First we have to set up the training method and how the loss are calculated.(MLModel.py)
    -I wrote a callback function to validate the model after each epoch on a sample sentence(ValidationCallback.py)
    - After that we can start the training(Training.py)
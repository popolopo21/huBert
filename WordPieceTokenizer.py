from tokenizers import BertWordPieceTokenizer
from pathlib import Path
##LATER USE THE TOKENIZERCONFIG


##Initalize WordPieceTokenizer for Bert
tokenizer = BertWordPieceTokenizer(        
        clean_text = True,
        handle_chinese_chars = False,
        strip_accents = False,
        lowercase = True)

#Path for cleaned txt files
paths = [str(x) for x in Path('./CleanedTextFiles').glob('**/*.txt')]

#Train the tokenizer
tokenizer.train(
    files = paths,
    vocab_size = 52_000,
    min_frequency = 3,
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet = 100,
    wordpieces_prefix = '##'
    )

#Save the vocab.txt file
tokenizer.save('./WordPieceTokenizer/')
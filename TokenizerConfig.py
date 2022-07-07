from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    clean_text = True
    handle_chinese_chars = False
    strip_accents = False
    lowercase = True
    vocab_size = 52_000
    min_frequency = 3
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    limit_alphabet = 1000
    wordpieces_prefix = '##'


tok_config = TokenizerConfig()

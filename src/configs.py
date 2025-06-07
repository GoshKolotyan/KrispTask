import re
from pathlib import Path
from typing import  Optional
class ArmenianAudioDatasetConfig:
    
    def __init__(
        self,
        dataset_path: str = "mozilla-foundation/common_voice_16_0",
        language_code: str = "hy-AM",
        vocab_file_path: str = "vocab.json",
        special_tokens: Optional[dict[str, str]] = None
    ):
        self.dataset_path = dataset_path
        self.language_code = language_code
        self.vocab_file_path = Path(vocab_file_path)
        
        self.remove_columns = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
        
        self.unusable_chars = ['`', '«', '´', '»', '՚', '՛', '՜', '՝', '՞', '։', "'", '֊', "'", '…', "(",")",",","-",".",":","’"]

        self.local_train_path = "./datasets/train_dataset"
        self.local_test_path = "./datasets/test_dataset"
        
        
        #special tokens
        self.special_tokens = special_tokens or {
            "word_separator": "|",
            "unknown": "[UNK]",
            "padding": "[PAD]"
        }
        
        #regex pattern
        self.unusable_chars_pattern = self._create_char_pattern()
    
    def _create_char_pattern(self) -> str:
        escaped_chars = re.escape(''.join(self.unusable_chars))
        return f'[{escaped_chars}]'


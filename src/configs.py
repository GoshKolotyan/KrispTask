import re
import yaml
from pathlib import Path

class ArmenianAudioDatasetConfig:
    
    def __init__(self,config_path: str = 'configs/configs.yml'):
        self._load_from_yaml(config_path)
        #regex pattern
        self.unusable_chars_pattern = self._create_char_pattern()
    
    def _load_from_yaml(self, config_path: str)->None:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        self.dataset_path = config.get('dataset_path')
        self.language_code = config.get('language_code')
        self.sampling_rate = config.get('sampling_rate')
        self.vocab_file_path = Path(config.get('vocab_file_path'))
        
        self.remove_columns = config.get('remove_columns')
        self.unusable_chars = config.get('unusable_chars')
        self.local_train_path = config.get('local_train_path')
        self.local_test_path = config.get('local_test_path')
        self.local_valid_path = config.get("local_valid_path")
        self.model_name = config.get("model_name")
        
        self.special_tokens = config.get('special_tokens')

        self.get_model_args_dict = config.get("get_model_args_dict")
        self.script_args = config.get("script_args")  # Add this line

        self.get_training_args_dict = config.get("get_training_args_dict")


    def _create_char_pattern(self) -> str:
        escaped_chars = re.escape(''.join(self.unusable_chars))
        return f'[{escaped_chars}]'
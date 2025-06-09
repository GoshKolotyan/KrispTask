import re
import unicodedata

from typing import Tuple
from pprint import pprint
from datasets import Dataset, load_dataset, Audio

from configs import ArmenianAudioDatasetConfig
from _logging import log

class ArmenianAudioDataLoading:
    def __init__(self, configs:ArmenianAudioDatasetConfig):
        self.config = configs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.vocab_dict = None
        self.original_size = {}

        log.log_config(configs, "Data Loading configs")

    @log.log_phase("Dataset Loading")
    def load_data(self, use_separate_validation:bool)-> Tuple[Dataset]:
        if use_separate_validation:
            self.train_dataset = load_dataset(self.config.dataset_path, self.config.language_code, split='train')
            self.val_dataset = load_dataset(self.config.dataset_path, self.config.language_code, split='validation',)

        else:
            try:
                self.train_dataset = load_dataset(self.config.dataset_path, self.config.language_code, split='train+validation')
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset: {e}")      

        self.test_dataset = load_dataset(self.config.dataset_path, self.config.language_code, split='test')  

        #remove unnecessary colums 
        self.remove_unnecessary_colums()


        #show random samples
        # Helper.show_random_samples(self.train_dataset, num_examples=3, seed=42)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def remove_unnecessary_colums(self):
        columns_to_remove = [col for col in self.config.remove_columns if col in self.train_dataset.column_names]

        self.train_dataset = self.train_dataset.remove_columns(columns_to_remove)
        #add loging
        
        if self.val_dataset:
            self.val_dataset = self.val_dataset.remove_columns(columns_to_remove)
            #add loging
        
        self.test_dataset = self.test_dataset.remove_columns(columns_to_remove)
        #add loging

    @log.log_phase("Data Preporcessing")
    def preprocess_data(self, apply_quality_filters:bool, analuze_before_after:bool)->Tuple[Dataset]:

        #step 0
        # if analuze_before_after:
        #     #loging for analyze
        #     self._analyze_dataset("Before Prepro")
        
        #step 1
        #cleaning text
        self._clean_text()

        #step 2
        #quality filter
        # if apply_quality_filters:
            #logging for analyze
            # self._apply_quality_filters()
        
        #step 3 
        #create vocab
        self.vocab_dict = self._create_vocabulary()


        #step 4 
        #resampe audio
        self.resample_audio()

        # if analuze_before_after:
        #     self._analyze_dataset("After Prepro")
        

        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def _clean_text(self):
    
        def clean_armenain_text(batch: dict) -> dict:
            text = batch['sentence']

            original_text = text

            text = re.sub(self.config.unusable_chars_pattern, "", text)

            text = unicodedata.normalize("NFKC", text)

            text = re.sub(r'\s+', ' ',text).strip()

            text = text.lower()

            if len(text.split()) == 0:
                #logging
                text = ''

            batch['sentence'] = text
            return batch

        self.train_dataset = self.train_dataset.map(clean_armenain_text, desc="Cleaning armenain text")
    
        if self.val_dataset:
            self.val_dataset = self.val_dataset.map(clean_armenain_text, desc="Cleaning armenain text")
    
        self.test_dataset = self.test_dataset.map(clean_armenain_text, desc="Cleaning armenain text")

        self._remove_empty_texts()

        #logg success
    
    def _remove_empty_texts(self):

        train_before = len(self.train_dataset)
        val_before = len(self.val_dataset) if self.val_dataset else None
        test_before = len(self.test_dataset)

        self.train_dataset = self.train_dataset.filter(lambda x: len(x['sentence'].strip()) > 0)
        self.val_dataset = self.val_dataset.filter(lambda x: len(x['sentence'].strip()) > 0) if self.val_dataset else None
        self.test_dataset = self.test_dataset.filter(lambda x: len(x['sentence'].strip()) > 0)

        train_after = len(self.train_dataset)
        val_after = len(self.val_dataset) if self.val_dataset else None
        test_after = len(self.test_dataset)

        # if train_before != train_after:
        #     log.info(f"   Train: removed {train_before - train_after} empty texts")
        # if val_before != val_after:
        #     log.info(f"   Validation: removed {val_before - val_after} empty texts")
        # if test_before != test_after:
        #     log.info(f"   Test: removed {test_before - test_after} empty texts")
 
    def _create_vocabulary(self) -> dict[str, int]:
        # Extract characters from all datasets
        all_chars = set()
        
        # Get characters from train dataset
        for example in self.train_dataset:
            all_chars.update(list(example['sentence']))
        
        # Get characters from validation dataset if it exists
        if self.val_dataset:
            for example in self.val_dataset:
                all_chars.update(list(example['sentence']))
        
        # Get characters from test dataset
        for example in self.test_dataset:
            all_chars.update(list(example['sentence']))
        
        vocab_list = sorted(list(all_chars))
        vocab_dict = {char: inx for inx, char in enumerate(vocab_list)}

        if " " in vocab_dict:
            space_idx = vocab_dict[" "]
            del vocab_dict[" "]
            vocab_dict[self.config.special_tokens['word_separator']] = space_idx
            #logging
        
        vocab_dict[self.config.special_tokens['unknown']] = len(vocab_dict)
        vocab_dict[self.config.special_tokens['padding']] = len(vocab_dict)

        return vocab_dict
    
    def resample_audio(self):

        audio_config = Audio(sampling_rate=self.config.sampling_rate)

        self.train_dataset = self.train_dataset.cast_column('audio', audio_config)

        if self.val_dataset:
            self.val_dataset = self.val_dataset.cast_column('audio', audio_config)
        
        self.test_dataset = self.test_dataset.cast_column('audio', audio_config)
    
    def get_dataset(self)-> Tuple[Dataset]:
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_vocab(self)->dict[str, int]:
        return self.vocab_dict

    @log.log_phase("DATA PIPLINE")
    def __call__(self, use_separate_validation: bool):
        self.load_data(use_separate_validation=use_separate_validation)
        
        self.preprocess_data(apply_quality_filters=False, analuze_before_after=False)

        return self.train_dataset, self.val_dataset, self.test_dataset, self.vocab_dict

if __name__ == '__main__':
    from helper import HelperFunctions
    configs = ArmenianAudioDatasetConfig()

    dataloader = ArmenianAudioDataLoading(configs=configs)
    train_data, val_data, test_data, vocab_dict = dataloader(use_separate_validation=True)

    HelperFunctions.show_random_samples(dataset=train_data, num_examples=1)
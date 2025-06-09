import torch 
import random
import pandas as pd
from datasets import Dataset
from _logging import log

class HelperFunctions:
    @staticmethod
    @log.log_execution(log_args=False, log_time=False)
    def show_random_samples(dataset:Dataset, num_examples:int=1, seed:int=None):

        if seed is not None:
            random.seed(seed)

        # Ensure num_examples doesn't exceed dataset size
        num_examples = min(num_examples, len(dataset))

        log.info("Random sample from dataset")
        log.info(60 * "=")
        
        # Use random.sample for efficient sampling without replacement
        picks = random.sample(range(len(dataset)), num_examples)

        df = pd.DataFrame(dataset[picks])
        log.info(df.to_string())
        
        log.info("=" * 60)

    @staticmethod
    def set_random_seed(seed:int=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
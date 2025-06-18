import json
import numpy as np

from pathlib import Path
from evaluate import load
from transformers import Wav2Vec2BertProcessor

class Metrics:
    def __init__(self, processor:Wav2Vec2BertProcessor):
        self.processor = processor
        self.wer_metric = load('wer')
        self.cer_metric = load('cer')
    
    def compute_metrics(self, pred) -> dict[str, float]:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer":wer, "cer": cer}

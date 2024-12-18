
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import json
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import MulticlassF1Score, Accuracy

from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

class BertForSentenceClassification(PreTrainedModel):

    """
    BERT architecture is intended to be from "dbmdz/bert-base-italian-xxl-cased"
    but other models can be tried.
    """


    def __init__(self, config, model_name, num_labels, class_weights=None):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.class_weights = class_weights
        self.accuracy = Accuracy(num_classes=num_labels, task='multiclass')
        self.f1 = MulticlassF1Score(num_classes=num_labels, average='micro') # changed weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])

        loss = None
        if labels is not None:

            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1) #,weight=self.class_weights
            loss = loss_fct(logits, labels)

            f1_score = self.f1(logits.argmax(dim=1), labels)
            accuracy_score = self.accuracy(logits.argmax(dim=1), labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

def init(context):
    model_name = "document-classifier"
   
    model = context.project.get_model(model_name)
    local_path_model = model.download(overwrite=True)

    tokenizer = AutoTokenizer.from_pretrained(local_path_model)
    config = AutoConfig.from_pretrained(local_path_model)
    
    mm = BertForSentenceClassification.from_pretrained(
        local_path_model,
        config=config,
        model_name="dbmdz/bert-base-italian-xxl-cased",
        num_labels=config.num_labels
    ) 

    label_mapping = {
        0: 0,
        1: 2,
        2: 4,
        3: 7,
        4: 8,
        5: 10,	
        6: 11,
        7: 15,
        8: 16,
        9: 17,
        10: 20,
        11: 22,
        12: 25,
        13: 27,
        14: 28,
        15: 29,
        16: 36,
        17: 39,
        18: 45,
        19: 50,
        20: 51,
        21: 53,
        22: 54,
        23: 55,
        24: 56,
        25: 57,
        26: 61,
        27: 62,
        28: 63,
        29: 64,
        30: 65,
        31: 67,
        32: 68,
        33: 69,
        34: 72,
        35: 74,
        36: 81,
        37: 88,
        38: 89,
        39: 91,
        40: 96,
        41: 102,
        42: 107,
        43: 108,
        44: 109,
        45: 112,
        46: 113,
        47: 115,
        48: 116,
        49: 119,
        50: 120,
        51: 126,
        52: 130,
        53: 133,
        54: 134,
        55: 195
    }

    setattr(context, "model", mm)
    setattr(context, "tokenizer", tokenizer)
    setattr(context, "label_mapping", label_mapping)

def serve(context, event):

    context.logger.info(f"Received event: {event}")
    
    if isinstance(event.body, bytes):
        body = json.loads(event.body)
    else:
        body = event.body
        
    inference_input = body["inference_input"]
    
    pdf = pd.DataFrame(inference_input, index=[0])
    k = int(pdf['k'])
    inputs = context.tokenizer(str(pdf['text']), return_tensors="pt", truncation=True, padding=True, return_token_type_ids=False)
    context.logger.info(f"k received: {k}")
    
    with torch.no_grad():
        logits = context.model(**inputs).logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        indices_above_threshold = (probs >= 0.6).nonzero(as_tuple=True)[0]
        probs_above_threshold = probs[indices_above_threshold]
        sorted_indices = probs_above_threshold.argsort(descending=True)
        top_k_indices = indices_above_threshold[sorted_indices][:k]
        result = [context.label_mapping[int(idx)] + 1 for idx in top_k_indices]
        context.logger.info(f"result: {result}")
        
    # Convert the result to a json string.
    jsonstr = '{"results": ' + str(list(result)) + '}'
 
    return json.loads(jsonstr)


import pandas as pd
import numpy as np
import json
import torch
from numpyencoder import NumpyEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset, DatasetDict
from os import path, makedirs, listdir

class Dataloader:
    def __init__(self, file_path, file_type=None, label_column='label', test_size=0.2, val_size=0.25, random_state=25, **kwargs):
        """
        Initialize the Dataloader object for training CustomBertForSequenceClassification
        Args:
            file_path (str): Path to the parquet file.
            label_column (str): The column name representing labels in the dataset.
            test_size (float): Proportion of the dataset to use as the test set.
            val_size (float): Proportion of the train/validation split to use as the validation set.
            random_state (int): Seed for reproducibility.
        """
        self.file_path = file_path
        self.file_type = file_path.split('.')[-1]
        self.label_column = label_column
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.kwargs = kwargs
        self.df = None
        self.dataset = None
        self.num_labels = None
        self.class_weights = None
        self.encoding, self.reverse_encoding = None, None

    def load_data(self):
            """Loads the file and prepares the dataset."""
            loaders = {
                'csv': pd.read_csv,
                'gzip': pd.read_parquet,
                'excel': pd.read_excel,
                'json': pd.read_json,
                'feather': pd.read_feather,
            }
    
            if self.file_type not in loaders:
                raise ValueError(f"Unsupported file type: {self.file_type}")
            self.df = loaders[self.file_type](self.file_path, **self.kwargs).reset_index()
    
            # Labels should start from 0; they will be mapped back
            # when saving the predicted results
            if self.df[self.label_column].min() == 1:
                self.df[self.label_column] -= 1
    
            # Map labels to dense for CrossEntropyL (the italian BERT doesn't like sparse arrays)
            unique_labels, label_counts = np.unique(self.df[self.label_column], return_counts=True)
            self.num_labels = len(unique_labels)
            self.encoding = {label:idx for idx,label in enumerate(unique_labels)}
            self.reverse_encoding = {idx:label for idx,label in enumerate(unique_labels)}
            self.df[self.label_column] = self.df[self.label_column].map(self.encoding)
            # saving the reverse indexing
            with open("reverse_encoding.json", "w") as f:
                json.dump(self.reverse_encoding, f,
                          indent=4, sort_keys=True,
                          separators=(', ', ': '), ensure_ascii=False,
                          cls=NumpyEncoder)
    
            # Class weights
            inverse = 1 / label_counts
            normalized_weights = inverse / inverse.sum()
            print(self.df.head())
            # self.class_weights = torch.FloatTensor(normalized_weights).to('cuda')


    def stratified_split(self):
        """
        Performs stratified train/validation/test split.
        Given that the data has many labels and they are often unequally distributed,
        many classes can be not represented at all in the training session.
        This method forces similar class distributions.
        If some classes are in less than 3 observations, it will raise an error.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # First split: train+val/test
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_val_idx, test_idx = next(split.split(self.df, self.df[self.label_column]))
        train_val_data = self.df.iloc[train_val_idx].reset_index(drop=True)
        test_set = self.df.iloc[test_idx].reset_index(drop=True)

        # Second split: train/validation
        split = StratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
        train_idx, val_idx = next(split.split(train_val_data, train_val_data[self.label_column]))
        train_set = train_val_data.iloc[train_idx].reset_index(drop=True)
        val_set = train_val_data.iloc[val_idx].reset_index(drop=True)

        # Convert to Hugging Face DatasetDict
        self.dataset = DatasetDict({
            'train': Dataset.from_pandas(train_set),
            'validation': Dataset.from_pandas(val_set),
            'test': Dataset.from_pandas(test_set)
        })

    def get_dataset(self):
        """
        Retrieves the dataset dictionary containing train, validation, and test sets.

        Returns:
            DatasetDict: A dictionary containing the splits as Hugging Face datasets.
        """
        if self.dataset is None:
            raise ValueError("Dataset not created. Call stratified_split() first.")
        return self.dataset

    def get_class_weights(self):
        """
        Retrieves the class weights.

        Returns:
            torch.FloatTensor: Class weights for handling imbalanced classes.
        """
        if self.class_weights is None:
            raise ValueError("Class weights not computed. Call load_data() first.")
        return self.class_weights

    def get_num_labels(self):
        """
        Retrieves the number of unique labels.

        Returns:
            int: Number of unique labels in the dataset.
        """
        if self.num_labels is None:
            raise ValueError("Number of labels not available. Call load_data() first.")
        return self.num_labels

    def get_encoding(self):
        """
        Retrieves the mapping of labels to categories.

        Returns:
            dict: Mapping of label IDs to their respective categories.
        """
        if self.encoding is None:
            raise ValueError("Label mapping nonexistent. Call load_data() first.")
        return self.encoding

    def get_r_encoding(self):
        """
        Retrieves the mapping of labels to categories.

        Returns:
            dict: Mapping of label IDs to their respective categories.
        """
        if self.reverse_encoding is None:
            raise ValueError("Label mapping nonexistent. Call load_data() first.")
        return self.reverse_encoding


from transformers import AutoTokenizer

class TokenizerFunction:
    def __init__(self, model_name, max_length=512, use_fast=True, use_cache=False):
        """
        Initialize the tokenizer function wrapper 🌯
        Args:
            model_name (str): Name of the pre-trained model.
            max_length (int): Maximum sequence length for tokenization.
            use_fast (bool): Whether to use the fast tokenizer implementation.
            use_cache (bool): Whether to cache the tokenizer results.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                        use_fast=use_fast,
                                                        use_cache=use_cache)
        self.max_length = max_length

    def __call__(self, data):
        """
        Tokenizes the input data.

        Args:
            data (dict): A dictionary containing the 'text' field to tokenize.

        Returns:
            dict: Tokenized data including input IDs, attention masks, etc.
        """
        return self.tokenizer(data['text'],
                              padding='max_length',
                              truncation=True,
                              max_length=self.max_length,
                              return_tensors="pt")

    def get_tokenizer(self):
        return self.tokenizer



import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.classification import MulticlassF1Score, Accuracy

from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoConfig, TrainingArguments, EarlyStoppingCallback, Trainer

from datetime import date, datetime
import os

import json
# import wandb

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
        self.f1 = MulticlassF1Score(num_classes=num_labels, average='macro') # changed weight,micro None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])

        loss = None
        if labels is not None:

            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1) #,weight=self.class_weights
            loss = loss_fct(logits, labels)

            f1_score = self.f1(logits.argmax(dim=1), labels)
            accuracy_score = self.accuracy(logits.argmax(dim=1), labels)
            # wandb.log({
            #     "f1_score": f1_score,
            #     "accuracy": accuracy_score,
            #     'CrossEntropyLoss': loss.item() if loss is not None else None
            # })
            # print("F1 score", f1_score)
            # print("accuracy", accuracy_score)
            # print("CrossEntropyLoss", loss.item())

        return SequenceClassifierOutput(loss=loss, logits=logits)

class TrainerHandler:
    def __init__(self, trainer, tokenized_datasets, num_labels, encoding, model_name,
                 model_save_path):
        """
        Args:
            trainer: Hugging Face Trainer instance.
            tokenized_datasets: DatasetDict with train/val/test splits.
            num_labels: Number of labels in the classification task.
            encoding: Mapping from dense to sparse original labelling.
            model_name
            model_save_path: Path to save the configuration, weights and tokenizer for reproducibility.
        """
        self.trainer = trainer
        self.tokenized_datasets = tokenized_datasets
        self.num_labels = num_labels
        self.encoding = encoding
        self.model_name = model_name
        self.model_save_path = model_save_path

    def compute_f1_score(self, predictions):
        """
        Micro F1 score from

        Args:
            predictions: Output from the trainer.predict() method.

        Returns:
            f1_score: The calculated F1 score.
        """
        logits = torch.tensor(predictions.predictions)
        labels = torch.tensor(predictions.label_ids)
        f1 = MulticlassF1Score(num_classes=self.num_labels, average='micro') # weight
        f1_score = f1(logits.argmax(dim=1), labels.float())
        return f1_score, logits, labels

    def save_f1_results(self, f1_score):
        """
        Save the F1 score to a CSV file.

        Args:
            f1_score: by default, it's the micro F1 weighted on class frequency.
        """
        now = datetime.today()
        
        csvFile = self.model_save_path + '/' +  f'{self.model_name}.csv';

        df = pd.DataFrame({
            'F1': [f1_score.item()],
            'modello': [self.model_name],
            'T': [now],
        })
        
        if not os.path.isfile(csvFile):
            df.to_csv(csvFile, header='column_names', index=False)
        else:  # else it exists so append without writing the header
            df.to_csv(csvFile, mode='a', header=False, index=False)        
        
        print(f"F1 score saved to {self.model_name}.csv")

    def save_predictions(self, logits, labels):
        """
        Save predicted and true labels to a CSV file.

        Args:
            logits: Model logits from the predictions.
            labels: Ground truth label indexes from the predictions.
        """
        now = datetime.today()
        inverted_encoding = {int(v): k for k, v in self.encoding.items()}
        predicted_indices = logits.argmax(dim=1)
        predicted_labels = [inverted_encoding[idx.item()] for idx in predicted_indices]
        true_labels = [inverted_encoding[idx.item()] for idx in labels]

        original_index = self.tokenized_datasets['test']['index'] #original indexes for test observations in the input df

        results = pd.DataFrame({
            'original_index': original_index,
            'true_label': true_labels,
            'predicted_label': predicted_labels,
        })

        results.true_label, results.predicted_label = results.true_label + 1, results.predicted_label + 1

        file_name = f'{self.model_name}_{now.month}_{now.day}-{now.hour}_{now.minute}.csv'
        results.to_csv(self.model_save_path + '/' + file_name, index=False)
        print(f"Predictions saved to {file_name}")

    def save_model_and_tokenizer(self):
        """
        Save the trained model and tokenizer to the specified path.
        """
        print(f"Saving the trained model as {self.model_name}...")
        self.trainer.model.save_pretrained(self.model_save_path)
        self.trainer.tokenizer.save_pretrained(self.model_save_path)
        print(f"Model and tokenizer saved to {self.model_save_path}")

    def run(self):
        """
        Execute the training and save the outputs.
        """
        print("Starting training...")
        #self.trainer.train(resume_from_checkpoint=True)
        self.trainer.train()
        self.trainer.save_state()

        print("Evaluating test set...")
        predictions = self.trainer.predict(self.tokenized_datasets['test'])

        f1_score, logits, labels = self.compute_f1_score(predictions)
        print(f"F1 score: {f1_score.item()}\n")

        self.save_f1_results(f1_score)
        self.save_predictions(logits, labels)
        self.save_model_and_tokenizer()
        print("Done.")


    
file_basepath = "document-classifier"

import evaluate
def compute_metrics(eval_pred):
    metrics = ["f1","accuracy", "recall", "precision"] #List of metrics to return ,
    metric={}
    for met in metrics:
        metric[met] = evaluate.load(met)#load_metric(met)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_res={}
    for met in metrics:
        if (met != 'accuracy'):
            metric_res[met]=metric[met].compute(predictions=predictions, references=labels, average='macro')[met]
        else:
            metric_res[met]=metric[met].compute(predictions=predictions, references=labels)[met]
    
    dir = f"{file_basepath}/models/"
    DATA_FILENAME = dir +  f'metrics.json';

    # Create the directory for the data
    if not path.exists(dir):
        makedirs(dir)
    
    file_data = []
    if (path.exists(DATA_FILENAME)):
        with open(DATA_FILENAME , "r") as json_file:
            file_data = json.loads(json_file.read())

    # change it
    file_data.append(metric_res)     

    # write it all back 
    with open(DATA_FILENAME , "w") as json_file:
        json_file.write(json.dumps(file_data))
    
    return metric_res
    
def train(project, train_data, data_path = "data/", model_save_path = "models/", target_model_name = ""):

    model_dir = f"{file_basepath}/{model_save_path}"
    data_dir = f"{file_basepath}/{data_path}"
    
    try:
        shutil.rmtree(data_dir)
    except:
        print("Error deleting data dir")
                
    # Create the directory for the data
    if not path.exists(data_dir):
        makedirs(data_dir)

    try:
        train_data.download(data_dir) # this must change in the function
    except:
        print("Error downloading data")
        
    try:
        shutil.rmtree(model_dir)
    except:
        print("Error deleting model dir")
        
     # Create the directory for the model
    if not path.exists(model_dir):
        makedirs(model_dir)    

    model_name = "dbmdz/bert-base-italian-xxl-cased"
    dataloader = Dataloader(file_path= f'{data_dir}/addestramento.gzip')
    dataloader.load_data()
    dataloader.stratified_split()
    dataset = dataloader.get_dataset()
    # class_weights = dataloader.get_class_weights()
    num_labels = dataloader.get_num_labels()
    encoding = dataloader.get_encoding()
    tokenize_function = TokenizerFunction(model_name=model_name, max_length=512)
    tokenized_datasets = (dataset.map(tokenize_function, batched=True)
                          .shuffle(seed=25)
                          .remove_columns(['text', 'token_type_ids']))
    print(f'--Loading the model for predicting {num_labels} labels--')
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)  #for later saving the config
    model = (BertForSentenceClassification
         .from_pretrained(pretrained_model_name_or_path=model_name,
                          model_name=model_name,
                          config=config,
                          num_labels=num_labels)) #class_weights=class_weights
    training_args = TrainingArguments(
        output_dir='tuned_model',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        weight_decay=0.005,
        learning_rate=1e-5,
        lr_scheduler_type='linear',
        load_best_model_at_end=True,
        evaluation_strategy = "epoch", #To calculate metrics per epoch
        logging_strategy="epoch"
        
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        compute_metrics=compute_metrics
    )
    trainer.tokenizer = tokenize_function.get_tokenizer()
    
    handler = TrainerHandler(
        trainer=trainer,
        tokenized_datasets=tokenized_datasets,
        num_labels=num_labels,
        encoding=encoding,
        model_name=target_model_name,
        model_save_path=model_dir
    )
    handler.run()

    # metrics={}
    # df = pd.read_csv(model_dir + '/' +  f'{target_model_name}.csv')
    # last_row = df.tail(1)
    # metrics['F1'] = last_row['F1'].values[0]
    # for key, value in df.iterrows():
    #     metrics[f'F1_epoch{key+1}'] = value['F1']
    
    df = pd.read_json(model_dir + '/metrics.json')
    last_row = df.tail(1)
    model_metrics = ["f1","accuracy", "recall", "precision"]
    metrics = {}
    for met in model_metrics:    
        metrics[met] = last_row[met].values[0]
    
    project.log_model(
        name=target_model_name,
        kind="huggingface",
        base_model="dbmdz/bert-base-italian-xxl-cased",
        metrics=metrics,
        source=model_dir,
    )             


# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:28:19 2023

@author: alvar
"""

import os
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from datasets import Dataset, DatasetDict
from transformers.pipelines.pt_utils import KeyDataset
from transformers import (
    AutoTokenizer, 
    AutoConfig,
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TextClassificationPipeline
)
import evaluate

from tqdm.auto import tqdm

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining

from api_setup import wandb_setup, huggingface_setup
from datatools import balance_data_with_undersampling, 
from misc import save_json
from typing import List, Dict, Tuple, Union, Callable, Optional


f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

class SexismDataset:
    def __init__(
            self, 
            filepath_or_df: Union[str, pd.DataFrame], 
            task_name: str, 
            model_checkpoint: Optional[str] = None, 
            seed: int = 1234, 
            max_instances: Optional[Union[int, float]] = None
        ):
        """
        Initialize the SexismDataset object.

        Parameters:
            - filepath_or_df (Union[str, pd.DataFrame]): Path to a CSV/TSV 
                file or a pandas DataFrame containing the dataset.
            - task_name (str): Task name, either "detection" or "classification".
            - model_checkpoint (Optional[str]): Model checkpoint for tokenization.
            - seed (int): Random seed for reproducibility.
            - max_instances (Optional[Union[int, float]]): Maximum number of 
                instances to include in the dataset.
        """
        self.seed = seed
        self.task = task_name
        self._max_instances = max_instances
        
        if isinstance(filepath_or_df, str):
            # Load dataset from a file
            self._df = pd.read_csv(
                filepath_or_buffer=filepath_or_df, 
                sep="\t" if filepath_or_df.endswith(".tsv") else ";"
            )
        elif isinstance(filepath_or_df, pd.DataFrame):
            # Use the provided DataFrame
            self._df = filepath_or_df.copy()
        
        self._df.rename(
            columns={"sexist" if self.task == "detection" else "type": "label"},
            inplace=True
        )
        
        if max_instances:
            # Perform undersampling if max_instances is specified
            self._df = self.sample_df(max_instances)
        
        self.num_labels = len(self._df.label.unique())
        
        self.dataset = self._prepare_dataset()
        
        if model_checkpoint:
            # Tokenize the dataset if a model checkpoint is provided
            print("Tokenizing dataset...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
            self.tokenized = self.dataset.map(self._tokenize, batched=True)
          
    def _make_dataset_from_pandas(self, data: Union[pd.DataFrame,dict]):
        """
        Convert a pandas DataFrame or dictionary of DataFrames to a Hugging 
        Face Dataset object.

        Parameters:
            - data (Union[pd.DataFrame, dict]): Input data to convert.

        Returns:
            Dataset: Hugging Face Dataset object.
        """
        if isinstance(data, pd.DataFrame):
            # Convert single DataFrame
            return Dataset.from_pandas(data)
        elif isinstance(data, dict):
            # Convert dictionary of DataFrames
            ds_dict = DatasetDict()
            for split, df in data.items():
                ds_dict[split] = Dataset.from_pandas(df)
            return ds_dict 

    def _prepare_dataset(self):
        """
        Prepare the dataset for training.

        Returns:
            Dataset: Prepared dataset.
        """
        data = self._df.loc[:, ["text", "label"]]
        
        if self.task == "detection":
            # Perform label balancing (undersampling) for detection task
            data = balance_data_with_undersampling(data, positive_ratio=0.45)

        elif self.task == "classification":
            # Perform label encoding for classification task
            self.le = LabelEncoder()
            data.label = self.le.fit_transform(data.label.to_list())

        # Split the data into training, validation, and test sets
        train_data, eval_data = train_test_split(data, 
                                                 test_size=0.25, 
                                                 stratify=data.label, 
                                                 shuffle=True, 
                                                 random_state=self.seed)
        
        # Split the evaluation data into validation and test sets
        val_data, test_data = train_test_split(eval_data, 
                                               test_size=0.1/0.25, 
                                               stratify=eval_data.label, 
                                               random_state=self.seed)
        
        dfs = {"train": train_data, "validation": val_data, "test": test_data}
        ds = self._make_dataset_from_pandas(dfs).select_columns(["text", "label"])
        if "__index_level_0__" in ds.column_names:
            ds.remove_columns("__index_level_0__")
        return ds

    def _tokenize(self, examples):
        """
        Tokenize input examples using the specified tokenizer.

        Parameters:
            - examples: Input examples to be tokenized.

        Returns:
            Dict: Tokenized examples.
        """
        return self.tokenizer(examples["text"], 
                              padding='max_length', 
                              max_length = 512, 
                              truncation = True)
    
    def sample_df(self, n_or_perc): 
        """
        Sample a fraction or a specific number of instances from the dataset.

        Parameters:
            - n_or_perc: Number of instances or percentage to sample.

        Returns:
            pd.DataFrame: Sampled dataset.
        """
        if n_or_perc < 0:
            raise ValueError("'n_or_perc' must be positive")
        if n_or_perc > 1: 
            n_or_perc /= len(self._df)
        data, _ =  train_test_split(self._df, 
                                    train_size=n_or_perc,
                                    stratify=self._df.label,
                                    shuffle=True, 
                                    random_state=self.seed)
        return data
        
    def build_compute_metrics(self):
        """
        Build a compute_metrics function for evaluating the model.

        Returns:
            function: Compute metrics function.
        """
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            compute_kwargs = {
                "predictions": predictions,
                "references": labels
            }
            average = "binary" if self.task == "detection" else "macro"
            return {
                **accuracy_metric.compute(**compute_kwargs),
                **f1_metric.compute(**compute_kwargs, average=average),
                **precision_metric.compute(**compute_kwargs, average=average),
                **recall_metric.compute(**compute_kwargs, average=average)
            }
        return compute_metrics

    def __getitem__(self, key):
        """
        Get the dataset or tokenized examples.

        Parameters:
            - key: Key to access dataset or tokenized examples.

        Returns:
            Dataset or Tokenized: Dataset or tokenized examples.
        """
        if key == "dataset":
            return self.dataset
        elif key == "tokenized":
            return self.tokenized
        else:
            raise KeyError(f"{key} not supported. Must be 'dataset' or 'tokenized'")
         
            
def train_transformer(
    
    fancy_name: str,
    tokenized_dataset: Dataset,
    model_name: str,
    num_labels: int,
    tokenizer: AutoTokenizer,
    compute_metrics: Callable,
    hyper_params: dict = {},
    push_to_hub: bool = False,
    evaluate: bool = True
    
) -> Trainer:
    """
    Trains a transformer-based model using the provided tokenized dataset 
    and hyperparameters.

    Args:
        - fancy_name (str): A fancy name to identify the training run.
        - tokenized_dataset (datasets.Dataset): The tokenized dataset for 
            training, validation, and testing.
        - model_name (str): The name or path of the pre-trained transformer model.
        - num_labels: The number of labels/classes for the classification task.
        - tokenizer (transformers.AutoTokenizer): The tokenizer for tokenizing 
            the input sequences.
        - compute_metrics (Callable): A function to compute evaluation metrics.
        - hyper_params (dict, optional): Specific hyperparameters for training 
            the model. Defaults to {}.
        - push_to_hub (bool, optional): Whether to push the trained model to   
            the Hugging Face model hub. Defaults to False.
        - evaluate (bool, optional): Whether to evaluate the trained model on 
            the validation and test datasets. Defaults to True.

    Returns:
        Trainer: The trainer object used for training the model.
    """
    
    #if push_to_hub:
    #    huggingface_setup()
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    batch_size = hyper_params.get("batch_size", 8)
    args = TrainingArguments(
        output_dir=fancy_name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=hyper_params.get("lr", 1.93009e-05),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=hyper_params.get("epochs", 4),
        warmup_steps=hyper_params.get("warmup_steps", 0),
        weight_decay=hyper_params.get("weight_decay", 0.01),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=push_to_hub,
        max_steps=-1,
        report_to=None
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    if evaluate:
        print("\nFinal model evaluation:")
        trainer.evaluate()
        print("\nResults over test data:")
        preds = trainer.predict(tokenized_dataset["test"])
        print(preds.metrics)
    
    return trainer


def hp_tune_transformer(
        fancy_name: str,
        model_name: str,
        tokenized_dataset: Dataset,
        num_labels: int,
        tokenizer: AutoTokenizer,
        compute_metrics: Callable, 
        num_samples: int = 8, 
        cpus_per_trial: int = 1, 
        gpus_per_trial: int = 0, 
        report_to: Optional[str] = None,
        save_best_model: bool = False,
        smoke_test: bool = False
    ):
    """
    Perform hyperparameter search using Population-Based Training (PBT) to 
    find optimal hyperparameters
    for training a transformer model.

    Args:
        - fancy_name (str): Name for the fancy experiment.
        - model_name (str): Name or path of the pre-trained transformer model.
        - tokenized_dataset (Dataset): Tokenized dataset containing train, 
            validation, and test splits.
        - num_labels (int): Number of labels/classes in the dataset.
        - tokenizer (AutoTokenizer): Tokenizer for the transformer model.
        - compute_metrics (Callable): Function to compute evaluation metrics.
        - num_samples (int, optional): Number of hyperparameter samples to try. 
            Defaults to 8.
        - cpus_per_trial (int, optional): Number of CPUs to allocate per trial. 
            Defaults to 1.
        - gpus_per_trial (int, optional): Number of GPUs to allocate per trial. 
            Defaults to 0.
        - report_to (Optional[str], optional): Reporting destination (e.g., 
            'wandb' for Weights & Biases).Defaults to None.
        - save_best_model (bool, optional): Whether to save the best model. 
            Defaults to False.
        - smoke_test (bool, optional): Whether to perform a smoke test. 
            Defaults to False.

    Returns:
        BestRun: The best hyperparameter run.
    """
    print("\nSTARTING HYPER-PARAMETER SEARCH")
    print("\nDownloading and caching pre-trained model")
    model_config = AutoConfig.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    # Triggers model download to cache
    AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config,
    )

    def get_model():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_config,
        )

    # Sets training and hyperparameter search configuration
    training_args = TrainingArguments(
        output_dir=fancy_name,
        learning_rate=1e-5,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_train_epochs=2,  # config
        max_steps=-1,
        per_device_train_batch_size=16,  # config
        per_device_eval_batch_size=16,  # config
        warmup_steps=0,  # config
        weight_decay=0.01,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to=report_to
    )

    trainer = Trainer(
        model_init=get_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )

    tune_config = {
        "per_device_train_batch_size": tune.choice([8, 16, 32, 64]),
        "per_device_eval_batch_size": tune.choice([8, 16, 32, 64]),
        "num_train_epochs": tune.choice([2, 3, 4, 5]),
        "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
    }

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=1,
        hyperparam_mutations={
            "learning_rate": tune.uniform(1e-5, 1e-6),
            "weight_decay": tune.uniform(0.0, 0.5),
            "warmup_steps": tune.randint(0, 500),
            "per_device_train_batch_size":  tune.choice([8, 16, 32, 64]),
        },
    )

    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "warmup_steps": "warmup_steps",
            "learning_rate": "lr",
            "per_device_train_batch_size": "train_bs/gpu",
            "num_train_epochs": "num_epochs",
        },
        metric_columns=[
            "epoch", "training_iteration", "eval_loss", "eval_accuracy", 
            "eval_f1", "eval_precision", "eval_recall"
        ],
    )
    
    ray.shutdown()
    ray.init(log_to_driver=True, ignore_reinit_error=True)

    if report_to == "wandb":
        wandb_setup(project_name=fancy_name)

    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=num_samples,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        scheduler=scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="ray_results/",
        name=fancy_name + "-tune_transformer_pbt",
        log_to_file=not(bool(report_to)),
    )
    
    print("\nFinal model evaluation:")
    best_run.evaluate()
    print("\nResults over test data:")
    preds = best_run.predict(tokenized_dataset["test"])
    print(preds.metrics)
    
    print("\nSaving results...")
    save_json(best_run.hyperparameters, 
              path=fancy_name + "-best_hyperparameters.json")
    
    if save_best_model: 
        best_run.save_model(fancy_name + "-model")
    
    return best_run


def predict_from_transformer(
        
        dataset: Dataset, 
        checkpoint_or_path: str,
        mode: str = "label", 
        local_files_only: bool = False,
        tokenizer_kwargs: dict = {}
        
) -> np.array:
    """
    Performs inference using a pre-trained transformer model on a given dataset.

    Args:
        - dataset (Dataset): The dataset to perform inference on. Must 
            contain a "text" column.
        - checkpoint_or_path (str): The checkpoint or path to the pre-trained model.
        - mode (str, optional): The mode for prediction. Possible values are 
            'label' or 'proba'. 'label' returns the predicted label for each 
            sample.'proba' returns the predicted probabilities for each class.
            Defaults to 'label'.
        - local_files_only (bool, optional): Whether to only use local files 
            when loading the model. Defaults to False.
        - tokenizer_kwargs (dict, optional): Additional keyword arguments to 
            be passed to the tokenizer. Defaults to an empty dictionary.

    Returns:
        numpy.array: An array of predicted labels or probabilities.
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_or_path, local_files_only=local_files_only
    )
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_or_path, local_files_only=local_files_only
    )
    
    # Create a text classification pipeline
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=None)
    
    preds = []
    for out in tqdm(pipe(KeyDataset(dataset, "text"), **tokenizer_kwargs)):
        if mode == "label":
            preds.append(out[0]["label"])  # Append the predicted label
        elif mode == "proba":
            out.sort(key=lambda x: x["label"])
            preds.append([class_["score"] for class_ in out])  # Append the predicted probabilities for each class
    
    return np.array(preds)
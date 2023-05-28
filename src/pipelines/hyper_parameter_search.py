# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:24:56 2023

@author: alvar
"""

import argparse
import pandas as pd
from ..utils.modeling import SexismDataset, hp_tune_transformer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Hyper-parameter search for fine-tuning a large \
                     language model for text classification'
    )
    parser.add_argument(
        "data_path",
        action="store",
        default=".",
        help="Path to the data used to train the model."
    )
    parser.add_argument(
        "save_path",
        action="store",
        default=".",
        help="Path to the folder where the model and other files will be saved."
    )
    parser.add_argument(
        "model",
        dest="model_choice"
        action="store",
        default="beto",
        help="HuggingFace model to fine-tune. Supported models are 'beto', \
              'roberta', and 'bertin'."
    )
    parser.add_argument(
        "task"
        dest="task_choice",
        action="store",
        default="detection",
        help="The task to perform. Must be either 'detection' or \
              'classification' for sexism task."
    )
    parser.add_argument(
        "-i",
        "--n-instances",
        metavar="N",
        type=int,
        dest="max_instances",
        action="store",
        default=None,
        help="The number of data instances to use for training. If not \
              specified, all data will be used. Can be either the proportion \
              or the absolute value."
    )
    parser.add_argument(
        "-r",
        "--runs",
        metavar="N",
        type=int,
        dest="runs",
        action="store",
        default=12,
        help="The number of runs (models to train)."
    )
    parser.add_argument(
        "-c",
        "--cpus",
        metavar="N",
        type=int,
        dest="ncpus",
        action="store",
        default=1,
        help="The number of available CPUs."
    )
    parser.add_argument(
        "-g",
        "--gpus",
        metavar="N",
        type=int,
        dest="ngpus",
        action="store",
        default=1,
        help="The number of available GPUs."
    )
    parser.add_argument(
        "-w",
        "--report-to-wandb",
        dest="to_wandb",
        action="store_true",
        default=True,
        help="Send log reports to W&B (requires authentication)."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Finish quickly for testing purposes."
    )
    args = parser.parse_args()
    return args



def hp_search(args):
    if args.model_choice == "beto":
        MODEL_CHECKPOINT = "dccuchile/bert-base-spanish-wwm-uncased"
    elif args.model_choice == "bertin":
        MODEL_CHECKPOINT = "bertin-project/bertin-roberta-base-spanish"
    elif args.model_choice == "roberta":
        MODEL_CHECKPOINT = "PlanTL-GOB-ES/roberta-base-bne"
    #elif args.model_choice == "robertuito":
    #    MODEL_CHECKPOINT = "pysentimiento/robertuito-base-uncased"


    print("\nPreparing and Tokenizing Dataset")
    # prepare the data for the task
    msd = SexismDataset(filepath_or_df=args.data_path, 
                        task_name=args.task_name, 
                        model_checkpoint=MODEL_CHECKPOINT,
                        max_instances = args.max_instances)
    
    fancy_name = f"{args.save_path}/{args.model_choice}-sexism-{args.task_choice}"
    report_to = "wandb" if args.to_wandb else None
    
    if args.smoke_test:
        hp_tune_transformer(fancy_name=fancy_name,
                            model_name=MODEL_CHECKPOINT,
                            tokenized_dataset=msd.tokenized,
                            num_labels=msd.num_labels,
                            tokenizer=msd.tokenizer,
                            compute_metrics=msd.build_compute_metrics(),
                            num_samples=1, 
                            cpus_per_trial=1, 
                            gpus_per_trial=1,
                            report_to=report_to,
                            smoke_test=True)
    else:
        hp_tune_transformer(fancy_name=fancy_name,
                            model_name=MODEL_CHECKPOINT,
                            tokenized_dataset=msd.tokenized,
                            num_labels=msd.num_labels,
                            tokenizer=msd.tokenizer,
                            compute_metrics=msd.build_compute_metrics(),
                            num_samples=args.runs, 
                            cpus_per_trial=args.cpus, 
                            gpus_per_trial=args.gpus,
                            report_to=report_to,
                            smoke_test=False)

if __name__ == "__main__": 
    args = parse_args()
    hp_search(args)
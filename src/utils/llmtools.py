# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:28:01 2023

@author: alvar
"""


import pandas as pd
from datasets import Dataset

import tiktoken
import openai 

from typing import List, Dict, Tuple, Iterator, Union, Any, Optional, Callable

from utils.modeling import predict_from_transformer


###############################################################################
# OPENAI GPT TRANSLATION
###############################################################################

def split_texts_given_max_tokens(
        
        texts: pd.Series, 
        gpt_model: str, 
        max_tokens: int

    ) -> Iterator[Tuple[list, list]]:
    """
    Split long pd.Series of texts into small batches optimized to fit into 
    max_tokens allowed.
    
    Args:
        - texts: (large amount of) indexed tweets to be splitted.
        - gpt_model: GPT model name used for token encoding.
        - max_tokens: maximum number of GPT tokens 
            (computed using tiktoken.encoding_for_model.encode method).
        
    Yield:
        batches of text with their respective indexes (ids).
    """
    
    # Initialize variables
    ids, batch, aux = [], [], 0
    
    # Get encoding for GPT model
    encoding = tiktoken.encoding_for_model(gpt_model)
    
    # Iterate over texts
    for i, txt in texts.items():
        # Get the length of the tweet in tokens
        len_tweet = len(encoding.encode(txt))
        
        # Check if adding the current tweet exceeds the maximum tokens
        if aux + len_tweet < max_tokens:
            # Add tweet and index to the current batch
            batch.append(txt)
            ids.append(i)
            aux += len_tweet
        else:
            # Yield the current batch if the maximum tokens are reached
            yield ids, batch
            
            # Reset batch variables for the next iteration
            ids, batch, aux = [i], [txt], len_tweet
    
    # Yield the last batch if it exists
    if batch:
        yield ids, batch


        
def gpt_translation(texts: list, input_language: str, output_language: str, model: str) -> list:
    """
    Translate text using OpenAI's GPT. 
    
    PARAMETERS:
        - texts: list of raw text to be translated (total tokens contained 
            must not exceed 1500 aprox)
        - input_language: original text language.
        - output_language: language to be translated.
        - model: GPT model name (expected for openai.ChatCompletion.create method)
        
    RETURNS translated texts.
    """
    
    # Set the system message for GPT model
    context = (
        f"Assistant is an intelligent chatbot designed to translate {input_language} tweets into {output_language}.\n" \
        "Instructions:\n" \
        " - You will be provided with one tweet per line. Each line contains tweet's id (first number) and the tweet.\n" \
        " - Ensure that the translations are accurate and preserve the original meaning and tone of each tweet.\n" \
        " - Take into account any slang or informal language used in the tweets, as well as any potential variations in spelling or grammar." \
        " - You must sound as a native speaker when translating tweets."
        " - Each line in your response must correspond to each tweet provided, in the same order and starting with the corresponding tweet's id.\n" \
        " - If you are not able to translate one of them, just omit it"
    )
    
    # Ask GPT for the translation
    return openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": "\n".join(texts)}
        ]
    ).choices[0].message.content.split("\n")


def specific_translation(
        
        texts: pd.Series, 
        original: str, 
        new: str, 
        model_name: str,
        max_tokens: int,
        verbose: bool = False
        
    ) -> Tuple[pd.DataFrame, bool]:
    """
    Translate the pandas Series passed.
    
    Args:
        - texts: indexed tweets to be translated
        - original: data source language
        - new: language to get translations
        - model_name: GPT model name for translation
        - max_tokens: maximum number of tokens allowed for translation
        - verbose: flag to enable verbose output
        
    Returns: 
        pandas.DataFrame: translated text (maintaining tweets' original 
            associated indexes) and a flag indicating success.
    """
    translated, translated_ids, original_ids = [], [], []
    time_buffer, requests_buffer = 0, 0
    
    # Calculate the number of tokens allowed for translation
    allowed_tokens = int((max_tokens-1000)/2)
    
    # For each batch of tweets
    for i, (ids, batch) in enumerate(split_texts_given_max_tokens(texts, 
                                                                  model_name, 
                                                                  allowed_tokens)):
        t0 = time.time()

        # Transform texts into the form <id_tweet><blank_space><tweet> for a better correspondance with response
        proper_input = pd.Series(ids).astype(str) + " " + pd.Series(batch)
        
        # Ask GPT to translate the texts
        try:
            response = gpt_translation(proper_input, input_language=original, 
                                       output_language=new, model=model_name)
        except Exception as ex:
            print(f"Oops, there were some issues while translating data: {ex}\n")
            return pd.DataFrame({"original_id": translated_ids, "translated": translated}), False
        
        # Process each translation (response is a list of strings)
        retrieved, correct, incorrect = 0, 0, 0
        for sent in response:
            
            # Normal cases with the same format as the input
            try:
                # Match with the first blank space to split the sentence into id and text
                pos = sent.index(" ")  
                id_ = int(sent[:pos].strip(".").strip(",").strip(":"))  # Get tweet id
                trans = sent[pos+1:]  # Get translated tweet
                correct += 1
            
            # No spaces, just a number; GPT model did not return the sentence translated or the model
            # did not return the id correctly
            except ValueError: 
                continue
            
            # Accumulate ids and text from the response
            translated_ids.append(id_)
            translated.append(trans)
            retrieved += 1
        
        # Accumulate original ids for checking purposes
        original_ids.extend(ids) 

        t1 = time.time()
        
        if verbose:
            print(f"Batch: {i:3} " \
                  f"| Sent: {len(ids)} " \
                  f"| Retrieved: {retrieved}; Translated: {round(correct/retrieved,2)*100}% " \
                  f"| Time: {prettytime(t1-t0)}")
        
        # Control the number of requests to the API
        time_buffer += t1-t0
        requests_buffer += 1
        if requests_buffer == RPM:
            if time_buffer < 60:
                time.sleep(60-time_buffer)
            time_buffer, requests_buffer = 0, 0
            
    return pd.DataFrame({"original_id": translated_ids, "translated": translated}), True


@timeinfo
def llm_translation(
    
        texts: Union[list, pd.Series], 
        original: str, 
        new: str, 
        model_settings: dict = {},
        callback: str = "simple", 
        verbose: bool = False,
        tol: int = 10
    
    ) -> pd.DataFrame:
    """
    Translate the given texts.
    
    PARAMETERS:
        - texts: indexed tweets to be translated
        - original: data source language
        - new: language to get translations
        - model_settings: dictionary containing model settings (name and max_tokens)
        - callback: callback type for handling translation failures
        - verbose: flag to enable verbose output
        - tol: tolerance value for remaining untranslated texts in exhaustive mode
        
    RETURNS translated Series (maintaining tweets' original associated indexes)
    """
    if isinstance(texts, list):
        texts = pd.Series(texts)
    
    # Warming up the model
    _ = gpt_translation(texts[:5], original, new, "gpt-3.5-turbo")
    
    if verbose:
        print(f"Starting translation. Callback: {callback}\n")
        
    if callback == "simple":
        # Just calls the default function, without considering connection errors or similar
        translated_text, _ = specific_translation(texts, 
                                                  original=original, 
                                                  new=new, 
                                                  model_name=model_settings.get("name", "gpt-3.5-turbo"),
                                                  max_tokens=model_settings.get("max_tokens", 4096),
                                                  verbose=verbose)

    elif callback == "retrial":
        # Prevents failures due to unexpected errors with the API connection
        translated_text = pd.DataFrame()
        _ok, start = False, texts.index[0]
        while not _ok:
            translated, _ok = specific_translation(texts.loc[start:], 
                                                   original=original, 
                                                   new=new, 
                                                   model_name=model_settings.get("name", "gpt-3.5-turbo"),
                                                   max_tokens=model_settings.get("max_tokens", 4096),
                                                   verbose=verbose)
            translated_text = pd.concat([translated_text, translated], axis=0)
            if _ok:
                break

            print(f"Translation broken at {dt.datetime.now()}")
            start = translated_text.original_id.iloc[-1] + 1
            print(f"Continuing translation phase with text {len(translated_text)+1} out of {len(texts)}.")

            try:
                # Wait for one minute before trying again
                print(f"Waiting one minute before restarting translation. \
                        Press Ctrl+C to stop and get the translated text until now.\n")
                time.sleep(60)
            except KeyboardInterrupt:
                return translated_text
        
    elif callback == "exhaustive":
        # Apart from preventing unexpected errors, calls to translate until 
        # just a constant number (tol) of texts remain untranslated
        translated_text = pd.DataFrame()
        orig_ids = set(texts.index)
        failed = orig_ids - set(translated_text.original_id)
        
        while len(failed) > tol:
            translated = translate(texts.loc[list(failed)], 
                                   original=original, 
                                   new=new, 
                                   call="retrial", 
                                   verbose=verbose)
            # Accumulate translated texts
            translated_text = pd.concat([translated_text, translated], axis=0)
            
            # Check remaining texts to translate
            failed = orig_ids - set(translated_text.original_id)
            
            print(f"Translated {len(translated_text)} out of {len(texts)}. \
                    Tolerance: {tol}. Relaunching translation with {failed} remaining.")
            
            # Wait for half a minute before trying again
            time.sleep(30)
    
    else:
        raise ValueError("'callback' must be 'simple', 'retrial', or 'exhaustive'")
    
    return translated_text


###############################################################################
# :) Hugging Face Large Language Model for Feature Extraction 
###############################################################################

@timeinfo
def llm_feature_extraction(
        
    df: pd.DataFrame, 
    models: List[Dict[str, Union[str, Callable]]], 
    verbose: bool = False,
    return_Dataset: bool = False
    
) -> pd.DataFrame:
    """
    Extracts features using language models specified in the 'models' parameter.

    Args:
        df (pd.DataFrame): Input DataFrame containing the text data.
        models (List[Dict[str, Union[str, Callable]]]): List of model settings
            including the model name, output column name, and optional 
            cleaning function.
        verbose (bool, optional): Whether to print extraction progress. 
            Defaults to False.
        return_Dataset (bool, optional): Whether to return the Dataset 
            object instead of converting it to a pandas DataFrame. 
            Defaults to False.
    
    Returns:
        pd.DataFrame or datasets.Dataset: Extracted features as a DataFrame 
            if 'return_Dataset' is False, otherwise, the Dataset object 
            containing the extracted features.
    """
    # Convert input DataFrame to Hugging Face Dataset
    dataset = datasets.Dataset.from_pandas(df)
    
    # Remove redundant index column if present
    if "__index_level_0__" in dataset.column_names:
        dataset.remove_columns("__index_level_0__")
    
    # Iterate over each model to extract features
    for model_settings in models:
        if verbose:
            print(f"Extracting \'{model_settings['out_col']}\' feature...")
        
        out_col = model_settings['out_col']
        
        try:
            # Perform predictions using the specified language model
            preds = predict_from_transformer(dataset, model_settings["model_name"])
        except Exception as ex:
            print(f"Oops, encountered an error: {ex}")
            if return_Dataset:
                return dataset
            return dataset.to_pandas(), dataset
        
        # Add the extracted feature column to the Dataset
        dataset = dataset.add_column(model_settings['out_col'], preds)
        
        # Apply optional cleaning function to the extracted feature column
        clean_func = model_settings.get("clean_label", None)
        if clean_func:
            def clean_label(example):
                example[out_col] = clean_func(example[out_col])
                return example
            dataset = dataset.map(clean_label)
    
    # Return the Dataset object if specified
    if return_Dataset:
        return dataset
    # Otherwise, convert the Dataset to a pandas DataFrame and return
    return dataset.to_pandas()
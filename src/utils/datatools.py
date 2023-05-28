# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:26:50 2023

@author: alvar
"""


import collections
import json
import pandas as pd
import numpy as np
import re
import time
import datetime as dt
import scipy.stats as st
import string
from copy import deepcopy
from typing import List, Dict, Tuple, Iterator, Union, Any, Optional, Callable

# pip install required 
import emoji
import unidecode
import nltk
#nltk.download('punkt')
from nltk.corpus import stopwords, wordnet
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

# custom modules
from utils.misc import prettytime, timeinfo


###############################################################################
# HOMOGENIZATION
###############################################################################

label2int = lambda x, positive: 1 if x == positive else 0
int2label = lambda x, positive, negative: positive if x else negative


def homogenize(
    
        *datasets: List[pd.DataFrame],
        common_columns = list,
        column_mapping: Dict[Union[str, None], Union[str, list]] = {},
        label_positive: Optional[Union[bool, str]] = None,
        target_column: Optional[str] = None,
        constant_columns: Optional[Tuple[str, str]] = None,
        language: Optional[str] = None,
        dataset_name: Optional[str] = None
    
    ) -> pd.DataFrame:
    """
    Homogenize the format of multiple data sources into a common format.

    PARAMETERS: 
        - datasets: A list of DataFrame objects to homogenize (expecting 
            the same format across datasets). The starred expression 
            allows passing multiple pandas.DataFrame objects directly.
        - common_columns: A list of column names that should be present in the 
            final homogenized data.
        - column_mapping: A dictionary specifying how to map actual column 
            names to the required common column names if they are different.
        - label_positive: The positive label to consider when converting the 
            target column to an integer (if needed).
        - target_column: The name of the target column to be converted to an 
            integer (if needed).
        - constant_columns: A tuple specifying source-specific constant 
            columns to be added to the homogenized DataFrame. It should contain 
            pairs of column names and their corresponding constant values.
    
    RETURNS: The homogenized data as a pandas DataFrame with the specified 
        common columns.
    """

    if len(datasets) == 1:
        aux = datasets[0].copy()
    else: 
        aux = pd.concat(datasets, ignore_index=True)

    # some relevant variables
    len_df = len(aux)

    # renaming columns to common names
    aux = aux.rename(columns=column_mapping) 

    # astype target column to int if needed
    if label_positive:

        # if its boolean, astyping directly more efficient
        if isinstance(label_positive, bool): 
            aux[target_column] = aux[target_column].astype(int)

        # if its string, conventional label to int
        elif isinstance(label_positive, str): 
            aux[target_column] = aux[target_column].apply(label2int, 
                                                          positive=label_positive)

    # add source specific constant columns        
    for colname, value in constant_columns:
        aux[colname] = [value] * len_df

    # select common columns
    aux = aux.loc[:, common_columns]

    return aux


def homogenize_column(df: pd.DataFrame, column: str, values_mapping: dict):
    df = df.copy()
    for new, actual in values_mapping.items():
        df[column] = df[column].apply(lambda x: new if x in actual else x)
    return df

    
    
###############################################################################
# INTEGRATION
###############################################################################


IntegrationSchemaType = Dict[str, Dict[str, Dict[str, Union[str, bool, List[str], Dict[str, str]]]]]
@timeinfo
def integrate_sources(
    
    schema: IntegrationSchemaType,
    target_col: str,
    common_columns: List[str],
    extras: Optional[List[Tuple[Callable, str]]] = None,
    original_data_path: Optional[str] = ".",
    save: bool = False,
    save_data_path: Optional[str] = "."
    
) -> pd.DataFrame:
    """
    Homogenize and integrate all the data sources specified in the integration 
    schema.

    PARAMETERS:
        - schema: Integration schema specifying the data sources and their 
            corresponding parameters. Further explained below.
        - target_col: Name of the target column for the integrated data.
        - common_columns: List of common columns across all data sources.
        - extras: Optional list of additional functions to apply on the 
            integrated data. Callables must take as first positional input
            a pandas.DataFrame and must return another one. 
        - original_data_path: Optional path to the original data folder.
        - save: Flag indicating whether to save the integrated data.
        - save_data_path: Optional path to save the integrated data.

    RETURNS: Integrated homogenized data.
    
    The integration schema is a dictionary that specifies the
    configuration for integrating multiple data sources. 
    It consists of key-value pairs, where each key represents 
    a data source folder name in the original data path.

    For each data source, the schema contains two sub-dictionaries: 
    "read" and "homogenize". The "read" dictionary specifies the 
    parameters to be used when reading the files within the data 
    source folder using pandas.read_csv. It includes a "files" key, 
    which holds a list of relative filenames to be read. 
    Additionally, it may contain a "kwargs" key that allows for 
    specifying additional keyword arguments required for reading the files.

    The "homogenize" dictionary corresponds to the keyword arguments 
    used for the data homogenization process. It includes a "column_mapping" 
    key, which maps the actual column names in the data source to the 
    required common column names for integration. If any column names differ 
    between the data source and the common columns, they should be specified 
    in the mapping.

    If necessary, the "homogenize" dictionary can also include a 
    "label_positive" key, which defines the positive label value for the 
    target column. Moreover, there is a "constant_columns" key, which is 
    a list of tuples specifying the required common columns that are not 
    present in the data source. These tuples consist of the column name and 
    either None or a constant value for that column.

    To create an integration schema template, you can use the 
    print_integration_schema_template() function, which prints a sample 
    schema structure. You can replace the 
    <data_source1_folder_name_in_original_data_path> with the actual 
    folder name and follow the same structure for other data sources 
    in the schema.
    
    """
    buffer = []  # Buffer to store data sources homogenized

    for dataset_name_, kwargs in schema.items():
        # Prevent different naming for multilingual cases
        folder_name = dataset_name_.split("_")[0]  
        folder_path = f"{original_data_path}/{folder_name}/"

        # Read data files contained in each data source
        kwargs_read = kwargs["read"].get("kwargs", {})
        datasets = [ pd.read_csv(folder_path + file, **kwargs_read) 
                     for file in kwargs["read"]["files"] ]  

        # Homogenize data source
        if "constant_columns" in kwargs['homogenize']:
            kwargs['homogenize']["constant_columns"].append(("dataset", folder_name))
        else:
            kwargs['homogenize']["constant_columns"] = [("dataset", folder_name)]

        homogenized = homogenize(*datasets,
                                 **kwargs['homogenize'],
                                 common_columns=common_columns,
                                 target_column=target_col)

        buffer.append(homogenized)

    # Concatenate all homogenized data sources stored in the buffer
    integrated = pd.concat(buffer, axis=0).reset_index(drop=True)

    # Apply additional functions on the integrated data
    if extras:
        for func, kwargs in extras:
            integrated = func(integrated, **kwargs)

    # Save results if specified
    if save:
        path = f"{save_data_path}/integrated_data.csv"
        integrated.to_csv(path, sep=";")
        print(f"Data correctly integrated and saved in \033[1m'{path}'\033[0m")

    return integrated


def print_integration_schema_template():
    print("""
    schema = {
     <data_source1_folder_name_in_original_data_path>: {
         "read": {
             "files" : [
                 <relative_filename1>,
                 <relative_filename2>,
                 ...
             ],
             "kwargs": {
                 <pandas_read_csv_kwarg1>: value_kwarg1,
                 <pandas_read_csv_kwarg2>: value_kwarg2,
                 ...
             } (if needed)                   
         },
         "homogenize": {
             "column_mapping": {
                 <actual_column_name1_in_data_source>: <required_common_column_name1>, (if different)
                 <actual_column_name2_in_data_source>: <required_common_column_name2>, (if different)
                 ... 
             },
             "label_positive": value (if needed)
             "constant_columns": [
                 (<required_common_column1_not_in_source>, None),
                 (<required_common_column2_not_in_source>, None),
                 ...
                 (<col_name1>, <const_value1>), 
                 (<col_name2>, <const_value2>), 
                 ...
             ]
         }
     },
     <data_source2_folder_name_in_original_data_path>: {
         ...
     }
    }
    """)
    

###############################################################################
# TEXT CLEANING
###############################################################################

def clean_text(
    
        text: str, 
        keep_case: bool = False,
        keep_accents: bool = False,
        keep_numbers: bool = False,
        lmhe_tokens: Optional[Dict[str, str]] = None, 
        constraints: Optional[List[Tuple[str, str]]] = None,
        allowed_punctuation: Optional[str] = None
    
    ) -> str:
    """
    Clean a given text
    
    PARAMETERS:
        - text: string to clean
        
        - keep_case: wether to keep original case (True) or not (False). 
            If not text will be lowercased.
        
        - keep_accents: wether to keep accents (True) or not (False).
        
        - keep_numbers: wether to keep numbers (True) or not (False).
        
        - lmhe_tokens: which stands for link-mention-hashtag-emoji_tokens. 
            A dict containing how to represent those items in the final text. 
            If nothing provided (neither dict or specific key), they will be removed.
        
        - constraints: any special substitution you may want to apply to the 
            text. It must be a list of tuples containing the corresponding 
            regex to capture (first element of the tuple) and the string to 
            substitue it (second element).
        
        - allowed_punctuation: string containing custom punctuation you may 
            want to avoid cleaning.
        
    RETURNS cleaned text
    """
    
    # lowercase
    if not keep_case:
        text = text.lower() 
    
    #remove \n and \r
    text = text.replace('\r', '').replace('\n', ' ')
    
    if lmhe_tokens is not None:
        # handle links
        text = re.sub(r'(?:www\.|https?://)\S+', lmhe_tokens.get("link", ''), text, flags=re.MULTILINE)  
        
        # handle mentions
        text = re.sub(r'\@\S+', lmhe_tokens.get("mention", ''), text) 
        
        # handle hashtags
        text = re.sub(r'#\S+', lmhe_tokens.get("hashtag", ''), text)
        
        # handle emojis
        text = emoji.replace_emoji(text, lmhe_tokens.get("emoji", ''))  
        
    else:
        # remove links, mentions, hashtags and emojis
        text = re.sub(r'(?:#|\@|www\.|https?://)\S+', '', text, flags=re.MULTILINE) 
        text = emoji.replace_emoji(text, '')
     
    # specific constraints
    if constraints is not None:
        for regex, token in constraints:
            text = re.sub(regex, token, text, flags=re.I)
    
    # remove accents
    if not keep_accents:
        text = unidecode.unidecode(text)  
    
    ## all symbols and punctuation
    banned_list = string.punctuation + 'Ã'+'±'+'ã'+'¼'+'â'+'»'+'§'+'—'  
    ## allowed punctuation
    if allowed_punctuation is not None:  
        banned_list = re.sub(r"[%s]" % re.escape(allowed_punctuation), "", banned_list)
    # remove symbols and punctuation
    text = text.translate(str.maketrans('', '', banned_list)) 
    
    # remove numbers
    if not keep_numbers:
        text = re.sub(r'\d+', '', text)  
    
    # remove extra and leading blanks
    text = re.sub("\s\s+" , " ", text).strip()  
    
    return text


def remove_punctuation(text):
    """
    Removes punctuation marks from the text.
    
    Parameters:
    - text: Input text
    
    Returns:
    - Text with punctuation removed
    """
    return re.sub(f"[{re.escape(string.punctuation)}]", "", text)


def remove_numbers(text):
    """
    Removes numbers from the text.
    
    Parameters:
    - text: Input text
    
    Returns:
    - Text with numbers removed
    """
    return re.sub(r"\d+", "", text)


def remove_stopwords(text, lang):
    """
    Removes stopwords from the text based on the specified language.
    
    Parameters:
    - text: Input text
    - lang: Language code
    
    Returns:
    - Text with stopwords removed
    """
    return re.sub(fr'\b(?:{"|".join(stopwords.words(lang))})\b', "", text)


def deepclean(texts, lang):
    """
    Applies a series of text cleaning steps to the input texts.
    
    Parameters:
    - texts: Series of texts to clean
    - lang: Language code
    
    Returns:
    - Cleaned texts
    """
    return texts.apply(lambda x: " ".join(nltk.word_tokenize(x.lower(), language=lang))) \
                .apply(remove_punctuation) \
                .apply(remove_numbers) \
                .apply(remove_stopwords, lang=lang) \
                .apply(lambda x: re.sub("\s+", " ", x))
                
                
###############################################################################
# Sampling
###############################################################################

def balance_data_with_undersampling(df, positive_ratio) -> pd.DataFrame:
    """
    Balances the data by performing undersampling on the majority class.

    This function takes a DataFrame containing labeled data and performs undersampling
    on the majority class to balance the distribution of positive and negative instances.
    It ensures that the positive-to-negative ratio in the resulting dataset is not higher
    than the specified positive ratio.

    Args:
        - df (pandas.DataFrame): The DataFrame containing the labeled data.
        - positive_ratio (float): The desired positive-to-negative ratio after balancing.

    Returns:
        pandas.DataFrame: The balanced dataset after performing undersampling.

    """
    df = df.copy()
    
    # Calculate the actual positive count and ratio in the data frame
    actual_pos_count = df.label.sum()
    actual_pos_ratio = actual_pos_count / len(df)
    
    # Check if the actual positive ratio is higher than the desired positive ratio
    _changed = False
    if actual_pos_ratio > positive_ratio:
        # Invert the labels if the actual positive ratio is higher
        df.label = df.label.apply(lambda x: not x)
        actual_pos_count = len(df) - actual_pos_count
        actual_pos_ratio = 1 - actual_pos_ratio
        positive_ratio = 1 - positive_ratio
        _changed = True
    
    # Get all the positive instances
    pos_instances = df.loc[df.label == 1, :]
    
    # Undersample the negative class
    neg_instances_allowed = int(actual_pos_count * ((1 - positive_ratio) / positive_ratio))
    neg_instances = df.loc[df.label == 0, :].sample(n=neg_instances_allowed)
    
    # Concatenate positive and negative instances to create a balanced dataset
    balanced_data = pd.concat([pos_instances, neg_instances], axis=0)
    
    # Revert the label inversion if performed earlier
    if _changed:
        balanced_data.label = balanced_data.label.apply(lambda x: not x).astype(int)
    
    return balanced_data


def stratified_sample(
    
    data: pd.DataFrame, 
    column: str, 
    confidence: int,
    random_state: Optional[int] = None, 
    shuffle: bool = False
    
) -> pd.DataFrame:
    """
    Returns a stratified sample from a DataFrame, using a specified column as strata and a 
    dictionary of sample sizes for each stratum.
    
    Args:
        data (pd.DataFrame): DataFrame to sample from.
        column (str): Name of the column to use as strata.
        confidence (int): Level of confidence for the sample.
        random_state (int or None, optional): Seed for the random number generator. Default is None.
        shuffle (bool, optional): Whether to shuffle the final sample. Defaults to False.
    
    Returns:
        pd.DataFrame: Stratified sample of the original DataFrame.
    """
    # Initialize the list to store the stratified samples
    sample_list = []
    
    # Calculate the relative frequency of each stratum in the population
    stratum_counts = data[column].value_counts()
    stratum_props = stratum_counts / data.shape[0]
    
    # Calculate the sample size for each stratum using the proportion and the desired overall sample size
    for stratum in stratum_props.index:
        
        stratum_data = data[data[column] == stratum]
        
        # Calculate sample size: n = (z^2 * p * (1-p)) / error^2
        # z-value corresponding to the desired level of confidence
        z = st.norm.ppf(confidence)
        # estimated proportion of the population 
        p = float(stratum_props[stratum])
        # proper stratum size
        stratum_size = round(((z**2)*p*(1-p))/((1-confidence)**2))
        
        # Sample from the current stratum
        stratum_sample = stratum_data.sample(n=stratum_size, random_state=random_state)
        sample_list.append(stratum_sample)
    
    # Concatenate the samples from each stratum into a single DataFrame
    sample = pd.concat(sample_list, axis=0)
    
    # Shuffle the final sample if specified
    if shuffle:
        sample = sample.sample(len(sample))
    
    return sample
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:28:29 2023

@author: alvar
"""

import json
import spacy
import yake
import pandas as pd
from copy import deepcopy
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from misc import timeinfo


def get_keywords_tfidf(texts: list, top: int):
    """
    Extracts the top keywords from a list of texts using TF-IDF.
    
    Args:
        texts (list): List of texts.
        top (int): Number of top keywords to extract.
        
    Returns:
        list: List of top keywords.
    """
    # Create a TfidfVectorizer object to generate the TF-IDF matrix
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Get the list of words in order of importance based on their TF-IDF score
    word_importance = [(word, tfidf_matrix.getcol(idx).sum())
                       for word, idx in vectorizer.vocabulary_.items()]
    word_importance = sorted(word_importance, key=lambda x: -x[1])

    # Return the 'top' most important words
    return [word for word, score in word_importance[:top]]


def get_keywords_counts(texts: list, top: int):
    """
    Extracts the top keywords from a list of texts based on their frequency counts.
    
    Args:
        texts (list): List of texts.
        top (int): Number of top keywords to extract.
        
    Returns:
        list: List of top keywords.
    """
    counter = Counter(texts)
    del counter["''"]
    return [word for word, count in counter.most_common(top)]


def get_keywords_spacy(texts: pd.Series, top: int, spacy_background, sintaxis: list):
    """
    Extracts the top keywords from a Pandas Series of texts using Spacy and 
    syntactic categories.
    
    Args:
        texts (pd.Series): Pandas Series of texts.
        top (int): Number of top keywords to extract.
        spacy_background: Pre-initialized Spacy model.
        sintaxis (list): List of syntactic categories to consider.
        
    Returns:
        list: List of top keywords.
    """
    return get_keywords_counts([
        token.lemma_ for doc in texts.apply(spacy_background)
        for token in doc if token.pos_ in sintaxis
    ], top=top)


def get_keywords_yake(texts: list, top: int, lang: str):
    """
    Extracts the top keywords from a text using YAKE (Yet Another Keyword Extractor).
    
    Args:
        texts: Text to extract keywords from.
        top: Number of top keywords to extract.
        lang: Language code.
        
    Returns:
        list: List of top keywords.
    """
    kw_extractor = yake.KeywordExtractor(lan=lang, top=top)
    return kw_extractor.extract_keywords(texts)

@timeinfo
def mine(
        
    df: pd.DataFrame, 
    n: int, 
    algorithm: str, 
    yake_lang: str = None, 
    spacy_sintaxis: list = None, 
    include_non_sexist: bool = False
    
) -> dict:
    """
    Extracts keywords from text data based on the specified algorithm.
    
    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        n (int): Number of top keywords to extract.
        algorithm (str): Keyword extraction algorithm to use. Must be one of: 
            'counts', 'tf-idf', 'spacy', or 'yake'.
        yake_lang (str, optional): Language code for YAKE algorithm.
        spacy_sintaxis (list, optional): List of syntactic categories for 
            Spacy algorithm.
        include_non_sexist (bool, optional): Whether to include non-sexist 
            texts in the output.
        
    Returns:
        dict: Dictionary containing the extracted keywords for each category.
    """
    # Define the keyword extraction function and its arguments based on the specified algorithm
    if algorithm == "counts":
        get_keyw_func = get_keywords_counts
        keyw_func_kwargs = {}
    elif algorithm == "tf-idf":
        get_keyw_func = get_keywords_tfidf
        keyw_func_kwargs = {}
    elif algorithm == "spacy":
        get_keyw_func = get_keywords_spacy
        keyw_func_kwargs = {"spacy_background": nlp, "sintaxis": spacy_sintaxis}
        nlp = spacy.load('es_core_news_sm')
    elif algorithm == "yake":
        get_keyw_func = get_keywords_yake
        keyw_func_kwargs = {"lang": yake_lang}
    else:
        raise ValueError(f"{algorithm} not supported. Must be \
                           'counts', 'tf-idf', 'spacy' or 'yake'.")
    
    # Extract keywords for each category in the DataFrame
    keywords = {}
    for category in df.type.unique():
        if include_non_sexist or category != "non-sexist":
            texts = df.loc[df.type == category, "text"]
            keywords[category] = get_keyw_func(texts=texts, top=n, **keyw_func_kwargs)
    
    return keywords


def prune(keywords, mode="transversal"):
    """
    Prunes the extracted keywords based on the specified mode.
    
    Args:
        keywords (dict): Dictionary of extracted keywords for each category.
        mode (str, optional): Pruning mode. Must be one of: 'transversal' or 'repeated'.
        
    Returns:
        dict: Dictionary containing the pruned keywords for each category 
            and the candidate keywords.
    """
    
    # Select candidates
    if mode == "transversal":
        # Select words that appear in ALL the lists of keywords
        candidates = set(list(keywords.values())[0])
        for sexism_type, keyw in keywords.items():
            candidates = candidates.intersection(set(keyw))
        candidates = list(candidates)
    
    elif mode == "repeated":
        # Select words repeated (appeared > 1) from the lists of keywords
        candidates = []
        aux = defaultdict(int)
        for sexism_type, keyw in keywords.items():
            for word in keyw:
                aux[word] += 1
                if aux[word] > 1:
                    candidates.append(word)
    
    _keywords = {}
    
    # Prune candidates
    for sexism_type, keyw in keywords.items():
        _keywords[sexism_type] = [word for word in keyw if word not in candidates]
    
    # Save candidates
    _keywords["candidates"] = candidates
    
    return _keywords

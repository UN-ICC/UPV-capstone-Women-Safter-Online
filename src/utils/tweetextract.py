# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:28:47 2023

@author: alvar
"""

import os
import time
#import tweepy
import pandas as pd

from misc import timeinfo, prettytime

def extract_tweets(api, query, geocode):
    """
    Extracts tweets based on a query and geocode.
    
    Args:
        api (tweepy.API): tweepy api to make requests
        query (str): Search query for tweets.
        geocode (str): Geographical coordinates for location-based search.
    
    Returns:
        pd.DataFrame: DataFrame containing extracted tweets.
    """
    buffer, tweets = [None], [None, None]
    while len(tweets) > 1:
        last = buffer.pop()
        if last is None:
            tweets = []
            max_id_ = None
        else:
            max_id_ = last[0]
        tweets = api.search_tweets(
            q=query, 
            geocode=geocode, 
            lang='es', 
            result_type='recent', 
            count=100, 
            max_id=max_id_,
            tweet_mode="extended"
        )
        buffer.extend([tweet.id, tweet.created_at, tweet.full_text] 
                      for tweet in tweets)
        time.sleep(5)
    return pd.DataFrame(data=buffer, columns=['id', 'timestamp', 'tweet'])

def make_query(keywords):
    # Helper function to generate a query from a list of keywords
    raise NotImplemented

@timeinfo
def extract_tweets_given_keywords(api, keywords, geocode, verbose=False):
    """
    Extracts tweets given a set of keywords and geolocation information.
    
    Args:
        api (tweepy.API): tweepy api to make requests
        keywords (dict): Dictionary of keyword types and their corresponding lists of keywords.
        geocode (str): Geographical coordinates for location-based search.
        verbose (bool, optional): Whether to print extraction progress information. Defaults to False.
    
    Returns:
        pd.DataFrame: DataFrame containing the extracted tweets.
    """
    corpus = pd.DataFrame()
    print(f'Extracting tweets from {tuple(map(lambda x: float(x), geocode.split(",")[:2]))}\n') 
    for typ, kwords in keywords.items():
        t0 = time.time()
        query = " OR ".join(kwords)
        #query = make_query(kwords)
        data = extract_tweets(api, query, geocode)
        t1 = time.time()
        data['supposed'] = [typ] * len(data)
        corpus = pd.concat([corpus, data], axis=0, ignore_index=True)
        if verbose: print(f"{len(data)} tweets extracted for '{typ}' keywords in {prettytime(t1-t0)}")
    return corpus
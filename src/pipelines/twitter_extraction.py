# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:25:55 2023

@author: alvar
"""

from ..utils.misc import read_json
from ..utils.apisetup import tweepy_setup
from ..utils.tweetextract import extract_tweets_given_keywords

def twitter_extraction(args):
    # read keywords 
    keywords = read_json("{args.keywords_path}//keywords.json")

    # Guadalajara coords (lat,long) plus 100km radius search
    location = '20.659698,-103.349609,100km'
    
    #
    api = tweepy_setup()
    
    # extracting our desired mexican corpus
    mexican_corpus = extract_tweets_given_keywords(api
                                                   keywords, 
                                                   geocode=location, 
                                                   lang="es",
                                                   verbose=True)
    
    mexican_corpus.drop_duplicates(subset="id", inplace=True)
    
    mexican_corpus.to_csv("{args.save_path}/mexican_corpus.tsv", sep="\t", index=False)
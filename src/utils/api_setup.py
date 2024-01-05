# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:55:49 2023

@author: alvar
"""

import os


def get_setup_keys(*keys):
    aux, _suggest = {}, False
    for key in keys:
        value = os.getenv(key)
        if value is None:
            value = input(f"Enter your `{key}`: ")
            _suggest = True
        aux[key] = value
    if _suggest:
        print("\nConsider setting keys in system environ to not " \
              "enter them each time setting up is required.")
    return aux

def openai_setup():
    import openai
    key = get_setup_keys("OPENAI_API_KEY")
    openai.api_key = key["OPENAI_API_KEY"]

    
def tweepy_setup():
    import tweepy
    print("#"*38+"\nAuthenticating with the Twitter API...\n")
    
    keys = get_setup_keys("TWITTER_CONSUMER_KEY", 
                          "TWITTER_CONSUMER_SECRET", 
                          "TWITTER_ACCESS_TOKEN", 
                          "TWITTER_ACCES_TOKEN_SECRET")
    
    # Authenticate with the Twitter API.
    auth = tweepy.OAuthHandler(keys["TWITTER_CONSUMER_KEY"], 
                               keys["TWITTER_CONSUMER_SECRET"])
    auth.set_access_token(keys["TWITTER_ACCESS_TOKEN"], 
                          keys["TWITTER_ACCES_TOKEN_SECRET"])
    
    print("Authentication succesfully complete.\n"+("#"*38))
    
    # Create Twitter API object
    return tweepy.API(auth)

def huggingface_setup(): ...

def wandb_setup(project_name):
    import wandb
    #wandb.login()
    #"8341d82206b6449a99e9247eaa964cddd9785494"
    key = get_setup_keys("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = key["WANDB_API_KEY"]
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = project_name
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"
    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"
    wandb.init()
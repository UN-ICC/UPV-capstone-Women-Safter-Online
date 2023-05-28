# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:25:15 2023

@author: alvar
"""

import pandas as pd
from ..utils.datatools import deepclean
from ..utils.keywtools import mine, prune
from ..utils.misc import save_json


def keyword_mining(args):
    
    # read data
    data = pd.read_csv(args.data_path, 
                       delimiter='\t' if args.data_path.endswith(".tsv") else ";")
    
    # discard english data
    data_es = data.query("language != 'en'")
    
    # deepclean
    data_es.loc[:,"text"] = deepclean(data_es.text, lang="spanish")
    
    # mine keywords
    keywords = mine(data_es, 
                    n=args.nwords, 
                    algorithm=args.algo, 
                    yake_lang="es-mx",
                    spacy_sintaxis=["NOUN", "ADJ"])
    
    # prune keywords
    if args.prune:
        keywords = prune(keywords, mode=args.prune)
    
    # save keywords
    path = f'{args.save_path}/sexism_keywords.json'
    save_json(keywords, path)


if __name__ == "__main__":
    import argparse
    
    parser = argparser.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-path", 
        dest="data_path",
        action="store",
        default=".",
        help="Path to data to mine."
    )
    parser.add_argument(
        "-s",
        "--save-path", 
        dest="save_path",
        action="store",
        default=".",
        help="Path to folder to save the mined keywords."
    )
    parser.add_argument(
        "-w",
        "--words",
        metavar="N",
        type=int,
        dest="nwords",
        action="store",
        default=12,
        help="Number of keywords to mine."
    )
    parser.add_argument(
        "-a",
        "--algorith", 
        dest="algo",
        action="store",
        default="tfidf",
        help="Algorithm used for minning keywords. Can be 'counts', 'tfidf', 'spacy', or 'yake'."
    )
    parser.add_argument(
        "-p",
        "--prune", 
        dest="prune",
        action="store",
        default=None,
        help="How to prune the results. Can be 'transversal' or 'repeated'."
    )
    
    args, _ = parser.parse_known_args()
    keyword_mining(args)
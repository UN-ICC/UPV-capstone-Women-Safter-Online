# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:23:26 2023

@author: alvar
"""


import pandas as pd

from ..utils.datatools import  (
    homogenize_column, 
    integrate_sources, 
    clean_text
)
from ..utils.llmtools import llm_translation
from ..utils.misc import timeinfo, read_json


###############################################################################
# MAIN SETTINGS
###############################################################################

GPT_MODEL = "gpt-3.5-turb0"
MAX_TOKENS = 4096

COMMON_COLUMNS = ["original_id", "text", "sexist", "type", "language", "dataset"]

INTEGRATION_SCHEMA = {
    "callme": {
        "read": {
            "files": [
                "sexism_data.csv"
            ]
        },
        "homogenize": {
            "column_mapping": {
                "id": "original_id",  
            },
            "label_positive": True,
            "constant_columns": [
                ("type", None), 
                ("language", "en")
            ]
        }
    },
    
    "edos": {
        "read": {
            "files": [
                "edos_labelled_aggregated.csv"
            ]
        },
        "homogenize": {
            "column_mapping": {
                "rewire_id": "original_id",
                "label_sexist": "sexist",
                "label_category": "type",
            },
            "label_positive": "sexist",
            "constant_columns": [
                ("language", "en")
            ]
        }
    },
    
    "evalita": {
        "read": {
            "files": [
                "en_training.tsv", 
                "en_testing.tsv"
            ],
            "kwargs": {
                "sep": "\t"
            }
        },
        "homogenize": {
            "column_mapping": {
                "id": "original_id", 
                "misogynous": "sexist",
                "misogyny_category": "type"
            },
            "constant_columns": [
                ("language", "en")
            ]
        }
    },
    
    "exist": {
        "read": {
            "files": [
                "EXIST2021_training.tsv", 
                "EXIST2021_test_labeled.tsv"
            ],
            "kwargs": {
                "sep": "\t"
            }
        },
        "homogenize": {
            "column_mapping": {
                "id": "original_id", 
                "task1": "sexist",
                "task2": "type",
            },
            "label_positive": "sexist",
        }
    },
    
    "ibereval_en": {
        "read": {
            "files": [
                "en_AMI_TrainingSet_NEW.csv"
            ],
            "kwargs": {
                "sep": ";"
            }
        },
        "homogenize": {
            "column_mapping": {
                "id": "original_id", 
                "tweet": "text", 
                "misogynous": "sexist",
                "misogyny_category": "type"
            },
            "constant_columns": [
                ("language", "en")
            ]
        }
    },
    
    "ibereval_es": {
        "read": {
            "files": [
                "es_AMI_TrainingSet_NEW.csv"
            ],
            "kwargs": {
                "sep": ";"
            }
        },
        "homogenize": {
            "column_mapping": {
                "id": "original_id", 
                "tweet": "text", 
                "misogynous": "sexist",
                "misogyny_category": "type"
            },
            "constant_columns": [
                ("language", "es")
            ]
        }
    },
    
    "metwo": {
        "read": {
            "files": [
                "targetResultFile_full2.csv"
            ],
            "kwargs": {
                "sep": ";", 
                "names": [
                    "original_id", 
                    "text", 
                    "sexist"
                ]
            }
        },
        "homogenize": {
            "label_positive": "SEXIST",
            "constant_columns": [
                ("type", None), 
                ("language", "es")
            ]
        }
    }
}

TYPE_COLUMN_MAPPING =  {
        "abuse": [
            'dominance', 
            'stereotyping-dominance', 
            'derailing'
        ],
        'hate': [
            '2. derogation', 
            '3. animosity', 
            '4. prejudiced discussions', 
            'ideological-inequality'
        ],
        'profanities': [
            'stereotype', 
            'misogyny-non-sexual-violence', 
            'discredit'
        ],
        'violent': [
            'misogyny-non-sexual-violence', 
            '1. threats, plans to harm and incitement', 
            'sexual-violence'
        ],
        'sexually-explicit': [
            'sexual_harassment', 
            'objectification'
        ],
        'non-sexist': [
            None, 'none', '0', 'NaN', np.nan
        ]
}    


###############################################################################
# MAIN FUNCTION
###############################################################################

@timeinfo
def data_preparation(args):
    """Performs all the pre-processing stage"""
    
    print(f"PRE-PROCESSING (started at {dt.datetime.now()}).\n")
    print(f"{'-'*30}\n")
    
    # read the shema for the integration phase contained in soruces folder
    #INTEGRATION_SCHEMA = read_json(args.sources_folder+"/integration_schema.json")
    #with open(args.sources_folder+"/integration_schema.json") as js_file:
    #    INTEGRATION_SCHEMA = json.load(js_file)
    
    # INTEGRATION
    print("Integrating data sources:\n")
    homogenize_col_kwargs = {
        "column": "type", 
        "values_mapping": TYPE_COLUMN_MAPPING
    }
    
    integrated_data = integrate_sources(INTEGRATION_SCHEMA,
                                        target_col="sexist",
                                        common_columns=COMMON_COLUMNS,
                                        extras=[(homogenize_column, homogenize_col_kwargs)],
                                        original_data_path=args.sources_folder, 
                                        save=True, 
                                        save_data_path=args.save_folder)
    print(f"\n{'-'*30}\n")
    
    if args.smoke_test:
        integrated_data = integrated_data.sample(10)
    
    # CLEANING
    print("Cleaning raw text...")
    clean_kwargs = {
            "keep_case": False, 
            "keep_accents": True,
            "keep_numbers": True,
            "constraints": [
                (r"\[URL\]|\[USER\]", ""), 
                (r"MENTION\d+", ""), 
                (r"\bRT\b", "")
            ],
            "allowed_punctuation": "'\"!¿?.,"
    }
    
    if args.allow_lmhe:
        clean_kwargs["constraints"] = [
            (r"MENTION\d+", "[USER]"), 
            (r"\bRT\b", "")
        ]
        clean_kwargs["lmhe_tokens"] = {
            "link": "[URL]", 
            "mention": "[USER]", 
            "hashtag": "[HASHTAG]", 
            "emoji": "[EMOJI]"
        }
        clean_kwargs["allowed_punctuation"] += "[]"
    
    @timeinfo
    def apply_clean(data):
        data.text = data.text.apply(clean_text, **clean_kwargs)
        ## because of the transformations it is posible that some texts get completly empty
        data = data.query("text != ''")
        
        ## saving the resulting data set
        clean_path = f"{args.save_folder}/cleaned_data.tsv"
        data.to_csv(clean_path, sep="\t", index=False)
        print(f"\nData correctly cleaned and saved in \033[1m'{clean_path}'\033[0m")
        return data
    
    cleaned_data = apply_clean(integrated_data.copy())
    print(f"\n{'-'*30}\n")
    
    # TRANSLATION
    print("Translating English data:\n")
    cleaned_data_en = cleaned_data.query("language == 'en'")
    
    translated_text = llm_translation(cleaned_data_en.text, 
                                      original="English", 
                                      new="Mexican Spanish", 
                                      model_settings={"name":GPT_MODEL, 
                                                      "max_tokens":MAX_TOKENS}
                                      callback="retrial",
                                      verbose=True)
        
    trans_txt_path = f"{args.save_folder}/translated_text.tsv"
    cleaned_data_en.to_csv(trans_txt_path, sep="\t", index=False)
    print(f"\nText correctly translated and checkpoint saved in \033[1m'{trans_txt_path}'\033[0m")
    
    complete_translated_data = translated_text.dropna(subset=["translated"])
    failed = [ i for i, row in complete_translated_data.iterrows() if "no puedo traducir" in row.translated ]  
    complete_translated_data = complete_translated_data.drop(failed, axis=0)
    print(f"\n{'-'*30}\n")
    
    orig_idxs = complete_translated_data.original_id.apply(int)
            
    if args.save_trans:
        cleaned_data_en["translated"] = np.nan
        cleaned_data_en.loc[orig_idxs, "translated"] = complete_translated_data.translated.to_list()
        
        trans_path = f"{args.save_folder}/translated_data.tsv"
        cleaned_data_en.to_csv(trans_path, sep="\t", index=False)
        print(f"Data correctly translated and saved in \033[1m'{trans_path}'\033[0m\n")

    # FINAL DATA
    print("Unifying results...\n")
    final_data = cleaned_data.copy()
    final_data.loc[orig_idxs, 'text'] = complete_translated_data.translated.str.lower().to_list()
    final_data.loc[orig_idxs, 'language'] = ['es-mx'] * len(complete_translated_data)
    
    final_path = f"{args.save_folder}/gold_data.tsv"
    final_data.to_csv(final_path, sep="\t", index=False)
    print(f"Final data correctly saved in \033[1m'{final_path}'\033[0m\n")
    print(f"{'-'*30}")


if __name__ == "__main__":
    
    import sys
    import argparse
    from utils.api_setup import openai_setup
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sources_folder", 
        default=".", 
        help="path to folder containing the folders for each source."
    )
    parser.add_argument(
        "save_folder", 
        default=".",
        help="path to folder where to save the resulting datasets."
    )
    parser.add_argument(
        "--allow-lmhe-tokens", 
        dest="allow_lmhe", 
        action="store_true", 
        default=False,
        help="maintain links, mentions, hashtags and emojis as \
             special tokens in the text during the cleaning phase."
    )
    parser.add_argument(
        "--save-trans-data", 
        dest="save_trans", 
        action="store_true", 
        default=False,
        help="saves an extra data frame containing the English \
             data plus its translations."
    )
    parser.add_argument(
        "--smoke-test", 
        dest="smoke_test",
        action="store_true", 
        default=False,
        help="makes all the pre-processing with a small sample \
             of the data to check performance."
    )
    args, _ = parser.parse_known_args()
    
    # set up the secret key for openai api
    openai_setup()
    
    # Launch all the pre-processing stage
    print(f"\n{'#'*60}\n")
    data_preparation(args)
    print(f"\n{'#'*60}")
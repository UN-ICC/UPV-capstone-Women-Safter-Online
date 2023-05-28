# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:29:32 2023

@author: alvar
"""

import time
import json

def prettytime(t: int):
    """
    Converts a given time in seconds to hh:mm:ss format.
    
    Args:
        t (int): Time in seconds.
    
    Returns:
        str: Time in hh:mm:ss format.
    """
    hh, mm = divmod(t, 3600)
    mm, ss = divmod(mm, 60)
    hh, mm, ss = map(lambda x: str(int(x)), [hh, mm, ss])
    return f'{hh.zfill(2)}:{mm.zfill(2)}:{ss.zfill(2)}'

def timeinfo(func):
    """
    Decorator that calculates the execution time of a function.
    
    Args:
        func (function): Function to be decorated.
    
    Returns:
        function: Decorated function.
    """
    def timecalculator(*args, **kwargs):
        t0 = time.time()
        ans = func(*args, **kwargs)
        t1 = time.time() 
        print(f'\nAll work done in {prettytime(t1-t0)} ({round(t1-t0, 3)}s) :)')
        return ans
    return timecalculator


def read_json(path):
    """
    Read JSON data from a file.

    PARAMETERS:
        - path: The path to the JSON file.

    RETURNS:
        The content of the JSON file as a dictionary or list.
    """
    with open(path, mode="r+", encoding='utf-8') as js_file:
        content = json.load(js_file)
    return content


def save_json(dict_, path):
    """
    Save a dictionary as JSON data to a file.

    PARAMETERS:
        - dict_: The dictionary to be saved as JSON.
        - path: The path to save the JSON file.

    RETURNS:
        None
    """
    with open(path, mode='w+', encoding='utf-8') as f:
        json.dump(dict_, f, ensure_ascii=False, indent=4)

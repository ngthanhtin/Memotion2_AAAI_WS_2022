import bcolz
import pickle
import numpy as np
import re


#------ CLEANING TEXT-------------
def scrub_words(text):
    """Basic cleaning of texts"""
    # remove html markup
    text=re.sub("(<.*?>)"," ",text)
    text=re.sub("(\\W)"," ",text)
    return text
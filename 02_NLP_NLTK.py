# The 'nlpTokenize' function that will used as helper function to convert your vocan corpus to tokens 
import re
import nltk
import emoji
import numpy as np
from nltk.tokenize import word_tokenize
from utils2 import get_dict

def nlpTokenize(vocabCorpus):
    words = re.sub(r'[,!?;-]+', '.', vocabCorpus)
    words = nltk.word_tokenize(words)
    words = [ ch.lower() for ch in data
             if ch.isalpha()
             or emoji.get_emoji_regexp().search(ch)
             or ch == '.'
           ]
    return words
    

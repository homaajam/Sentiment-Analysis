import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def clean_text(text):
  text =text.lower()
  text =re.sub(r"[^A-Za-z\s]","",text)
  return text

def tokenize(text):
  return word_tokenize(text)

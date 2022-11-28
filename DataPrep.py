import pandas as pd
import numpy as np

acmdata = pd.read_csv('Data/ACM.csv')
print(acmdata)

dblp2data = pd.read_csv('Data/DBLP2.csv', encoding = "ISO-8859-1")
print(dblp2data)

# https://textblob.readthedocs.io/en/dev/
# https://stanfordnlp.github.io/CoreNLP/
# https://github.com/RaRe-Technologies/gensim
# 

# Check if values are missing.
# Have a look at https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b

# Check for "bad" charaters 
# Have a look at https://towardsdatascience.com/data-processing-example-using-python-bfbe6f713d9c

# Check for abbreviations
# Have a look at 

# Check for the "order" eg venue switched
# Have a look at 

# Check vor duplicates
# Have a look at https://thispointer.com/python-3-ways-to-check-if-there-are-duplicates-in-a-list/

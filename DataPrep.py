
import pandas as pd
import numpy as np

acmdata = pd.read_csv('Data/ACM.csv')
#print(acmdata)

dblp2data = pd.read_csv('Data/DBLP2.csv', encoding = "ISO-8859-1")
#print(dblp2data)

### Part 0 Useful links
# https://textblob.readthedocs.io/en/dev/
# https://stanfordnlp.github.io/CoreNLP/
# https://github.com/RaRe-Technologies/gensim
# 



### Part 1 Prepare data
# Merge both files into one
dataframe = pd.concat([acmdata, dblp2data])
print(dataframe)

# Check if values are missing.
# Have a look at https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b
dataframe.dropna(
    axis=0,
    how='any', # all = row completely empty, any = a single cell empty
    subset=None,
    inplace=True
)
print("Missing values")
print(dataframe)

# Replace umlaut charaters 
# Have a look at https://towardsdatascience.com/data-processing-example-using-python-bfbe6f713d9c , 
# https://www.python-lernen.de/string-replace.htm and
# https://www.designerinaction.de/tipps-tricks/web-development/html-umlaute-sonderzeichen/

dictionary = {'&#196;': 'Ä', '&#228;': 'ä', '&#203;': 'Ë', '&#235;': 'ë', '&#207;': 'Ï', '&#239;': 'ï',
              '&#214;': 'Ö', '&#246;': 'ö', '&#220;': 'Ü', '&#252;': 'ü', '&#223;': 'ß', '&#192;': 'À',
              '&#224;': 'à', '&#193;': 'Á', '&#225;': 'á', '&#194;': 'Â', '&#226;': 'â', '&#199;': 'Ç',
              '&#231;': 'ç', '&#200;': 'È', '&#232;': 'è', '&#201;': 'É', '&#234;': 'ê', '&#209;': 'Ñ',
              '&#241;': 'ñ', '&#210;': 'Ò', '&#242;': 'ò', '&#211;': 'Ó', '&#243;': 'ó', '&#212;': 'Ô',
              '&#244;': 'ô', '&#245;': 'õ', '&#195;': 'Ÿ', '&#255;': 'ÿ', '&mdash;': '—'}
dataframe.replace(dictionary, regex=True, inplace=True)
print("replace umlaut")
print(dataframe)

# Check for duplicates
# Have a look at https://thispointer.com/python-3-ways-to-check-if-there-are-duplicates-in-a-list/
dataframe = dataframe.drop_duplicates()
print('drop dublicate')
print(dataframe)


### Part 2 Implement a blocking scheme

# Check for abbreviations and similar venue
# Have a look at


# Check for the "order" eg venue switched
# Have a look at 


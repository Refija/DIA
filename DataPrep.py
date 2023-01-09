import ngram
import pandas as pd
import numpy as np
import ngram as ng

acmdata = pd.read_csv('Data/ACM.csv')
#print(acmdata)

dblp2data = pd.read_csv('Data/DBLP2.csv', encoding = "ISO-8859-1")
#print(dblp2data)

### Part 0 Useful links
# https://textblob.readthedocs.io/en/dev/
# https://stanfordnlp.github.io/CoreNLP/
# https://github.com/RaRe-Technologies/gensim
# https://github.com/natalieahn/namematcher
# https://stackoverflow.com/questions/46186051/python-fuzzy-matching-of-names-with-only-first-initials
# https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams

# Check if values are missing.
# Have a look at https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b
acmdata.dropna(
    axis=0,
    how='any', # all = row completely empty, any = a single cell empty
    subset=None,
    inplace=True
)
#print("Missing values")
#print(acmdata)
dblp2data.dropna(
    axis=0,
    how='any', # all = row completely empty, any = a single cell empty
    subset=None,
    inplace=True
)
#print("Missing values")
#print(acmdata)


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
acmdata.replace(dictionary, regex=True, inplace=True)
#print("replace umlaut")
#print(dataframe)
dblp2data.replace(dictionary, regex=True, inplace=True)
#print("replace umlaut")
#print(dataframe)

# Check for duplicates
# Have a look at https://thispointer.com/python-3-ways-to-check-if-there-are-duplicates-in-a-list/
acmdata = acmdata.drop_duplicates()
#print('drop dublicate')
#print(acmdata)
dblp2data = dblp2data.drop_duplicates()
#print('drop dublicate')
#print(dblp2data)

# Check for abbreviations and similar venue
dictionary = {'Inc\.': 'Incoperated', 'vs\.': 'versus', 'ed\.' : 'edition', 'Jr\.' : 'Junior', 'Corp\.' : 'Corporation',
              'Oct\.' : 'October', 'Univ\.' : 'University', 'Dr\.' : 'Doctor', 'Dept\.' : 'Department',
              'Trans\.' : 'Transaction', 'Syst\.' : 'System', 'Vol\.' : 'Volume', 'J\.' : 'Journal',
              'VLDB' : 'Very Large Data Bases'}
acmdata.replace(dictionary, regex=True, inplace=True)
dblp2data.replace(dictionary, regex=True, inplace=True)

### Part 2 Implement a blocking scheme
# blocking scheme = title, authors, venue, year
# example project https://pypi.org/project/schema-matching/
# according to https://helios2.mi.parisdescartes.fr/~themisp/publications/csur20-blockingfiltering.pdf
# Step 0: same venue strings for comparison
acmdata_venue_strings = acmdata['venue'].unique()
dblp2data_venue_strings = dblp2data['venue'].unique()
#print(acmdata_venue_strings)
#print(dblp2data_venue_strings)

for acm_venue in acmdata_venue_strings:
    for dbl_venue in dblp2data_venue_strings:
        match = ngram.NGram.compare(acm_venue, dbl_venue, N=6) # value found via trial and error
        print("compare ->" + acm_venue + "<- with ->" + dbl_venue + "<-")
        print("score is " + str(match))
        if match > 0.1: # value found via trial and error
            acmdata.loc[acmdata['venue'] == acm_venue, 'venue'] = dbl_venue

# we need to redo it for both datasets, else the venues do not match
acmdata_venue_strings = acmdata['venue'].unique()
dblp2data_venue_strings = dblp2data['venue'].unique()

for dbl_venue in dblp2data_venue_strings:
    for acm_venue in acmdata_venue_strings:
        match = ngram.NGram.compare(acm_venue, dbl_venue, N=6) # value found via trial and error
        print("compare ->" + acm_venue + "<- with ->" + dbl_venue + "<-")
        print("score is " + str(match))
        if match > 0.1: # value found via trial and error
            dblp2data.loc[dblp2data['venue'] == dbl_venue, 'venue'] = acm_venue


# Step 1: candidate selection step: blocks are hashes of venue
## Create dict for dblp2data
global_dblp2data = {}
# Check for swapped values, not the case but for completeness
for item in dblp2data.iterrows():
    if not isinstance(item[1]['title'], str):
        if not isinstance(item[1]['year'], int):
            print("swap")
            tmp = item[1]['title']
            item[1]['title'] = item[1]['year']
            item[1]['year'] = tmp
            continue
    if not isinstance(item[1]['authors'], str):
        if not isinstance(item[1]['year'], int):
            print("swap")
            tmp = item[1]['authors']
            item[1]['authors'] = item[1]['year']
            item[1]['year'] = tmp
            continue
    if not isinstance(item[1]['venue'], str):
        if not isinstance(item[1]['year'], int):
            print("swap")
            tmp = item[1]['venue']
            item[1]['venue'] = item[1]['year']
            item[1]['year'] = tmp
            continue
    # create hashes
    venueHash = hash(item[1]['venue'])
    if venueHash in global_dblp2data:
        value = global_dblp2data[venueHash]
        value.append(item[1])
        global_dblp2data[venueHash] = value
    else:
        global_dblp2data[venueHash] = [item[1]]
 #print(global_dblp2data)

# Check if venue is really the venue value
for item in dblp2data.iterrows():
    if hash(item[1]['title']) in global_dblp2data.keys():
        print("title and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['title']
        item[1]['title'] = tmp
        print("swap done")
    if hash(item[1]['authors']) in global_dblp2data.keys():
        print("authors and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['authors']
        item[1]['authors'] = tmp
        print("swap done")
    if hash(item[1]['year']) in global_dblp2data.keys():
        print("year and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['year']
        item[1]['year'] = tmp
        print("swap done")
#print(global_dblp2data)

# now after possible(not in our case but for completeness) swap, create new dict with correct all correct hashes
global_dblp2data = {}
# Check for swapped values, not the case but for completeness
for item in dblp2data.iterrows():
    # create hashes
    venueHash = hash(item[1]['venue'])
    if venueHash in global_dblp2data:
        value = global_dblp2data[venueHash]
        value.append(item[1])
        global_dblp2data[venueHash] = value
    else:
        global_dblp2data[venueHash] = [item[1]]

## Create dict for acmdata
global_acmdata = {}
# Check for swapped values, not the case but for completeness
for item in acmdata.iterrows():
    if not isinstance(item[1]['title'], str):
        if not isinstance(item[1]['year'], int):
            print("swap")
            tmp = item[1]['title']
            item[1]['title'] = item[1]['year']
            item[1]['year'] = tmp
            continue
    if not isinstance(item[1]['authors'], str):
        if not isinstance(item[1]['year'], int):
            print("swap")
            tmp = item[1]['authors']
            item[1]['authors'] = item[1]['year']
            item[1]['year'] = tmp
            continue
    if not isinstance(item[1]['venue'], str):
        if not isinstance(item[1]['year'], int):
            print("swap")
            tmp = item[1]['venue']
            item[1]['venue'] = item[1]['year']
            item[1]['year'] = tmp
            continue
    # create hashes
    venueHash = hash(item[1]['venue'])
    if venueHash in global_acmdata:
        value = global_acmdata[venueHash]
        value.append(item[1])
        global_acmdata[venueHash] = value
    else:
        global_acmdata[venueHash] = [item[1]]
 #print(global_dblp2data)

# Check if venue is really the venue value
for item in acmdata.iterrows():
    if hash(item[1]['title']) in global_acmdata.keys():
        print("title and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['title']
        item[1]['title'] = tmp
        print("swap done")
    if hash(item[1]['authors']) in global_acmdata.keys():
        print("authors and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['authors']
        item[1]['authors'] = tmp
        print("swap done")
    if hash(item[1]['year']) in global_acmdata.keys():
        print("year and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['year']
        item[1]['year'] = tmp
        print("swap done")
#print(global_acmdata)

# now after possible(not in our case but for completeness) swap, create new dict with correct all correct hashes
global_acmdata = {}
# Check for swapped values, not the case but for completeness
for item in acmdata.iterrows():
    # create hashes
    venueHash = hash(item[1]['venue'])
    if venueHash in global_acmdata:
        value = global_acmdata[venueHash]
        value.append(item[1])
        global_acmdata[venueHash] = value
    else:
        global_acmdata[venueHash] = [item[1]]


# Step 2: candidate matching step, get blocks from both global arrays and compare titles
hashlist_acmdata = global_acmdata.keys()
hashlist_dblp2data = global_dblp2data.keys()
#print(hashlist_acmdata)
#print(hashlist_dblp2data)
print("acmdata")
for hash in hashlist_acmdata:
    print(hash)
    print(global_acmdata[hash][0]['venue'])
    print("")
print("dblp2data")
for hash in hashlist_dblp2data:
    print(hash)
    print(global_dblp2data[hash][0]['venue'])
    print("")

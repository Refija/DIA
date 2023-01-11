import csv
import ngram
import numpy
import pandas as pd

perfectmatch = pd.read_csv('Data/DBLP-ACM_perfectMapping.csv')

acmdata = pd.read_csv('Data/ACM.csv')
# print(acmdata)

dblp2data = pd.read_csv('Data/DBLP2.csv', encoding="ISO-8859-1")
# print(dblp2data)

### Part 0 Useful links
# https://textblob.readthedocs.io/en/dev/
# https://stanfordnlp.github.io/CoreNLP/
# https://github.com/RaRe-Technologies/gensim
# https://github.com/natalieahn/namematcher
# https://stackoverflow.com/questions/46186051/python-fuzzy-matching-of-names-with-only-first-initials
# https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams

# Input parameter
# Match = 1551(69,73%), 2, 0.70 false positive 238(10,70%)
# Match = 1003(44,09%), 2, 0.80 false positive 089(04,00%)
# Match = 1159(52,11%), 2, 0.75 false positive 201(09,03%)
# Match = 1809(81,33%), N=3, Threshold=0.55 with false positive = 254(11,42%) and missing = 161(07,23%) <-
# Match = 1695(76,21%), 3, 0.57 false positive 251(11,28%)
# Match = 1468(66,00%), 3, 0.60 false positive 236(10,61%)
# Match = 1187(53,37%), 3, 0.65 false positive 206(09,26%)
# Match = 1832(82,37%), N=4, Threshold=0.45 with false positive = 271(12,18%) and missing = 121(05,44%) <-
# Match = 1513(68,03%), 4, 0.50 false positive 247(11,10%)
# Match = 1400(62,94%), 4, 0.52 false positive 237(10,64%)

# Best match values
ngram_size = 4
match_percentage = 0.45

## Begin of pipeline
# Check if values are missing.
# Have a look at https://towardsdatascience.com/data-cleaning-with-python-and-pandas-detecting-missing-values-3e9c6ebcf78b
acmdata.dropna(
    axis=0,
    how='any',  # all = row completely empty, any = a single cell empty
    subset=None,
    inplace=True
)
# print("Missing values")
# print(acmdata)
dblp2data.dropna(
    axis=0,
    how='any',  # all = row completely empty, any = a single cell empty
    subset=None,
    inplace=True
)
# print("Missing values")
# print(acmdata)


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
# print("replace umlaut")
# print(dataframe)
dblp2data.replace(dictionary, regex=True, inplace=True)
# print("replace umlaut")
# print(dataframe)

# Check for duplicates
# Have a look at https://thispointer.com/python-3-ways-to-check-if-there-are-duplicates-in-a-list/
acmdata = acmdata.drop_duplicates()
# print('drop dublicate')
# print(acmdata)
dblp2data = dblp2data.drop_duplicates()
# print('drop dublicate')
# print(dblp2data)

# Check for abbreviations and similar venue
dictionary = {'Inc\.': 'Incoperated', 'vs\.': 'versus', 'ed\.': 'edition', 'Jr\.': 'Junior', 'Corp\.': 'Corporation',
              'Oct\.': 'October', 'Univ\.': 'University', 'Dr\.': 'Doctor', 'Dept\.': 'Department',
              'Trans\.': 'Transaction', 'Syst\.': 'System', 'Vol\.': 'Volume', 'J\.': 'Journal',
              'VLDB': 'Very Large Data Bases', 'MOD': ' International Conference on Management of Data'} #DANGER!!!
acmdata.replace(dictionary, regex=True, inplace=True)
dblp2data.replace(dictionary, regex=True, inplace=True)

### Part 2 Implement a blocking scheme
# blocking scheme = title, authors, venue, year
# example project https://pypi.org/project/schema-matching/
# according to https://helios2.mi.parisdescartes.fr/~themisp/publications/csur20-blockingfiltering.pdf
# Step 0: same venue strings for comparison
acmdata_venue_strings = acmdata['venue'].unique()
dblp2data_venue_strings = dblp2data['venue'].unique()
# print(acmdata_venue_strings)
# print(dblp2data_venue_strings)

for acm_venue in acmdata_venue_strings:
    for dbl_venue in dblp2data_venue_strings:
        match = ngram.NGram.compare(acm_venue, dbl_venue, N=6)  # value found via trial and error
        print("compare ->" + acm_venue + "<- with ->" + dbl_venue + "<-")
        print("score is " + str(match))
        if match > 0.1:  # value found via trial and error
            acmdata.loc[acmdata['venue'] == acm_venue, 'venue'] = dbl_venue
            print("new venue name = " + dbl_venue)

# we need to redo it for both datasets, else the venues do not match
acmdata_venue_strings = acmdata['venue'].unique()
dblp2data_venue_strings = dblp2data['venue'].unique()

for dbl_venue in dblp2data_venue_strings:
    for acm_venue in acmdata_venue_strings:
        match = ngram.NGram.compare(acm_venue, dbl_venue, N=6)  # value found via trial and error
        print("compare ->" + acm_venue + "<- with ->" + dbl_venue + "<-")
        print("score is " + str(match))
        if match > 0.1:  # value found via trial and error
            dblp2data.loc[dblp2data['venue'] == dbl_venue, 'venue'] = acm_venue
            print("new venue name = " + acm_venue)

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
    venueHash = str(hash(item[1]['venue']))
    if venueHash in global_dblp2data:
        value = global_dblp2data[venueHash]
        value.append(item[1])
        global_dblp2data[venueHash] = value
    else:
        global_dblp2data[venueHash] = [item[1]]
# print(global_dblp2data)

# Check if venue is really the venue value
for item in dblp2data.iterrows():
    if str(hash(item[1]['title'])) in global_dblp2data.keys():
        print("title and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['title']
        item[1]['title'] = tmp
        print("swap done")
    if str(hash(item[1]['authors'])) in global_dblp2data.keys():
        print("authors and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['authors']
        item[1]['authors'] = tmp
        print("swap done")
    if str(hash(item[1]['year'])) in global_dblp2data.keys():
        print("year and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['year']
        item[1]['year'] = tmp
        print("swap done")
# print(global_dblp2data)

# now after possible(not in our case but for completeness) swap, create new dict with correct all correct hashes
global_dblp2data = {}
# Check for swapped values, not the case but for completeness
for item in dblp2data.iterrows():
    # create hashes
    venueHash = str(hash(item[1]['venue']))
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
    venueHash = str(hash(item[1]['venue']))
    if venueHash in global_acmdata:
        value = global_acmdata[venueHash]
        value.append(item[1])
        global_acmdata[venueHash] = value
    else:
        global_acmdata[venueHash] = [item[1]]
# print(global_dblp2data)

# Check if venue is really the venue value
for item in acmdata.iterrows():
    if str(hash(item[1]['title'])) in global_acmdata.keys():
        print("title and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['title']
        item[1]['title'] = tmp
        print("swap done")
    if str(hash(item[1]['authors'])) in global_acmdata.keys():
        print("authors and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['authors']
        item[1]['authors'] = tmp
        print("swap done")
    if str(hash(item[1]['year'])) in global_acmdata.keys():
        print("year and venue are reversed")
        tmp = item[1]['venue']
        item[1]['venue'] = item[1]['year']
        item[1]['year'] = tmp
        print("swap done")
# print(global_acmdata)

# Check for swapped values, not the case but for completeness
# now after possible(not in our case but for completeness) swap, create new dict with correct all correct hashes
global_acmdata = {}
for item in acmdata.iterrows():
    # create hashes
    venueHash = str(hash(item[1]['venue']))
    if venueHash in global_acmdata:
        value = global_acmdata[venueHash]
        value.append(item[1])
        global_acmdata[venueHash] = value
    else:
        global_acmdata[venueHash] = [item[1]]

# Step 2: candidate matching step, get blocks from both global arrays and compare titles
hashlist_acmdata = global_acmdata.keys()
hashlist_dblp2data = global_dblp2data.keys()
# print(hashlist_acmdata)
# print(hashlist_dblp2data)
print("acmdata hash -> venue")
for ihash in hashlist_acmdata:
    print(ihash)
    print(global_acmdata[ihash][0]['venue'])
print("dblp2data hash -> venue")
for ihash in hashlist_dblp2data:
    print(ihash)
    print(global_dblp2data[ihash][0]['venue'])

# Create new acm dict with venue hash, title hash and row object
new_acmdict = {}
for venueHash in hashlist_acmdata:
    entries = global_acmdata[venueHash]
    for entry in entries:
        title = entry['title']
        #titlehash = str(hash(title))
        titlehash = str(title)
        if venueHash in new_acmdict:
            value = new_acmdict[venueHash]
            value.append([titlehash, entry])
            new_acmdict[venueHash] = value
        else:
            new_acmdict[venueHash] = []
            new_acmdict[venueHash].append([titlehash, entry])
# print(new_acmdict)

# Create new dblp2 dict with venue hash, title hash and row object
new_dbl2dict = {}
for venueHash in hashlist_dblp2data:
    entries = global_dblp2data[venueHash]
    for entry in entries:
        title = entry['title']
        #titlehash = str(hash(title))
        titlehash = str(title)
        if venueHash in new_dbl2dict:
            value = new_dbl2dict[venueHash]
            value.append([titlehash, entry])
            new_dbl2dict[venueHash] = value
        else:
            new_dbl2dict[venueHash] = []
            new_dbl2dict[venueHash].append([titlehash, entry])
# print(new_dbl2dict)

# combine hashlists to iterate over all venues
dblp2data_hashes = []
acmdata_hashes = []
for item in hashlist_dblp2data:
    dblp2data_hashes.append(item)
for item in hashlist_acmdata:
    acmdata_hashes.append(item)

int_hashes = numpy.intersect1d(dblp2data_hashes, acmdata_hashes)
array_hashes = []
for ihash in int_hashes:
    array_hashes.append(ihash)
result = []
count = []

for ahash in array_hashes:
    for acm in new_acmdict[ahash]:
        for dbl in new_dbl2dict[ahash]:
            match = ngram.NGram.compare(acm[0], dbl[0], N=ngram_size) #N=2
            #if match > 0.7:
            #    print("compare ->" + acm[0] + "<- with ->" + dbl[0] + "<-")
            #    print("score is " + str(match))
            #Best macht till now = N=3, 0.62
            if match > match_percentage: #0.6,N=2
                result.append([dbl[1]['id'], acm[1]['id']])
                count.append(match)
            # With hasche of Titles only 886 matches...
            #if acm[0] == dbl[0]:
            #    print("match")
            #    result.append([dbl[1]['id'], acm[1]['id']])

pd.DataFrame(count).to_csv('score.csv', header=["Score"])
result.sort(key=lambda x: x[0])
pd.DataFrame(result).to_csv('result.csv', index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC, header=["idDBLP", "idACM"])

perfect = []

for item in perfectmatch.iterrows():
    perfect.append([item[1]['idDBLP'], item[1]['idACM']])

perfect.sort(key=lambda x: x[0])
pd.DataFrame(perfect).to_csv('perfect.csv', index=False, sep=',', quoting=csv.QUOTE_NONNUMERIC, header=["idDBLP", "idACM"])



# Print results
print("Parameters are:")
print("Size of ngrams = " + str(ngram_size))
print("Percentage for matching = " + str(match_percentage))

pfmatch = pd.read_csv('perfect.csv')
mymatch = pd.read_csv('result.csv')

result = []
for item in mymatch.iterrows():
    result.append([item[1]['idDBLP'], item[1]['idACM']])
result.sort(key=lambda x: x[0])

perfect = []
for item in pfmatch.iterrows():
    perfect.append([item[1]['idDBLP'], item[1]['idACM']])
perfect.sort(key=lambda x: x[0])

# Amount of records
print("Size Perfect = " + str(len(perfect)))
print("Size Result  = " + str(len(result)))

# Find nr. of positive matches
positive_matches = 0
for pf in perfect:
    for rs in result:
        if pf[0] == rs[0] and pf[1] == rs[1]:
            positive_matches = positive_matches + 1
print("Number of positive matches = " + str(positive_matches))
print("Percent of Positive Matches = " + str(100 * float(positive_matches / len(perfect))) + "%.")

false_positive = 0
# Find nr. of false positive matches
for rs in result:
    found = False
    for pf in perfect:
        if rs[0] == pf[0] and rs[1] == pf[1]:
            found = True
    if not found:
        false_positive = false_positive + 1
print("Number of false positive matches = " + str(false_positive))
print("Percentage of false positive matches = " + str(100 * float((false_positive / len(perfect)))) + "%.")

# Find nr. of missing matches
print("Number of missing matches = " + str(len(perfect) - positive_matches - false_positive))
print("Percent of missing matches = " +
      str(100 * float(int(len(perfect) - positive_matches - false_positive) / len(perfect))) + "%.")

Data Integration and Large Scale Analysis

Authors: Günther Moser and Refija Oflaz

Task 2: ML model for Entity matching

Visual studio Code - IDE
Python - programming language
Libraries - csv, ngram, numpy, pandas

"DBLP-ACM" - dataset ( https://dbs.uni-leipzig.de/research/projects/object_matching/benchmark_datasets_for_entity_resolution )

Datasets from "DBLP-ACM":
"DBLP2.csv", 2295 records, attributes ("id","title","authors","venue","year")
"ACM.csv", 2617 records, attributes ("id","title","authors","venue","year")
"DBLP-ACM_perfectMapping.csv", 2225 records, attributes ("idDBLP","idACM")

Project description: The task is to create a pipeline for entity matching by following all the necessary
steps of cleaning, blocking and similarity matching. Once such pipeline is ready it can be used to
train a ML model for predicting new records.

Task1:
Clean data: Check and replace NA's, drop duplicates, check and correct abbrevations, replace umlaut letter.
Blocking schema: Step 1: candidate selection step: blocks are hashes of venue
		 Step 2: candidate matching step, get blocks from both global arrays and compare titles
Output files: "result.csv" prints out all matched pairs (their ID's)
	      "score.csv" prints out all matched pairs similarity value
	      "perfect.csv" prints out perfect matches (pipeline compared with the perfectly matched records)
	      
How to reproduce the resulsts:

Just execute the python file Task1.py
In the lines 35 and 36 you can specify the paraters for the last matching process.
These Parameters are for the ngram matching of the titles.

First we wanted to match the hashvalues of the titles, this would have had a better performace but the titles
are not exactly the same so we needet to match by strings. But as suggested in the paper we used hash values for
the block keys. We used venue as keys. This made the most sense to us.

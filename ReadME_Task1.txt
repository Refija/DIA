Data Integration and Large Scale Analysis

Authors: Günther Moser and Refija Oflaz

Task 2: ML model for Entity matching

Visual studio Code - IDE
Python - programming language

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
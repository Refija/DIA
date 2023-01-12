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

Task 2 is built upon the task 1. Task 1 must be run first in order to generate 2 csv files that are used for the task 2, 
"results.csv" that gives matching records from 2 datasets and "score.scv" with similarity score between the records.

Algorithms: K-Nearest Neighbors and Random Forest classifier for accuracy comparison, how accurately the model will predict match/no match outcome based on score(>0.5)

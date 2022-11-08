import pandas as pd
import numpy as np

acmdata = pd.read_csv('Data/ACM.csv')
print(acmdata)

dblp2data = pd.read_csv('Data/DBLP2.csv', encoding = "ISO-8859-1")
print(dblp2data)

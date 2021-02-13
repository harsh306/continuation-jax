import matplotlib.pyplot as plt
import numpy as np
import json
import jsonlines

path = '/opt/ml/output/pca_ae/version.json'
data = []

with jsonlines.open(path) as reader:
    for obj in reader.iter(type=list, skip_invalid=True, skip_empty=True):
        print(obj[0])


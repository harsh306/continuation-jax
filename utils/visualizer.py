import matplotlib.pyplot as plt
import numpy as np
import json

# df = pd.read_json('/opt/ml/pca_ae/version.json')

results = json.load('/opt/ml/pca_ae/version.json')
print(results)

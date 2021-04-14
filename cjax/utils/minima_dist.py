from cjax.utils.math_trees import *
from jax.experimental.optimizers import l2_norm
import pickle
from jax.flatten_util import ravel_pytree
from jax.tree_util import tree_flatten
import matplotlib.pyplot as plt

sgd_path = "/opt/ml/mlruns/0/cda8d69d4f1d4abaa3ec34f4294242d7/artifacts/output/params.pkl"
sadam_path = "/opt/ml/mlruns/0/499e2c7d99624bbebe5804832cdae845/artifacts/output/params.pkl"
nadam_path = "/opt/ml/mlruns/0/e62430f647274c6e98d6eab930a0150a/artifacts/output/params.pkl"
ngd_path = "/opt/ml/mlruns/0/236a47c6f2524156963a8cc766a06be4/artifacts/output/param.pkl"

pgd_path = "/opt/ml/mlruns/5/5a518d1762cb4b1eafc73e5cfdb11760/artifacts/output/parc.pkl"
ogd_path = "/opt/ml/mlruns/5/ef3650bc42a945ac8a3b5200bd181283/artifacts/output/params.pkl"
paths = [pgd_path, ogd_path]#, sadam_path, nadam_path]
trees = []
for p in paths:
    with open(p, 'rb') as file:
        tmp = pickle.load(file)
        ent, _ = ravel_pytree(tmp)
        trees.append(ent)

# sgd = trees[0]
# ngd = trees[1]
# sadam = trees[2]
# nadam = trees[3]
# name_trees = ['']

# sgd_ngd = l2_norm(pytree_sub((sgd), (nadam)))
# print("sgd_ngd", sgd_ngd)


#sgd_ngd = l2_norm(pytree_sub(sadam, nadam))
pairs = [(p1, p2) for p1 in trees for p2 in trees]
pairs_dist = []
for k1, k2 in pairs:
    pairs_dist.append(l2_norm(pytree_sub(k1, k2)))
print(pairs_dist)

labels = ['pgd','ogd'] #['sgd', 'ngd','sadam', 'nadam' ]
M = len(labels)
corr_matrix = np.asarray(pairs_dist).reshape(M,M)

fig, ax = plt.subplots()
im = ax.imshow(corr_matrix)
im.set_clim(0, 15)
ax.grid(False)
ax.xaxis.set(ticks=range(M), ticklabels=labels)
ax.yaxis.set(ticks=range(M), ticklabels=labels)
#ax.set_ylim(2.5, -0.5)
for i in range(M):
    for j in range(M):
        ax.text(j, i, corr_matrix[i, j], ha='center', va='center',
                color='r')
cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
plt.show()
# #pairs_sub = list(map(lambda x,y: pytree_sub(x,y), pairs))
# pairs_norm = list(map(l2_norm, pairs_sub))
# print(pairs_norm)

#### compute distqances
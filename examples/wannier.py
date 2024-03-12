"""
====================================
Wannier tight binding band structure
====================================

"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np
from fplore import FPLORun
import matplotlib.pyplot as plt
from fplore.util import linspace_ng


tbrun = FPLORun("../example_data/ReO3/wannier/")
ham = tbrun['+hamdata']

###########################################################################
#  Set up the k-path

G = (0, 0, 0)
X = (1, 0, 0)
M = (1, 1, 0)
R = (1, 1, 1)
k = np.vstack([linspace_ng(G, X),
               linspace_ng(X, M),
               linspace_ng(M, R),
               linspace_ng(R, G),
               ])
k = k * np.pi / ham.lattice_vectors[0, 0]

###########################################################################
#  Plot the tight binding band structure.

H = np.zeros((len(k), ham.nwan, ham.nwan), dtype=np.cdouble)
for (i, j), hopping in ham.data.items():
    H[..., i, j] = (np.exp(1j * np.dot(k, hopping['T'].T)) * hopping['H']).sum(axis=-1)
    H[..., j, i] = H[..., i, j].conj()

E = np.linalg.eigvalsh(H)

dk = np.sqrt(((k[1:]-k[:-1])**2).sum(axis=1))
kdist = np.insert(np.cumsum(dk), 0, 0)

fig, ax = plt.subplots()
lines = ax.plot(kdist, E, color='k', lw=0.5)
ax.set(ylabel=r"$E$ (eV)",
       xticks=[kdist[0], kdist[50], kdist[100], kdist[150], kdist[200-1]],
       xticklabels=["$\Gamma$", "X", "M", "R", "$\Gamma$"])

plt.show()

###########################################################################
#  Plot the Hamiltonian.

fig, ax = plt.subplots(constrained_layout=True, ncols=2, sharex=True, sharey=True)
im0 = ax[0].matshow(np.real(H[0])); ax[0].set_title(r"$\operatorname{Re} \mathbf{H}(k=\Gamma)$")
im1 = ax[1].matshow(np.imag(H[0])); ax[1].set_title(r"$\operatorname{Im} \mathbf{H}(k=\Gamma)$")


wannames_pretty = [x.replace('up','↑').replace('dn','↓') for x in ham.wannames]
for _im, _ax in zip([im0, im1], ax):
    fig.colorbar(_im, ax=_ax, location='bottom')
    _ax.set_xticks(np.arange(ham.nwan), labels=wannames_pretty, fontsize=6, rotation=90)
    _ax.set_yticks(np.arange(ham.nwan), labels=wannames_pretty, fontsize=6)

plt.show()

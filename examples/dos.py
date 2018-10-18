from fplore.loader import FPLORun
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

run = FPLORun("examples/graphene_slab")

dos_data = run["+dos.total.l001"].data

plt.plot(dos_data["e"], dos_data["dos"])

plt.axvline(0, alpha=0.2, color='k', lw=2)
plt.xlabel(r"$(E - E_\mathrm{F})/\mathrm{eV}$")
plt.ylabel(r"Density of States")

plt.savefig("examples/dos.png")
plt.show()

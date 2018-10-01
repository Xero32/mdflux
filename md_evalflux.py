### md_evalflux.py
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sf
import smooth

def NoPunctuation(d, places):
    # remove punctuation from decimal values
    # for filename saving/handling
    double = d
    integer = int(double)
    string = str(integer)

    for i in range(places):
        decimal = np.abs(integer - double)
        aux = 10. * decimal
        final = int(aux)
        string = string + ('{}'.format(final))

        double = np.abs(aux - final)
    return string

#TODO pass parameters as input values in shell
temp_S = 300
temp_P = 300
pressure = 1.0
angle = 0.52

# construct directory/file name
_temp_S = str(temp_S)
_temp_P = str(temp_P)
_pr = str(pressure)
_pressure = NoPunctuation(pressure,1)
_angle = NoPunctuation(angle, 2)
parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"

HOME = "/home/becker"
infolder = HOME + "/lammps/flux/111/hlrn/" + parameter_set_str

csv_file = "/home/becker/lammps/flux/parallel_df" + str(temp_S) + str(temp_P) + ".csv"
df = pd.read_csv(csv_file, sep=',')

step = 530000
assert(step % 1000 == 0)

zSlice = df.loc[df['step'] == step, ['z']]
density = zSlice.values

print(zSlice.describe())
# print(density)

zIncoming = df.loc[df['step'] == step, ['z', 'vz']]
zIncoming = zIncoming.loc[zIncoming['vz'] < 0, ['z']]
density_incoming = zIncoming.values

zOutgoing = df.loc[df['step'] == step, ['z', 'vz']]
zOutgoing = zOutgoing.loc[zOutgoing['vz'] >= 0, ['z']]
density_outgoing = zOutgoing.values

nbin = 150
nu = 21
smnu = 3
minval = float(min(density))
maxval = float(max(density))

bins = np.linspace(minval, maxval, nbin)

weight_array1 = np.full(density.shape, 1.0)
weight_array2 = np.full(density_incoming.shape, 1.0)
weight_array3 = np.full(density_outgoing.shape, 1.0)

hist_density = np.histogram(density, bins=bins, weights=weight_array1)
hist_density_incoming = np.histogram(density_incoming, bins=bins, weights=weight_array2)
hist_density_outgoing = np.histogram(density_outgoing, bins=bins, weights=weight_array3)


for k in range(0,len(hist_density[0])):
    hist_density[0][k] = smooth.GaussSmoothing(len(hist_density[0]), k, hist_density[0], dt=1, nu=smnu)
    hist_density_incoming[0][k] = smooth.GaussSmoothing(len(hist_density_incoming[0]), k, hist_density_incoming[0], dt=1, nu=smnu)
    hist_density_outgoing[0][k] = smooth.GaussSmoothing(len(hist_density_outgoing[0]), k, hist_density_outgoing[0], dt=1, nu=smnu)


'''
hist_density[0][:] = sf(hist_density[0], nu, 3, deriv=0)
hist_density_incoming[0][:] = sf(hist_density_incoming[0], nu, 3, deriv=0)
hist_density_outgoing[0][:] = sf(hist_density_outgoing[0], nu, 3, deriv=0)
'''


plt.plot(hist_density[1][:-1], hist_density[0], label='density profile')
plt.plot(hist_density_incoming[1][:-1], hist_density_incoming[0], label='incoming density')
plt.plot(hist_density_outgoing[1][:-1], hist_density_outgoing[0], label='outgoing density')
plt.plot(hist_density_incoming[1][:-1], hist_density_outgoing[0]+hist_density_incoming[0], 'r:', label='control')
plt.legend()
plt.show(block=False)
plt.clf()
plt.cla()

Bound = 17.6 # Angström #TODO
BoundedParticles = df.loc[df['z'] < Bound, ['step', 'vz', 'pe']] #add pot energy

maxsteps = 770000
TimeArr = np.arange(0,maxsteps,1000)
TimeArr = TimeArr * 0.25 / 1000.
# 1000 steps == 0.25*1000 fs == 0.25 ps

particleCount = []
for i in range(0,maxsteps,1000):
    TimeResolved = BoundedParticles.loc[BoundedParticles['step'] == i, ['vz', 'pe']]
    particleCount.append(TimeResolved['vz'].count())

# TODO somehow add a filter, so as not to count directly reflected particles,
# as those are never considered to be bound

particleCount = sf(particleCount, 77, 3, deriv=0)
plt.plot(TimeArr, particleCount, label='No. of bounded particles')
plt.legend()
plt.show()

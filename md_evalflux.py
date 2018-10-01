### md_evalflux.py
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from matplotlib import pyplot as plt


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
jobs = (2118671, 2118672)
trajectories = np.arange(1,10)
csv_file = "/home/becker/lammps/flux/parallel_df" + str(temp_S) + str(temp_P) + ".csv"

df = pd.read_csv(csv_file, sep=',')
density = df.hist('z', bins=1000)
plt.show()
plt.cla()
plt.clf()


bounded = np.cumsum(density[0])
plt.plot(bounded)
plt.cla()
plt.clf()

df.hist('vz', bins=1000)
plt.show()
plt.cla()
plt.clf()

### md_evalflux.py
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sf
import smooth
import argparse
from pathlib import Path
import plot

##### GLOBALS
home = str(Path.home())
savedir = home + "/lammps/flux/plot/"
saveflag = True
Block = False
writeflag = True
parameter_set_str = ""
zSurface = 11.8
area = 1557e-20

def InpParams(parser):
    parser.add_argument("a", help='Angle', default='30', nargs='?')
    parser.add_argument("ts", help='Surface Temperature', default='300', nargs='?')
    parser.add_argument("tp", help='Plasma Temperature', default='300', nargs='?')
    parser.add_argument("p", help='Plasma Pressure', default='1.0', nargs='?')

    args = parser.parse_args()
    if args.a:
        angle = float(args.a)
    if args.ts:
        temp_S = int(args.ts)
    if args.tp:
        temp_P = int(args.tp)
    if args.p:
        pressure = float(args.p)

    return angle, temp_S, temp_P, pressure



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


def ReadFile(name, zSurface):
    # zSurface = 11.8
    print("Read File")
    home = home = str(Path.home())
    infolder = home + "/lammps/flux/111/hlrn/" + name
    fluxfolder = home + "/lammps/flux/"
    csv_file = fluxfolder + name + ".csv"
    df = pd.read_csv(csv_file, sep=',')
    df['z'] = df['z'] - zSurface
    return df


def SliceDataFrame(df, MaxStep=600000):
    step = MaxStep
    assert(step % 1000 == 0)
    zSurface = 11.8
    zSlice = df.loc[df['step'] == step, ['z']]
    density = zSlice.values

    # print(zSlice.describe())
    # print(density)

    zIncoming = df.loc[df['step'] == step, ['z', 'vz']]
    zIncoming = zIncoming.loc[zIncoming['vz'] < 0, ['z']]
    density_incoming = zIncoming.values

    zOutgoing = df.loc[df['step'] == step, ['z', 'vz']]
    zOutgoing = zOutgoing.loc[zOutgoing['vz'] >= 0, ['z']]
    density_outgoing = zOutgoing.values

    return density, density_incoming, density_outgoing

def CreateHistogram_zDensity(density, density_incoming, density_outgoing):
    print("Create Density Profile")
    area = 1557e-20
    nbin = 300
    nbin2 = nbin * 2
    nu = 35
    smnu = 4
    minval = float(min(density))
    maxval = float(max(density)) / 2.

    # use different resolutions in histogram
    bins = np.linspace(minval, maxval, nbin2)
    bins2 = np.linspace(maxval, maxval*2., nbin)
    bins = np.concatenate((bins, bins2[1:]))

    # bins = np.linspace(minval, maxval*3, nbin*2)
    weight_array1 = np.full(density.shape, 1.0)
    weight_array2 = np.full(density_incoming.shape, 1.0)
    weight_array3 = np.full(density_outgoing.shape, 1.0)

    hist_density = np.histogram(density, bins=bins, weights=weight_array1)
    hist_density_incoming = np.histogram(density_incoming, bins=bins, weights=weight_array2)
    hist_density_outgoing = np.histogram(density_outgoing, bins=bins, weights=weight_array3)


    # normalize density histogram
    binwidth1 = float(bins[1]-bins[0])
    binwidth2 = float(bins2[1]-bins2[0])

    ratio = binwidth1 / binwidth2
    if ratio < 1.:
        hist_density[0][nbin2:] *= ratio
        hist_density_incoming[0][nbin2:] *= ratio
        hist_density_outgoing[0][nbin2:] *= ratio
    else:
        hist_density[0][nbin:] /= ratio
        hist_density_incoming[0][nbin2:] /= ratio
        hist_density_outgoing[0][nbin2:] /= ratio


    # normalize density histogram to 1 / nm
    # note: actual normalization regarding surface area (-> 1 nm^2) should happen in plot function
    hist_density[0][:] *= binwidth1 * 1e-10 / 1e-9
    hist_density_incoming[0][:] *= binwidth1 * 1e-10 / 1e-9
    hist_density_outgoing[0][:] *= binwidth1 * 1e-10 / 1e-9

    for k in range(0,len(hist_density[0])):
        hist_density[0][k] = smooth.GaussSmoothing(len(hist_density[0]), k, hist_density[0], dt=1, nu=smnu)
        hist_density_incoming[0][k] = smooth.GaussSmoothing(len(hist_density_incoming[0]), k, hist_density_incoming[0], dt=1, nu=smnu)
        hist_density_outgoing[0][k] = smooth.GaussSmoothing(len(hist_density_outgoing[0]), k, hist_density_outgoing[0], dt=1, nu=smnu)

    return hist_density, hist_density_incoming, hist_density_outgoing


# hist_density[0][:] = sf(hist_density[0], nu, 3, deriv=0)
# hist_density_incoming[0][:] = sf(hist_density_incoming[0], nu, 3, deriv=0)
# hist_density_outgoing[0][:] = sf(hist_density_outgoing[0], nu, 3, deriv=0)

def AnalyticalSolution(pressure, temp_P, Max):
    from flux import CalcFlux
    global area
    ##### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    pi = 3.14159265359
    au = 1.66053904e-27
    atm = 101325
    NA = 6.02214086e23
    ###### VdW Constants for Argon
    a = 1.355       # l^2 bar /mol^2
    a = a * 1e-7 / (NA**2)
    b = 0.03201     # l/mol
    b = b * 1e-6 / NA

    pressureSI = pressure * atm
    te = 20. / 0.00025 # steps
    mass = 40.
    m = mass * au
    incidentTemp = temp_P
    incidentEnergy = incidentTemp * kB # = kB * T_inc
    incidentmeV = incidentEnergy / e0 * 1e3
    velocity = np.sqrt(2.*incidentmeV*e0/(mass*au*1000.))
    density0 = pressureSI / (kB*temp_P)
    particleflux_fs, time_betw = CalcFlux(m, temp_P, velocity, pressureSI, area, density0, a=a, b=b)

    # TODO example, hard coding for angle = 0.52, input energy = 300 K, surface temp = 300 K
    ## te = 20 ps
    lambda1 = -0.0915583532347296
    lambda2 = -1.557621222635234
    c1 = 0.2944678097129901
    c2 = -0.04750945670084901
    R21 = 0.15995954605273582
    R22 = -1.412410333264265
    N1 = 0.3958472799754253
    N2 = 0.03950334604175337
    # new try for te = 12 ps
    lambda1= -0.0922376300970974
    lambda2= -1.614236133833023
    c1= 0.3177651526931371
    c2= -0.05210739213053787
    R21= 0.16054033617623356
    R22= -1.468335080482311
    N1= 0.44487833984149816
    N2= 0.04264878618854504


    ## trial data
    # N1 = 0.45
    # N2 = 0.06
    c11 = c1 * (lambda1 - R22)
    c12 = c2 * (lambda2 - R22)
    c21 = c1 * R21
    c22 = c2 * R21
    phi = particleflux_fs * 6.53 * 1e3 # particles deposited on the surface per t0
    print("phi:", phi)
    print("time between:", time_betw)
    g1 = phi * N1 # number of incoming particles per t0 in trapped state
    g2 = phi * N2 # -""- in quasi-trapped state
    C1 = c22 * g1 - c12 * g2
    C2 = -c21 * g1 + c11 * g2
    C0 = c11 * c22 - c12 * c21

    maximum = Max
    tarr = np.arange(0,maximum,1000) - (te)
    tarr = tarr * 0.00025 / 6.53 # conversion to ps, accounting for units in lambda: [lambda] = 1/t0
    avgHeight = 55. # angstöm
    avgVelo = np.sqrt(2. * kB * temp_P / m)
    tarrShift = tarr - (avgHeight / avgVelo * 1e2 * 0.00025 / 6.53)
    fluxterm1 = (1. - np.exp(lambda1 * tarrShift)) * C1 / (C0 * np.abs(lambda1)) * (c11 + c21)
    fluxterm2 = (1. - np.exp(lambda2 * tarrShift)) * C2 / (C0 * np.abs(lambda2)) * (c12 + c22)

    singletermT1 = c1 * (lambda1 - R22) * np.exp(lambda1 * tarr)
    singletermT2 = c2 * (lambda2 - R22) * np.exp(lambda2 * tarr)
    singletermQ1 = c1 * R21 * np.exp(lambda1 * tarr)
    singletermQ2 = c2 * R21 * np.exp(lambda2 * tarr)
    # singletermT1 = c11 * np.exp(lambda1 * tarr)
    # singletermT2 = c12 * np.exp(lambda2 * tarr)
    # singletermQ1 = c21 * np.exp(lambda1 * tarr)
    # singletermQ2 = c22 * np.exp(lambda2 * tarr)
    Population = (fluxterm1 + fluxterm2)
    Population += (singletermT1 + singletermT2 + singletermQ1 + singletermQ2)
    # Population -= Population[int(12.*0.00025/6.53)]

    Stationary = np.full(len(tarr), 1.) * C1 / (C0 * np.abs(lambda1)) * (c11 + c21) + np.full(len(tarr), 1.) * C2 / (C0 * np.abs(lambda2)) * (c12 + c22)

    t = tarr
    t0 = (avgHeight / avgVelo * 1e2)
    t0 = 12. * 0.00025 / 6.53
    Slope = -lambda1 * np.exp(lambda1 * t0) * C1 / (C0 * np.abs(lambda1)) * (c11 + c21) * (t-t0)
    Slope += -lambda2 * np.exp(lambda2 * t0) * C2 / (C0 * np.abs(lambda2)) * (c12 + c22) * (t-t0)
    Slope += c1 * (lambda1 - R22) * lambda1 * np.exp(lambda1 * t0) * (t-t0)
    Slope += c2 * (lambda2 - R22) * lambda2 * np.exp(lambda2 * t0) * (t-t0)
    Slope += c1 * R21 * np.exp(lambda1 * t0) * lambda1 * (t-t0)
    Slope += c2 * R21 * np.exp(lambda2 * t0) * lambda2 * (t-t0)

    #TODO convert relevant quantities so that they are comparable
    return Population, Stationary, Slope

def SetParamsName(angle, temp_S, temp_P, pressure):
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"
    return parameter_set_str



def main():
    global parameter_set_str
    global Block, saveflag, savedir, writeflag
    global area, zSurface
    parser = argparse.ArgumentParser()
    angle, temp_S, temp_P, pressure = InpParams(parser)
    # temp_S = 190
    # temp_P = 190
    # pressure = 1.0
    # angle = 0.52
    # construct directory/file name
    parameter_set_str = SetParamsName(angle, temp_S, temp_P, pressure)

    df = ReadFile(parameter_set_str, zSurface)
    # change density
    print(df.describe())
    NumOfTraj = df['traj'].max()
    MaxStep = df['step'].max()
    MaxStep2 = MaxStep - 10000
    density, density_incoming, density_outgoing = SliceDataFrame(df, MaxStep=MaxStep2)
    if (temp_S == 300 and angle == 0.52 and temp_P == 300):
        AnSol, Stat, Slope = AnalyticalSolution(pressure, temp_P, MaxStep2)
    else:
        AnSol = []
        Stat = []
        Slope = []
    hist_density, hist_density_incoming, hist_density_outgoing = CreateHistogram_zDensity(
                                                                    density, density_incoming, density_outgoing)
    savepath = savedir + "dens/"
    name = savepath + parameter_set_str + "DensTime"
    plot.PlotDensityOverTime(df, block=Block, NumOfTraj=NumOfTraj, MaxStep=MaxStep2, Stationary=Stat, Slope=Slope,
        xlabel="t / ps", ylabel=r'Particles per nm$^2$', Population=AnSol, saveflag=saveflag, savedir=name, writeflag=writeflag)
    savepath = savedir + "dens/"
    name = savepath + parameter_set_str + "DensHeight"
    plot.PlotDensityHistogram(
        X=[hist_density[1][:-1], hist_density_incoming[1][:-1], hist_density_outgoing[1][:-1]],
        Y=[hist_density[0], hist_density_incoming[0], hist_density_outgoing[0]],
        Label=['Total density','Incoming density','Outgoing density'], writeflag=writeflag,
        block=Block, NumOfTraj=NumOfTraj, xlabel="z / Angström", ylabel=r"Density / nm$^{-3}$", saveflag=saveflag, savedir=name)

    savepath = savedir + "cov/"
    name = parameter_set_str
    plot.PlotCoverage(df, block=Block, MaxStep=MaxStep2, xlabel="x / Angström", ylabel="y / Angström",
        saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)
    savepath = savedir + "ke/"
    name = parameter_set_str + "KinEnTime"
    plot.PlotKineticEnergyOverTime(df, block=Block, xlabel='t / ps', ylabel='E / K',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)
    savepath = savedir + "ke/"
    name = parameter_set_str + "KinEnHeight"
    plot.PlotKineticEnergyOverHeight(df, block=Block, xlabel='z / Angström', ylabel='E / K',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)
    savepath = savedir + "pe/"
    name = parameter_set_str + "PotEnHeight"
    plot.PlotPotentialEnergyOverHeight(df, block=Block, xlabel='z / Angström', ylabel='E / K',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)

if __name__ == "__main__":
    main()

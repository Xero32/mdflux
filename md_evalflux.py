### md_evalflux.py
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sf
import smooth
import argparse

def InpParams(parser):
    parser.add_argument("a", help='Angle', default='30', nargs='?')
    parser.add_argument("ts", help='Surface Temperature', default='300', nargs='?')
    parser.add_argument("tp", help='Plasma Temperature', default='300', nargs='?')
    parser.add_argument("p", help='Plasma Pressure', default='10', nargs='?')

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
    HOME = "/home/becker"
    infolder = HOME + "/lammps/flux/111/hlrn/" + name
    fluxfolder = HOME + "/lammps/flux/"
    csv_file = fluxfolder + name + ".csv"
    df = pd.read_csv(csv_file, sep=',')
    df['z'] = df['z'] - zSurface
    return df


def SliceDataFrame(df):
    step = 600000
    assert(step % 1000 == 0)
    zSurface = 11.8
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

    return density, density_incoming, density_outgoing

def CreateHistogram_zDensity(density, density_incoming, density_outgoing):
    area = 1557e-20
    nbin = 150
    nu = 35
    smnu = 1
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

    return hist_density, hist_density_incoming, hist_density_outgoing


# hist_density[0][:] = sf(hist_density[0], nu, 3, deriv=0)
# hist_density_incoming[0][:] = sf(hist_density_incoming[0], nu, 3, deriv=0)
# hist_density_outgoing[0][:] = sf(hist_density_outgoing[0], nu, 3, deriv=0)

def PlotHistogram(X=[], Y=[], Label=[], block=False):
    for i in range(len(X)):
        plt.plot(X[i], Y[i], label=Label[i])
    plt.legend()
    plt.show(block=block)
    plt.clf()
    plt.cla()

def PlotDensityOverTime(df, block=True, NumOfTraj=40):
    ###### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27
    mass = 40
    m = mass * au
    area = 1557e-20
    Bound = 17.6 # Angström #TODO

    BoundedParticles = df.loc[df['z'] < Bound, ['step', 'vz', 'pe']]
    # BoundedParticles['zKE'] = 0.5 * m * ((BoundedParticles['vz'] * 100.) ** 2) / e0
    # TrappedParticles = BoundedParticles.loc[BoundedParticles['zKE'] + BoundedParticles['pe'] > 0, ['step', 'vz', 'pe']]

    maxsteps = 770000
    TimeArr = np.arange(0,maxsteps,1000)
    TimeArr = TimeArr * 0.25 / 1000.
    # 1000 steps == 0.25*1000 fs == 0.25 ps

    particleCount = []
    for i in range(0,maxsteps,1000):
        TimeResolved = BoundedParticles.loc[BoundedParticles['step'] == i, ['vz', 'pe']]
        # TimeResolved = TrappedParticles.loc[TrappedParticles['step'] == i, ['vz', 'pe']]
        particleCount.append(TimeResolved['vz'].count())

    # TODO somehow add a filter, so as not to count directly reflected particles,
    # as those are never considered to be bound

    particleCount = sf(particleCount, 77, 3, deriv=0)

    particleCount = particleCount / NumOfTraj
    plt.plot(TimeArr, particleCount, label=str(r'No. of particles per area below %.2f Angström' % Bound))
    plt.xlabel('time / ps')
    plt.ylabel('Number of bounded particles')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(str('population%s.pdf' % parameter_set_str))
    plt.show(block=block)
    plt.clf()
    plt.cla()

def PlotKineticEnergy(df, block=True):
    ###### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27

    mass = 40
    m = mass * au
    VelocityDF = df.loc[:, ['vx', 'vy', 'vz']]
    VelocityDF['ke'] = 0.5 * m * ( (VelocityDF['vx'] * 100.) ** 2
                                    + (VelocityDF['vy'] * 100.) ** 2
                                    + (VelocityDF['vz'] * 100.) ** 2)

    KinEnergy = VelocityDF['ke'].values / e0 * 1000 # gives kinetic energy in meV
    TimeArr = df['step'].values
    HeightArr = df['z'].values
    plt.scatter(TimeArr, KinEnergy, label='Kin Energy over Time')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.clf()
    plt.cla()
    plt.scatter(HeightArr, KinEnergy, label='Kin Energy over Height')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.clf()
    plt.cla()

    return KinEnergy

def PlotCoverage(df, block=True):
    step = 600000
    traj = 1
    assert(step % 1000 == 0)
    Boundaries = [3.0, 6.5, 10.0, 13.5]
    Locations = df.loc[(df['step'] ==  step) & (df['traj'] == traj), ['x','y','z']]
    FirstLayer = Locations.loc[(Boundaries[0] < Locations['z']) & (Locations['z'] <= Boundaries[1]), ['x','y']]
    SecondLayer = Locations.loc[(Boundaries[1] < Locations['z']) & (Locations['z'] <= Boundaries[2]), ['x','y']]
    ThirdLayer = Locations.loc[(Boundaries[2] < Locations['z']) & (Locations['z'] <= Boundaries[3]), ['x','y']]
    print(FirstLayer['x'])
    plt.scatter(FirstLayer['x'], FirstLayer['y'], label='First Layer Adatoms Coordinates')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.clf()
    plt.cla()
    plt.scatter(SecondLayer['x'], SecondLayer['y'], label='First Layer Adatoms Coordinates')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.clf()
    plt.cla()
    plt.scatter(ThirdLayer['x'], ThirdLayer['y'], label='First Layer Adatoms Coordinates')
    plt.legend()
    plt.tight_layout()
    plt.show(block=block)
    plt.clf()
    plt.cla()

def main():
    zSurface = 11.8
    area = 1557e-20
    parser = argparse.ArgumentParser()
    angle, temp_S, temp_P, pressure = InpParams(parser)
    # temp_S = 190
    # temp_P = 190
    # pressure = 1.0
    # angle = 0.52
    # construct directory/file name
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"

    df = ReadFile(parameter_set_str, zSurface)
    print(df.describe())
    NumOfTraj = df['traj'].max()
    print(NumOfTraj)
    density, density_incoming, density_outgoing = SliceDataFrame(df)
    hist_density, hist_density_incoming, hist_density_outgoing = CreateHistogram_zDensity(
                                                                    density, density_incoming, density_outgoing)
    PlotHistogram(
        X=[hist_density[1][:-1], hist_density_incoming[1][:-1], hist_density_outgoing[1][:-1]],
        Y=[hist_density[0], hist_density_incoming[0], hist_density_outgoing[0]],
        Label=['Total density','Incoming density','Outgoing density'],
        block=True)
    PlotDensityOverTime(df, block=True, NumOfTraj=NumOfTraj)
    PlotKineticEnergy(df, block=False)
    PlotCoverage(df, block=True)





if __name__ == "__main__":
    main()

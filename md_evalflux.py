### md_evalflux.py
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter as sf
import smooth
import argparse
import math
import mdsolutions as ms
from pathlib import Path
import plot

##### GLOBALS
home = str(Path.home())
savedir = home + "/lammps/flux/plot/"
saveflag = True
Block = False
writeflag = True
hidesolflag = False
parameter_set_str = ""
zSurface = 11.8
area = 1557e-20
max_cov = 121.75

dt = 0.00025
t_0 = 6.53
stw = 1000 # StepsToWrite

def InpParams(parser):
    parser.add_argument("a", help='Angle', default='30', nargs='?')
    parser.add_argument("ts", help='Surface Temperature', default='300', nargs='?')
    parser.add_argument("tp", help='Plasma Temperature', default='300', nargs='?')
    parser.add_argument("p", help='Plasma Pressure', default='1.0', nargs='?')
    parser.add_argument("--beam", help="Choose whether Plasma Particles are to be parallel to each other", action='store_true')
    parser.add_argument("--save", help="Save figures", action='store_true')
    parser.add_argument("--show", help="Show figures", action='store_true')
    parser.add_argument("--write", help="Write figures", action='store_true')
    parser.add_argument("--hsol", help="Hide analytical solution to the Rate Equation Model", action='store_true')
    parser.add_argument("--pt", help="Analyze Platinum data for comparison purposes", action='store_true')

    args = parser.parse_args()
    if args.a:
        angle = float(args.a)
    if args.ts:
        temp_S = int(args.ts)
    if args.tp:
        temp_P = int(args.tp)
    if args.p:
        pressure = float(args.p)
    if args.beam:
        beamFlag = True
    else:
        beamFlag = False
    if args.save:
        saveFlag = True
    else:
        saveFlag = False
    if args.show:
        showFlag = True
    else:
        showFlag = False
    if args.write:
        writeFlag = True
    else:
        writeFlag = False
    if args.hsol:
        hsolFlag = True
    else:
        hsolFlag = False
    if args.pt:
        ptflag = True
    else:
        ptflag = False

    return angle, temp_S, temp_P, pressure, beamFlag, saveFlag, showFlag, writeFlag, hsolFlag, ptflag






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


def SliceDataFrame(df, zSurface, MaxStep=600000, LoopSize=10):

    density = np.array([])
    density_incoming = np.array([])
    density_outgoing = np.array([])
    for step in range(MaxStep-LoopSize*1000, MaxStep+1000, 1000):
        assert(step % 1000 == 0)
        zSlice = df.loc[df['step'] == step, ['z']]
        density = np.append(density, zSlice.values)

        zIncoming = df.loc[df['step'] == step, ['z', 'vz']]
        zIncoming = zIncoming.loc[zIncoming['vz'] < 0, ['z']]
        density_incoming = np.append(density_incoming, zIncoming.values)

        zOutgoing = df.loc[df['step'] == step, ['z', 'vz']]
        zOutgoing = zOutgoing.loc[zOutgoing['vz'] >= 0, ['z']]
        density_outgoing = np.append(density_outgoing, zOutgoing.values)

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
def InterpolateSlope(A, sB):
    #TODO Work on that!
    sA = len(A)-1
    assert(sA < 250)
    assert(sB < 200)
    C = [0.0 for i in range(sA*sB)]

    for i in range(sA):
        try:
            diff = (A[i+1] - A[i]) / sB
        except:
            diff = 0.0
        for j in range(sB):
            C[j + i*sB] = (diff + C[(j-1) + (i)*sB])

    B = [0.0 for i in range(sB)]
    for i in range(1,sB):
        slope = (C[i*sB] - C[(i-1)*sB]) * (sA / sB)
        B[i] = slope + B[i-1]

    del C

    return B

def FindBoundaries(A):
    earliest = 0
    latest = 0
    for i, value in enumerate(A):
        if value > 0:
            earliest = i
            break

    for i in range(earliest, len(A)):
        if A[i] == A[i-1]:
            latest = i
            break
    # print("earliest:",earliest)
    # print("latest:", latest)
    return earliest, latest

def OpenSingleTrajData(fname):
    f = open(fname, 'r')
    data = []
    for i, line in enumerate(f):
        if '#' in line:
            continue
        data.append(float(line.split('=')[1]))

    lambda1 = data[0]
    lambda2 = data[1]
    c1 = data[2]
    c2 = data[3]
    R21 = data[6]
    R22 = data[7]
    N1 = data[8]
    N2 = data[9]
    te = data[10] / dt
    f.close()
    print("Successfully read data from %s" % fname)
    del data

    return lambda1, lambda2, c1, c2, R21, R22, N1, N2, te

def InitPopulations(f1, f2):
    T = []
    Q = []
    dataTime = []
    for i, line in enumerate(f1):
        if '#' in line:
            continue
        t, n = line.split(',')
        T.append(float(n))
        dataTime.append(float(t))
    f1.close()
    for i, line in enumerate(f2):
        if '#' in line:
            continue
        t, n = line.split(',')
        Q.append(float(n))
    f2.close()
    return T, Q, dataTime

def AnalyticalSolution(angle, temp_S, energy, pressure, temp_P, Max, cov):
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
    te = 12. / dt # steps
    mass = 40.
    m = mass * au
    incidentTemp = temp_P
    incidentEnergy = incidentTemp * kB # = kB * T_inc
    incidentmeV = incidentEnergy / e0 * 1e3
    velocity = np.sqrt(2.*incidentmeV*e0/(mass*au*1000.))
    density0 = pressureSI / (kB*temp_P)
    particleflux_fs, time_betw = CalcFlux(m, temp_P, velocity, pressureSI, area, density0, a=a, b=b)
    home = str(Path.home())

    angle_deg = int(angle * 180/pi)
    Angle = {0.52:'30', 0.00:'0', 0.80:'45', 1.05:'60'}
    if (energy == 16.03) or (energy == 16.027):
        energy = 16.027
        paramname = str("Single_a%.2ft%de%.3f.dat" % (angle, temp_S, energy))
        fname = home + "/lammps/" + "111" + "/" + paramname
        f1name = home + "/lammps/" + "111" + "/" + str("a%st%de%.3fHLRNInitTrapped.csv" % (Angle[angle], temp_S, energy))
        f2name = home + "/lammps/" + "111" + "/" + str("a%st%de%.3fHLRNInitQuasiTr.csv" % (Angle[angle], temp_S, energy))
    else:
        paramname = str("Single_a%.2ft%de%.2f.dat" % (angle, temp_S, energy))
        fname = home + "/lammps/" + "111" + "/" + paramname
        f1name = home + "/lammps/" + "111" + "/" + str("a%st%de%.2fHLRNInitTrapped.csv" % (Angle[angle], temp_S, energy))
        f2name = home + "/lammps/" + "111" + "/" + str("a%st%de%.2fHLRNInitQuasiTr.csv" % (Angle[angle], temp_S, energy))
    try:
        f1 = open(f1name, 'r')
        f2 = open(f2name, 'r')
        T, Q, dataTime = InitPopulations(f1, f2)
    except:
        print("Could not open %s and %s" % (f1name,f2name))
        return [],[],[],[]
    try:
        lambda1, lambda2, c1, c2, R21, R22, N1, N2, te = OpenSingleTrajData(fname)
    except:
        print("Could not open %s" % fname)
        return [],[],[],[]
    theta = cov / max_cov
    print(theta)
    s = 1.0 # fit param
    N1 *= (1. - theta) ** s
    c11 = c1 * (lambda1 - R22)
    c12 = c2 * (lambda2 - R22)
    c21 = c1 * R21
    c22 = c2 * R21
    phi = particleflux_fs * t_0 * 1e3 # particles deposited on the surface per t0
    print("phi:", phi, "particles per t_0")
    print("time between:", time_betw)
    g1 = phi * N1 # number of incoming particles per t0 in trapped state
    g2 = phi * N2 # -""- in quasi-trapped state
    C1 = (c22 * g1) - (c12 * g2)
    C2 = (-c21 * g1) + (c11 * g2)
    C0 = (c11 * c22) - (c12 * c21)

    maximum = Max
    # maximum = 2577000  <-> 644 ps <-> delta = 644/2577 = 0.25
    tarr = np.arange(0,maximum,stw) * dt / t_0 # conversion to t_0, accounting for units in lambda: [lambda] = 1/t0

    avgHeight = 55. # angström
    avgVelo = np.sqrt(2. * kB * temp_P / m)
    avgTime = avgHeight / avgVelo * 1e1
    Shift = avgTime / t_0
    # Shift = 0.0

    tarr -= (te * dt / t_0)
    tarrShift = tarr - Shift
    fluxterm1 = (1. - np.exp(lambda1 * tarrShift)) * C1 / (C0 * np.abs(lambda1)) * (c11 + c21)
    fluxterm2 = (1. - np.exp(lambda2 * tarrShift)) * C2 / (C0 * np.abs(lambda2)) * (c12 + c22)

    singletermT1 = c1 * (lambda1 - R22) * np.exp(lambda1 * tarr)
    singletermT2 = c2 * (lambda2 - R22) * np.exp(lambda2 * tarr)
    singletermQ1 = c1 * R21 * np.exp(lambda1 * tarr)
    singletermQ2 = c2 * R21 * np.exp(lambda2 * tarr)

    Population = (fluxterm1 + fluxterm2)
    NonEqTerm = [0.0 for i in range(len(Population))]
    single_delta_t = (dataTime[1] - dataTime[0])
    single_delta_t0 = single_delta_t / 6.53
    flux_delta_t = dt * stw

    for i in range(1,len(T)):
        NonEqTerm[i] += (T[i] + Q[i]) * phi * single_delta_t0
        NonEqTerm[i] += NonEqTerm[i-1]

    for i in range(len(T),len(Population)):
        NonEqTerm[i] = NonEqTerm[i-1]

    Stationary = np.full(len(tarr), 1.) * C1 / (C0 * np.abs(lambda1)) * (c11 + c21)
    Stationary += np.full(len(tarr), 1.) * C2 / (C0 * np.abs(lambda2)) * (c12 + c22)

    # convert the slope of the NonEqTerm array, since we have different time resolution in the single particle and flux simulations
    earliest, latest = FindBoundaries(NonEqTerm)

    # how many steps are equal to tE in each simulation
    Single_eq_steps = latest-earliest
    ratio = single_delta_t / flux_delta_t
    Flux_eq_steps = np.ceil(Single_eq_steps * ratio)

    NewNonEqTerm = InterpolateSlope(NonEqTerm[earliest:latest+1], int(Flux_eq_steps))
    del NonEqTerm

    b = len(NewNonEqTerm)
    Population += (singletermT1 + singletermT2 + singletermQ1 + singletermQ2)
    intercept = Population[0]
    intercept = 0.0 # probably deprecated
    Population -= intercept

    ctr = 0
    NumData = NewNonEqTerm

    for i in range(0, len(Population)):
        Population[i] += NewNonEqTerm[-1]


    ConstShift = [NewNonEqTerm[-1] for i in range(len(Stationary))]
    Stationary += ConstShift
    Stationary -= intercept


    t = tarr
    t0 = 0.0
    t1 = -Shift
    Slope = -lambda1 * np.exp(lambda1 * t1) * C1 / (C0 * np.abs(lambda1)) * (c11 + c21) * (t-t0)
    Slope += -lambda2 * np.exp(lambda2 * t1) * C2 / (C0 * np.abs(lambda2)) * (c12 + c22) * (t-t0)
    Slope += c1 * (lambda1 - R22) * lambda1 * np.exp(lambda1 * t0) * (t-t0)
    Slope += c2 * (lambda2 - R22) * lambda2 * np.exp(lambda2 * t0) * (t-t0)
    Slope += c1 * R21 * np.exp(lambda1 * t0) * lambda1 * (t-t0)
    Slope += c2 * R21 * np.exp(lambda2 * t0) * lambda2 * (t-t0)
    shift = int(te / 1000.)
    Slope += Population[shift]

    #TODO convert relevant quantities so that they are comparable
    return Population, Stationary, Slope, NumData

def SetParamsName(angle, temp_S, temp_P, pressure):
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"
    return parameter_set_str



def WriteTrajData(name, angle, temp_S, temp_P, pressure, coverage, pe):
    f = open(name, 'a')
    f.write("%f %f %f %f %f %f\n" %(angle, temp_S, temp_P, pressure, coverage, pe))
    f.close()


def GetPotentialOfTheta(df, lbound, rbound):

    auxdf = df.loc[(df['z'] >= lbound) & (df['z'] < rbound), ['pe']]
    num = auxdf['pe'].count()
    pe = auxdf['pe'].mean() # get PotE per particle
    return num, pe

'''
def writepairdist(simdata::SimData, params::SimParameters):
# TODO still in julia code
    filename = params.directory * "/pairdist.dat"
    file = open(filename, "a")

    rs = Vector{Float64}(length(simdata.pairdist))
    grs = Vector{Float64}(length(simdata.pairdist))

    for i in 1:length(simdata.pairdist)

        rlow    = params.dr*(i-1)
        rmiddle = rlow+0.5params.dr
        rhigh   = rlow+params.dr

        dv = ((params.dim+1)/3) * pi * (rhigh^params.dim - rlow^params.dim)
        vol = ((params.dim+1)/3) * pi

        rs[i]  = rmiddle
        grs[i] = 2vol * simdata.pairdist[i] / ( dv * (params.steps/params.stepstoupdate) * params.npart )#factor 2 due to no double counting

        println( file, rs[i], "\t", grs[i] )

    end

    close(file)
end
'''

def pairdist(dr, l, steps, stw, npart):
    pi = 3.14159265359
    rs = []
    grs = []
    stepstoupdate = stw
    length = int(l / dr)
    for i in range(length):
        rlo = dr * i
        rmi = rlo + 0.5 * dr
        rhi = rlo + dr
        dim = 3
        dv = 4./3. * pi * (rhi**dim - rlo**dim)
        vol = 4./3. * pi

        rs.append(rmi)
        grs.append(2. * vol * pairdist / (dv * (steps/stepstoupdate) * npart))

    return rs, grs

def CalcGofR(df, MaxStep, stw, dr, l):
    rs = [0.0 for i in range(MaxStep)]
    grs = [0.0 for i in range(MaxStep)]
    i = 0
    for step in range(0,MaxStep,stw):
        aux = df.loc[(df['step'] == step), ['z']]
        npart = aux.count()
        rs[i], grs[i] = pairdist(dr, l, MaxStep, stw, npart)

    return rs, grs

def GofR(df, MaxStep, stw, dr, length, grid, xmax, ymax):
    print("calcuate pair distribution function")
    time = np.arange(0,MaxStep,stw*100)
    g = np.full((len(time), length), 0.0)

    for t, tm in enumerate(time):
        print(tm)
        X = df.loc[(df['step'] == tm), ['x']].values
        Y = df.loc[(df['step'] == tm), ['y']].values
        Z = df.loc[(df['step'] == tm), ['z']].values
        iX = df.loc[(df['step'] == tm), ['ix']].values
        iY = df.loc[(df['step'] == tm), ['iy']].values
        n = len(X)
        for k,r in enumerate(grid):
            for i in range(n):
                x = float(X[i] + iX[i] * xmax)
                y = float(Y[i] + iY[i] * ymax)
                z = float(Z[i])
                for j in range(0,i):
                    dx = x - float(X[j] + iX[j] * xmax)
                    dy = y - float(Y[j] + iY[j] * ymax)
                    dz = z - float(Z[j])
                    r_norm = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if r_norm <= r+dr and r_norm > r:
                        g[t,k] += 1.0
                for j in range(i+1,n):
                    dx = x - float(X[j] + iX[j] * xmax)
                    dy = y - float(Y[j] + iY[j] * ymax)
                    dz = z - float(Z[j])
                    r_norm = math.sqrt(dx*dx + dy*dy + dz*dz)
                    if r_norm <= r+dr and r_norm > r:
                        g[t,k] += 1.0
        g[t] *= 1. / (n * (n-1))
    return g


def main():
    global parameter_set_str
    global Block, saveflag, savedir, writeflag, hidesolflag
    global area, zSurface
    lbound = 0.0
    rbound = 5.0
    parser = argparse.ArgumentParser()
    angle, temp_S, temp_P, pressure, beamFlag, saveflag, Block, writeflag, hidesolflag, ptflag = InpParams(parser)
    if ptflag:
        lattice_const = 3.96
    else:
        lattice_const = 4.08
    substrateZmax = lattice_const * 3.
    substrateYmax = lattice_const * 9. * np.sqrt(3) / np.sqrt(2)
    substrateXmax = lattice_const * 12. / np.sqrt(2)

    area = substrateXmax * substrateYmax * 1e-20

    # construct directory/file name
    parameter_set_str = SetParamsName(angle, temp_S, temp_P, pressure)
    if (beamFlag == True):
        parameter_set_str += 'Beam'
    df = ReadFile(parameter_set_str, zSurface)
    # change density
    print(df.describe())
    NumOfTraj = df['traj'].max()
    MaxStep = df['step'].max()
    MaxStep2 = MaxStep - 10000
    LoopSize = int(temp_S / 2)
    density, density_incoming, density_outgoing = SliceDataFrame(df, zSurface, MaxStep=MaxStep2, LoopSize=LoopSize)
    if temp_P ==  300:
        energy = 25.90
    if temp_P == 190:
        energy = 16.027

    savepath = savedir + "cov/"
    name = parameter_set_str

    dr = 0.1
    max_r = 12.0
    grid = np.arange(0.0, max_r, dr)
    g_length = int(max_r / dr)
    gofr = GofR(df, 1000000, stw, dr, g_length, grid, substrateXmax, substrateYmax)
    a = 8
    plt.plot(grid, gofr[a])
    plt.show()
    plt.clf()
    plt.cla()

    coverage = plot.PlotCoverage(df, angle, temp_S, temp_P, pressure, block=False, MaxStep=MaxStep2, xlabel="x / Angström", ylabel="y / Angström",
        saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)

    if not hidesolflag:
        AnSol, Stat, Slope, NumData = AnalyticalSolution(angle, temp_S, energy, pressure, temp_P, MaxStep2, coverage)
    else:
        AnSol, Stat, Slope, NumData = [], [], [], []
    if temp_S <= 160:
        Stat = []

    hist_density, hist_density_incoming, hist_density_outgoing = CreateHistogram_zDensity(
                                                                    density, density_incoming, density_outgoing)
    savepath = savedir + "dens/"
    name = savepath + parameter_set_str + "DensTime"
    plot.PlotDensityOverTime(df, block=Block, NumOfTraj=NumOfTraj, MaxStep=MaxStep2, Stationary=Stat, Slope=NumData,
        xlabel="t / ps", ylabel=r'Particles per nm$^2$', Population=AnSol, saveflag=saveflag, savedir=name, writeflag=writeflag)
    savepath = savedir + "dens/"
    name = savepath + parameter_set_str + "DensHeight"
    plot.PlotDensityHistogram(
        X=[hist_density[1][:-1], hist_density_incoming[1][:-1], hist_density_outgoing[1][:-1]],
        Y=[hist_density[0], hist_density_incoming[0], hist_density_outgoing[0]],
        Label=['Total density','Incoming density','Outgoing density'], writeflag=writeflag,
        block=Block, NumOfTraj=NumOfTraj, xlabel="z / Angström", ylabel=r"Density / nm$^{-3}$", saveflag=saveflag, savedir=name)

    savepath = savedir + "ke/"
    name = parameter_set_str + "KinEnTime"
    if temp_S == 300:
        effTemp_S = 200
    elif temp_S == 190:
        effTemp_S = 160
    elif temp_S == 80:
        effTemp_S = 80

    effTemp_P = temp_P

    plot.PlotKineticEnergyOverTime(df, block=Block, xlabel='t / ps', ylabel='E / K',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag,
        temp_S=effTemp_S, temp_P=effTemp_P)
    savepath = savedir + "ke/"
    name = parameter_set_str + "KinEnHeight"
    plot.PlotKineticEnergyOverHeight(df, block=Block, xlabel='z / Angström', ylabel='E / K',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)
    savepath = savedir + "pe/"
    name = parameter_set_str + "PotEnHeight"
    PE = plot.PlotPotentialEnergyOverHeight(df, block=Block, xlabel='z / Angström', ylabel='E / K',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)

    num, potE = GetPotentialOfTheta(df, lbound, rbound)
    WriteTrajData('param.dat', angle, temp_S, temp_P, pressure, coverage, potE)


if __name__ == "__main__":
    main()

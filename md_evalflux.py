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
from lmfit import Model
import coverage_sticking as cs
import sys

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

dt = 0.00025
t_0 = 6.53
stw = 1000 # StepsToWrite

def InpParams(parser):
    parser.add_argument("a", help='Angle', default='0.52', nargs='?')
    parser.add_argument("ts", help='Surface Temperature', default='300', nargs='?')
    parser.add_argument("tp", help='Plasma Temperature', default='300', nargs='?')
    parser.add_argument("p", help='Plasma Pressure', default='1.0', nargs='?')
    parser.add_argument("--beam", help="Choose whether Plasma Particles are to be parallel to each other", action='store_true')
    parser.add_argument("--save", help="Save figures", action='store_true')
    parser.add_argument("--show", help="Show figures", action='store_true')
    parser.add_argument("--write", help="Write figures", action='store_true')
    parser.add_argument("--hsol", help="Hide analytical solution to the Rate Equation Model", action='store_true')
    parser.add_argument("--pt", help="Analyze Platinum data for comparison purposes", action='store_true')
    parser.add_argument("--gofr", help="Calculate Radial Pair Distribution Function", action='store_true')
    parser.add_argument("--nia", help="Save data for the non-interacting case", action='store_true')

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
    if args.gofr:
        gofrflag = True
    else:
        gofrflag = False
    if args.nia:
        niaFlag = True
    else:
        niaFlag = False

    return angle, temp_S, temp_P, pressure, beamFlag, saveFlag, showFlag, writeFlag, hsolFlag, ptflag, gofrflag, niaFlag






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
    home = str(Path.home())
    infolder = home + "/lammps/flux/111/hlrn/" + name
    fluxfolder = home + "/lammps/flux/"
    csv_file = fluxfolder + name + ".csv"
    df = pd.read_csv(csv_file, sep=',')
    df['z'] = df['z'] - zSurface
    return df


def SliceDataFrame(df, zSurface, MaxStep=1600000, LoopSize=20):

    density = np.array([])
    density_incoming = np.array([])
    density_outgoing = np.array([])
    LoopSize_inv = 1.0 / LoopSize
    for step in range(MaxStep-LoopSize*1000, MaxStep, 1000):
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

def integrate(y, dx):
    I = [0.0 for i in range(len(y))]
    sum = 0.0
    for i in range(1,len(y)):
        I[i] = y[i] * dx + I[i-1]
    return I

def CreateHistogram_zDensity(density, density_incoming, density_outgoing, TimeSteps):
    print("Create Density Profile")
    area = 15.57 # nm^2
    nbin = 600
    # nbin2 = nbin * 2
    nu = 35
    smnu = 7
    # minval = float(min(density)) - 1.0
    minval = 0.0
    maxval = float(max(density))

    LoopSize_inv = 1.0 / TimeSteps
    NumOfTraj = 20 # caution! hardcoded

    bins = np.linspace(minval, maxval, nbin)

    weight_array1 = np.full(density.shape, 1.0)
    weight_array2 = np.full(density_incoming.shape, 1.0)
    weight_array3 = np.full(density_outgoing.shape, 1.0)

    hist_density = np.histogram(density, bins=bins, weights=weight_array1, density=False)
    hist_density_incoming = np.histogram(density_incoming, bins=bins, weights=weight_array2, density=False)
    hist_density_outgoing = np.histogram(density_outgoing, bins=bins, weights=weight_array3, density=False)

    # normalize density histogram

    binwidth_inv = 1.0 / float(bins[1]-bins[0])
    binwidth = float(bins[1]-bins[0])

    # normalize density histogram to 1 / nm
    # note: actual normalization regarding surface area (-> 1 nm^2) should happen in plot function

    # to normalize the density we need to multiply by the correct binwidth
    # and further convert from AA to nm
    conversionAngstromToNano_inv = 1e-9 / 1e-10
    hist_density[0][:] *= binwidth_inv * conversionAngstromToNano_inv * LoopSize_inv / NumOfTraj / area
    hist_density_incoming[0][:] *= binwidth_inv * conversionAngstromToNano_inv * LoopSize_inv / NumOfTraj / area
    hist_density_outgoing[0][:] *= binwidth_inv * conversionAngstromToNano_inv * LoopSize_inv / NumOfTraj / area

    mean = np.mean(hist_density[0][nbin//2:])

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
    T_QT = data[11]
    T_CT = data[12]
    T_TQ = data[13]
    T_CQ = data[14]
    f.close()
    print("Successfully read data from %s" % fname)
    del data

    return lambda1, lambda2, c1, c2, R21, R22, N1, N2, te, T_QT, T_CT, T_TQ, T_CQ

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

# with this function we try to find the point in time, where
# the simulation data and the proposed solution for the bound particle density over time
# show a too large discrepancy (larger than the tolerance)
# then return the exact time, and the value of the solution at that time
# which shall serve as the initial value for the SRT model afterwards
def findTimeOfDiscrepancy(md, solution, tolerance):
    for i, _ in enumerate(md):
        # the solution may start out smaller than 0
        # this is unphysical and just an artefact of the analytical description
        if(md[i] <= 0 or solution[i] < 0):
            continue

        # define normalized difference
        normedDiff = (md[i] - solution[i]) / md[i]

        # find a way to make sure the discrepancy is not only momentarily
        if (np.abs(normedDiff) > tolerance):
            # take the value some timesteps before, when the discrepancy is not hopeless
            return i-10, solution[i-10]

    # default value takes last values in the arrays
    # TODO
    return len(md), solution[len(md)]




def main():
    global parameter_set_str
    global Block, saveflag, savedir, writeflag, hidesolflag
    global area, zSurface
    lbound = 0.0
    rbound = 5.0
    parser = argparse.ArgumentParser()
    angle, temp_S, temp_P, pressure, \
        beamFlag, saveflag, Block, writeflag, \
        hidesolflag, ptFlag, gofrFlag, niaFlag = InpParams(parser)
    if ptFlag:
        lattice_const = 3.96
    else:
        lattice_const = 4.08
    substrateZmax = lattice_const * 3.
    substrateYmax = lattice_const * 9. * np.sqrt(3) / np.sqrt(2)
    substrateXmax = lattice_const * 12. / np.sqrt(2)
    trajnum = 20
    area = substrateXmax * substrateYmax * 1e-20

    # construct directory/file name
    if niaFlag:
        parameter_set_str = "nia"
    elif ptFlag:
        parameter_set_str = "Pt"
    else:
        parameter_set_str = ""
    parameter_set_str += SetParamsName(angle, temp_S, temp_P, pressure)

    print(" ")
    print("***********************************************************")
    print(" Evaluate", parameter_set_str)
    print("***********************************************************")
    print(" ")

    if (beamFlag == True):
        parameter_set_str += 'Beam'
    df = ReadFile(parameter_set_str, zSurface)
    # change density
    # print(df.describe())
    NumOfTraj = df['traj'].max()
    trajnum = NumOfTraj+1
    print("Number of trajectories:",trajnum)
    MaxStep = df['step'].max()
    MaxStep = MaxStep - (MaxStep % 1000)
    MaxStep2 = MaxStep - 5000
    print("MaxStep:",MaxStep)
    LoopSize = int(temp_S / 2)
    density, density_incoming, density_outgoing = SliceDataFrame(df, zSurface, MaxStep=MaxStep2, LoopSize=LoopSize)
    if temp_P ==  300:
        energy = 25.90
    if temp_P == 190:
        energy = 16.027


    '''
    getSticking()
    '''
    # TODO: setup parallelization
    if(False):

        stickFile = open("/home/becker/lammps/sticking"+parameter_set_str+'.txt', 'w')
        timeMin = 0
        deltaT = 200000
        timeMax = timeMin + deltaT


        duration = 80000
        duration = 40000
        while(timeMax+duration <= MaxStep2):
            totalParticles = 0
            reflectionCounter = 0


            depth = 0 # depth of recursion tree
            # should be a linear "tree", therefore depth should equal the number of timesteps we have checked
            for traj in range(20):
                print("\tCalculate Sticking Probability in Trajectory " + str(traj), sep=' ', end='\r', file=sys.stdout, flush=True)
                totalAux, reflectionAux, depth = cs.getSticking(df, timeMin, timeMax, duration, traj, 0, 0, 0, depth)
                reflectionCounter += reflectionAux
                totalParticles += totalAux

            print(" ")
            print("reflected particles:", reflectionCounter)
            print("total particles:", totalParticles)
            initialSticking = 1.0 - reflectionCounter / totalParticles
            print("Initial Sticking:", initialSticking)
            print("For time range", timeMin*0.00025, "to", timeMax*0.00025, "ps")
            print(" ")
            stickFile.write("%f, %f\n" %((timeMin + timeMax) * 0.5 * 0.00025, initialSticking))
            timeMin = timeMin + int(0.5 * deltaT)
            timeMax = timeMax + int(0.5 * deltaT)

        stickFile.close()
        sys.exit()



    # plot coverage configuration
    savepath = savedir + "cov/"
    name = parameter_set_str

    if gofrFlag:
        print("Error! Please calculate radial pair distribution via C program.")

    coverage = plot.PlotCoverage(df, angle, temp_S, temp_P, pressure, block=False, MaxStep=MaxStep2, xlabel="x / Angström", ylabel="y / Angström",
        saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)

    timeRange = 100000
    # coverage = plot.getCoverage(df, MaxStep, timeRange, trajnum)
    coverage = 38.3022
    print("Coverage:", coverage)


    if not hidesolflag:
        m = 40.0*au # argon mass
        AnSol, NumData = ms.analyticalSolution(angle, temp_S, energy, pressure, temp_P, m, MaxStep2, coverage)
        Stat = []
        Slope = []
    else:
        AnSol, Stat, Slope, NumData = [], [], [], []
    if temp_S <= 160:
        Stat = []


    if(False):
        '''
        execute fit to simple rate equation model
        '''
        Time = np.arange(0,MaxStep,1000)
        Time = Time * 0.25 / 1000.
        M = max_cov
        if pressure == 1.0:
            u0 = -0.087569 * e0 # convert u0 from eV to J
        elif pressure == 2.0:
            u0 = -0.089507 * e0
        else:
            u0 = -0.09 * e0
        _, Phi = ms.ChemPotentialGas(temp_P, pressure)
        # alpha = ms.PartitionFunction(u0, temp_P) * np.exp(Phi / (kB * temp_P))
        alpha = 0.05

        Bound = 5.0
        paramname = str("Single_a%.2ft%de%.2f.dat" % (angle, temp_S, energy))
        fname = home + "/lammps/" + "111" + "/" + paramname
        _,_,_,_,_,_, N1, N2, te, _,_,_,_ = OpenSingleTrajData(fname)

        AnSol = ms.SurfaceAtoms(Time, alpha, M, pressure * atm, int(te/stw), N1+N2)
        Bound_Particles = df.loc[df['z'] < Bound, ['step', 'vz', 'pe']]
        particleCount = []
        for i in range(0,MaxStep, stw):
            TimeResolved = Bound_Particles.loc[Bound_Particles['step'] == i, ['vz']]
            particleCount.append(TimeResolved['vz'].count())
        particleCount = sf(particleCount, 77, 3, deriv=0) / 20 # divide by traj number
        te = 48
        te = 88 # manually obtained starting value, where md data shows actual growth above 0
        xfit = np.arange(0,MaxStep,stw)
        yfit = particleCount

        independent_vars = ['time', 'pressure', 'N0', 'start']
        gmodel = Model(ms.IntegrateTheta, independent_vars=['time'], param_names=['K','pressure','N0','start','alpha'])
        delta=1.0e-11
        gmodel.set_param_hint('K', value=2.294273e-18)
        gmodel.set_param_hint('alpha', value=0.05, min=0.05-delta, max=0.05+delta)
        gmodel.set_param_hint('pressure', value=pressure, min=pressure, max=pressure+delta)
        gmodel.set_param_hint('N0', value=N1+N2, min=N1+N2, max=N1+N2+delta)
        gmodel.set_param_hint('start', value=te, min=te, max=te+delta)
        pars = gmodel.make_params()
        result = gmodel.fit(yfit, pars, time=xfit)

        fitparam = result.params['K'].value
        fitparamAlpha = result.params['alpha'].value
        print(result.fit_report())

        # plt.plot(xfit*0.00025, ms.IntegrateTheta(xfit, 1.947590e-18, pressure, N1+N2, int(te)), label='fitted solution, K=%e, p=%f' %(1.947590e-18, pressure))
        plt.plot(xfit*0.00025, ms.IntegrateTheta(xfit, 2.294273e-18, pressure, N1+N2, int(te)), label='fitted solution, K=%e, p=%f' %(2.294273e-18, pressure))
        # plt.plot(xfit*0.00025, ms.IntegrateTheta(xfit, fitparam, pressure, N1+N2, int(te)), label='fitted solution, K=%e, alpha=%e, p=%f' %(fitparam, fitparamAlpha, pressure))
        plt.plot(xfit*0.00025, particleCount, label='mddata')
        plt.legend()
        plt.savefig("/home/becker/lammps/flux/Fit"+parameter_set_str+".pdf")
        plt.show(block=False)
        plt.clf()
        plt.cla()
        sys.exit()
    '''
    end fit
    '''

    '''
    TODO:
    # find the time when we have too much of a discrepancy between our model and the numerical data
    # at this time (and probably some time before) our model breaks down and we want to revert to
    # an alternative rate equation model with the results from our solution as initial values
    endTime = -1
    for i, _ in enumerate(mdData):
        if(mdData[i] != 0):
            normedDifference = (population[i] - mdData[i]) / mdData[i]
        else:
            continue
        if(i > 2.0 * te and np.abs(normedDifference) > 1.1):
            endTime = i
            break
    print(endTime)
    '''

    # obtain parameter specific data for equilibrium coverage as well as average potential energy per bound particle (in equilibrium)
    num, potE = GetPotentialOfTheta(df, lbound, rbound)
    if(False):
        WriteTrajData('param.dat', angle, temp_S, temp_P, pressure, coverage, potE)

    # ******************************
    # plot different quantities
    # ******************************

    # density over time
    savepath = savedir + "dens/"
    name = savepath + parameter_set_str + "DensTime"
    # name = home + "/lammps/newFluxPlot/" + parameter_set_str + "DensTime"
    particleCount, partCountGas, TimeArr = plot.createDensityOverTime(df, NumOfTraj=NumOfTraj, MaxStep=MaxStep2)

    # find time, where our solution no longer describes the simulation data
    discrepantTime, initValueSRT = findTimeOfDiscrepancy(particleCount, AnSol, 0.3)

    # ******************************************
    # at this time, implement the SRT solution:
    # ******************************************
    # M and M0 are values inherent to the simulation setup
    M = 121. # maximum adsorption sites (taken from simulations at 80 K)
    M0 = 216 # number of particles in uppermost surface layer
    m = 40. * au # argon mass
    # coverage: taken from function plot.Coverage
    # area: standard value
    alpha = 0.05 # langmuir fit parameter for T = 300 K
    dt = 0.25 # dt in ps


    # reminder:
    # TimeArr = np.arange(0,maxsteps,1000)
    # vs.
    # TimeArr = TimeArr * 0.25 / 1000.
    # luckily the time definitions for both quantities (pressure and solution)
    # turn out to be the same
    # therefore we can use the timestep defined as discrepantTime directly as
    # initial value
    deltaTimeSolution = 0.25

    t0 = int(discrepantTime)
    print("discrepant time:", t0)
    t0 = 350
    if pressure == 2.0:
        t0 = 450
    if pressure == 4.0:
        t0 = 400
    if pressure >= 8.0:
        t0 = 250
    if pressure >= 12.0:
        t0 = 120


    theta0 = AnSol[t0]
    timeArray = np.arange(0.0, 1000.0, dt)

    # get bulk pressure

    # dir = home + "/lammps/flux/plot"
    # pressureBulk = ms.pressureNaive(dir, parameter_set_str)

    # for debugging
    # exponential fit to bulk pressure evolution
    # pressureBulk = ms.pressureExp(timeArray, 4.38060634, 0.97202180, 109.458062)

    # pressureBulk = ms.pressureExp(timeArray, 43.4523347, 1.51888070, 75.8685685)
    # tau should be around 55 ps
    pressureBulk = ms.pressureExp(timeArray, pressure*1.0, pressure, 54.25) # try to find sensible values

    # print("********************************")
    # print("lengths:")
    # print(len(pressureBulk))
    # print(len(timeArray))

    # get solution from SRT model
    if temp_S == 190:
        alpha = 0.2
    Population2 = ms.integrateTheta(pressureBulk, M, M0, coverage, area, m, temp_P, alpha, dt, t0, timeArray, theta0)

    for i, pop in enumerate(Population2):
        Population2[i] = Population2[i] * M0 / 15.57
    print("eq cov:", Population2[-1])

    conditionArray = [False for i in range(t0)]
    conditionArray = conditionArray + [True for i in range(t0, len(TimeArr))]
    # print(conditionArray)
    maskedAnSol = np.ma.masked_where(conditionArray, AnSol)

    maskedPopulation2 = np.ma.masked_where(not conditionArray, Population2)
    Stationary = []
    # mask analytical solution
    # / overwrite stationary values with SRT solution
    # for i, sol in enumerate(AnSol):
    #     if i < t0:
    #         continue
    #     else:
    #         AnSol[i] = Population2[i] * M0 / 15.57
    # Stationary = [Population2 for i in range(len(TimeArr2))]
    area_nm = 15.57
    plot.PlotDensityOverTime(xlabel="t / ps", ylabel=r'Particles per nm$^2$',
                            Population=AnSol / area_nm, Stationary=Stationary, Slope=[], mdData=particleCount,
                            mdDataGas=partCountGas, TimeArr=TimeArr, pressure=pressure,
                            Population2=Population2[t0:], TimeArr2=timeArray[t0:], ylabel2=r'Particles per nm$^{3}$',
                            saveflag=saveflag, savedir=name, writeflag=writeflag, block=Block)
    # sys.exit()

    # density over height
    savepath = savedir + "dens/"
    name = savepath + parameter_set_str + "DensHeight"
    hist_density, hist_density_incoming, hist_density_outgoing = CreateHistogram_zDensity(
                                                                    density, density_incoming, density_outgoing, LoopSize)
    plot.PlotDensityHistogram(
        X=[hist_density[1][:-1], hist_density_incoming[1][:-1], hist_density_outgoing[1][:-1]],
        Y=[hist_density[0], hist_density_incoming[0], hist_density_outgoing[0]],
        Label=['Total density','Incoming density','Outgoing density'], writeflag=writeflag,
        block=Block, NumOfTraj=NumOfTraj, xlabel="z / Angström", ylabel=r"Density / nm$^{-3}$", saveflag=saveflag, savedir=name)

    # kinetic energy over time
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

    # kinetic energy over height
    savepath = savedir + "ke/"
    name = parameter_set_str + "KinEnHeight"
    plot.PlotKineticEnergyOverHeight(df, block=Block, xlabel='z / Angström', ylabel='E / K',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)

    # potential energy over height
    savepath = savedir + "pe/"
    name = parameter_set_str + "PotEnHeight"
    PE = plot.PlotPotentialEnergyOverHeight(df, block=Block, xlabel='z / Angström', ylabel='E / eV',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)

    # potential energy over time
    savepath = savedir + "pe/"
    name = parameter_set_str + "PotEnTime"
    PE_t, PEstd_t = plot.PlotPotentialEnergyOverTime(df, block=Block, xlabel='t / ps', ylabel='E / eV',
        MaxStep=MaxStep2, saveflag=saveflag, savedir=savepath, savename=name, writeflag=writeflag)





if __name__ == "__main__":
    main()

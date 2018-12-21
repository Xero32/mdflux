import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from md_evalflux import NoPunctuation
from scipy.signal import savgol_filter as sf
from lmfit import Model
##### Constants
kB = 1.3806503e-23
e0 = 1.60217662e-19
pi = 3.14159265359
au = 1.66053904e-27
atm = 101325.0
NA = 6.02214086e23

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


def SetParamsName(angle, temp_S, temp_P, pressure):
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"
    return parameter_set_str


def pressureTensor(dir, paramsString, pressure, saveFlag=False, Block=False):
    home = str(Path.home())
    pFile = open(home+"/lammps/ptensor" + paramsString + ".csv")
    kinEnInFile = open(dir + "/ke/" + paramsString + "KinEnHeightIn.csv")
    kinEnOutFile = open(dir + "/ke/" + paramsString + "KinEnHeightOut.csv")
    densFile = open(dir + "/dens/" + paramsString + "DensHeight.csv")

    pTensor = pd.read_csv(pFile, sep=',')
    pTensor.columns = ['z', 'p_u', 'p_k', 'p_s', 'p_ref']
    kinEnIn = pd.read_csv(kinEnInFile, sep=',', na_filter=True, skiprows=lambda x: x in np.arange(1,10))
    kinEnIn.columns = ['z', 'Ekin']
    kinEnOut = pd.read_csv(kinEnOutFile, sep=',', na_filter=True, skiprows=lambda x: x in np.arange(1,10))
    kinEnOut.columns = ['z', 'Ekin']
    dens = pd.read_csv(densFile, sep=',')
    dens.columns = ['z', 'dens']

    pFile.close()
    kinEnInFile.close()
    kinEnOutFile.close()
    densFile.close()


    dh = 0.5
    minHeight = kinEnIn['z'].min()
    heightArr = np.arange(minHeight, 60.0, dh)
    print(pTensor.describe())
    print(kinEnIn.describe())
    print(kinEnOut.describe())
    dens['dens'] *= 1e27
    print(dens.describe())

    pressureApprox = []
    for h in heightArr:
        auxDens = dens.loc[(h < dens['z']) & ( dens['z'] <= h+dh), ['dens']]
        auxKinEnIn = kinEnIn.loc[(h < kinEnIn['z']) & (kinEnIn['z'] <= h+dh), ['Ekin']]
        auxKinEnOut = kinEnOut.loc[(h < kinEnOut['z']) & (kinEnOut['z'] <= h+dh), ['Ekin']]

        # calculate n k T
        # with local density and kinetic energy
        # whereby the kin energy is the mean of the incoming and outgoing contributions
        # np.append(pressureApprox, auxDens['dens'].mean() * 0.5 * (auxKinEnIn['Ekin'].mean() + auxKinEnOut['Ekin'].mean()) * kB)

        value = auxDens['dens'].mean() * 0.5 * (auxKinEnIn['Ekin'].mean() + auxKinEnOut['Ekin'].mean()) * kB / atm
        # value = auxDens['dens'].mean() * auxKinEnIn['Ekin'].mean() * kB / atm

        pressureApprox.append(value)



    # plot full pressure tensor
    plt.plot(pTensor['z'], pTensor['p_u']+pTensor['p_k']-pTensor['p_s'], label=r'p$_{zz}$')
    plt.plot(pTensor['z'], pTensor['p_k'], label=r'p$_{zz}^{kin}$')
    # plt.plot(kinEnIn['z'], kinEnIn['Ekin'] * kB * density / atm, label=r'p$_{ref}$')
    plt.plot(kinEnIn['z'], pressureApprox[:-1], label=r'p$_{ref}$')
    plt.tight_layout()
    plt.legend()
    if saveFlag:
        plt.savefig(home + "/lammps/pComp" + paramsString + ".pdf")
    if Block:
        plt.show(block=True)
    plt.clf()
    plt.cla()
    # plot zoomed pressure tensor
    plt.plot(pTensor['z'], pTensor['p_u']+pTensor['p_k']-pTensor['p_s'], label=r'p$_{zz}$')
    plt.plot(pTensor['z'], pTensor['p_k'], label=r'p$_{zz}^{kin}$')
    # plt.plot(kinEnIn['z'], kinEnIn['Ekin'] * kB * density / atm, label=r'p$_{ref}$')
    plt.plot(kinEnIn['z'], pressureApprox[:-1], label=r'p$_{ref}$')
    plt.axis([0.0,60.0, -0.5, 50.0])
    if(pressure > 8):
        plt.axis([0.0,60.0, 0.0, 250.0])
    plt.tight_layout()
    plt.legend()
    if saveFlag:
        plt.savefig(home + "/lammps/pCompZoom" + paramsString + ".pdf")
    if Block:
        plt.show(block=True)
    plt.clf()
    plt.cla()

def pressureExp(t, pss, p0, tau, t0):
    return pss - (pss - p0) * np.exp(-t/tau + t0/tau)

def pressureNaive(dir, paramsString, pressure, block=True):
    kinEnGasFile = open(dir + "/ke/" + paramsString + "KinEnTimegas.csv") # contains kin energy in units of kelvin
    densGasFile = open(dir + "/dens/" + paramsString + "DensTimeGas.csv")

    kinEnGas = pd.read_csv(kinEnGasFile, sep=',', names=['t', 'xKE', 'yKE', 'zKE'], dtype=np.float64, skiprows=1)
    # kinEnGas.columns = ['t', 'xKE', 'yKE', 'zKE']
    densGas = pd.read_csv(densGasFile, sep=',')
    densGas.columns = ['t', 'dens']
    # densGas['dens'] *= 1e27 # work on conversion, since we saved the gas density as area density
    area = 1557e-20
    conversion = 1. / area * (1e-9)**2
    densGas['dens'] /= conversion # reconvert to total particle number
    print(densGas.describe())

    volume = area * 55e-10 # volume in m^-3
    densGas['dens'] *= 1.0 / volume * kB # == n * kB * T
    kinEnGasFile.close()
    densGasFile.close()

    minTime = 0
    maxTime = densGas['t'].max()
    STW = 1000
    dt = 2.5
    timeArr = np.arange(minTime, maxTime, dt)
    print(kinEnGas.describe())

    pressureGas = []

    for t in timeArr:
        auxDens = densGas.loc[(t <= densGas['t']) & (densGas['t'] < t+dt), ['dens']]
        auxKinEn = kinEnGas.loc[(t <= kinEnGas['t']) & (kinEnGas['t'] < t+dt), ['xKE', 'yKE', 'zKE']]

        value = 2.0 * auxDens['dens'].mean() * auxKinEn['zKE'].mean() + auxDens['dens'].mean() * (auxKinEn['xKE'].mean() + auxKinEn['yKE'].mean())
        pressureGas.append(value / atm)

    # print(pressureGas)
    pressureGas = sf(pressureGas, 67, 3, deriv=0)



    # time at which exponential pressure evolution should hold true
    # before that, we can not make the assumption of strict exponential pressure evolution
    t0 = 50 # corresponds to 125 ps
    yfit = pressureGas[t0:]
    ts = 100 # = 250 ps
    saturatedPressure = np.mean(pressureGas[ts:])
    xfit = timeArr[t0:]
    independent_vars = ['t', 'pss', 'p0', 'tau', 't0']
    gmodel = Model(pressureExp, independent_vars=['t'], param_names=['pss','p0','tau','t0'])
    delta=1.0e-11
    gmodel.set_param_hint('pss', value=saturatedPressure, min=0.85*saturatedPressure, max=1.15*saturatedPressure)
    gmodel.set_param_hint('p0', value=pressureGas[t0], min=0.5*pressureGas[t0], max=1.5*pressureGas[t0])
    gmodel.set_param_hint('tau', value=55.0, min=30.0, max=120.0)
    gmodel.set_param_hint('t0', value=t0, min=t0-10, max=t0+10)

    pars = gmodel.make_params()
    result = gmodel.fit(yfit, pars, t=xfit)

    fitPss = result.params['pss'].value
    fitP0 = result.params['p0'].value
    fitTau = result.params['tau'].value
    fitT0 = result.params['t0'].value
    print(result.fit_report())


    plt.plot(timeArr, pressureGas, label='pressureGas')

    # factor 2.5: conversion between array index and timescale
    plt.plot(timeArr, pressureExp(timeArr, fitPss, fitP0, fitTau, t0*2.5), label='Fit')
    # plt.plot(timeArr, pressureExp(timeArr, saturatedPressure, pressureGas[t0], 55.0, t0*2.5), label='input values')
    # p_ss = 1.0
    # plt.plot(timeArr, p_ss - (p_ss - p_0) * np.exp(-timeArr / tau))
    plt.legend()
    plt.axis((0,maxTime, 0.95 * np.min(pressureGas), 1.05 * np.max(pressureGas)))
    plt.show(block=block)
    plt.clf()
    plt.cla()

    # return fit parameters + values obtained from MD: saturated pressure, pressure at time t0, and t0 itself
    return fitPss, fitP0, fitTau, saturatedPressure, pressureGas[t0], t0*2.5




def main():
    parser = argparse.ArgumentParser()
    # most of these parameters and especially flags are obsolete
    # I just copied it from md_evalflux.py
    angle, temp_S, temp_P, pressure, \
        beamFlag, saveFlag, Block, writeFlag, \
        hidesolFlag, ptFlag, gofrFlag, niaFlag = InpParams(parser)
    home = str(Path.home())
    dir = home + "/lammps/flux/plot"

    paramsString = SetParamsName(angle, temp_S, temp_P, pressure)
    print(type(paramsString))
    print(" ")
    print("***********************************************************")
    print(" Evaluate", paramsString)
    print("***********************************************************")
    print(" ")
    # pressureTensor(dir, paramsString)
    pressureNaive(dir, paramsString, pressure)
    pressureTensor(dir, paramsString, pressure, Block=Block)

if __name__ == '__main__':
    main()

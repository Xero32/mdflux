import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from md_evalflux import NoPunctuation
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
    print(" ")
    print("***********************************************************")
    print(" Evaluate", paramsString)
    print("***********************************************************")
    print(" ")

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
        np.append(pressureApprox, auxDens['dens'].mean() * 0.5 * (auxKinEnIn['Ekin'].mean() + auxKinEnOut['Ekin'].mean()) * kB)
        value = auxDens['dens'].mean() * 0.5 * (auxKinEnIn['Ekin'].mean() + auxKinEnOut['Ekin'].mean()) * kB / atm
        pressureApprox.append(value)

    # density = dens['dens'].loc[(dens['z'] > 30)].mean()
    #
    # print("density",density)
    # meanNRG = (kinEnIn['Ekin'] + kinEnOut['Ekin']) * 0.5
    #
    # roomTemp = np.full(len(kinEnIn['z']), 300.0)
    # print(meanNRG.describe())


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
if __name__ == '__main__':
    main()

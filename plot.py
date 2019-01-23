# plot.py
import matplotlib.pyplot as plt
import smooth
import math
import stats
from scipy.signal import savgol_filter as sf
import numpy as np
import numpy.ma as ma
import cfg as g
from scipy.optimize import curve_fit
from lmfit import Model
from pathlib import Path
import sys

def WritePlot(X=[], Y=[], name='', xlabel='', ylabel='', saveflag=True, header=True, action='w'):
    if saveflag == False:
        return -1
    name = name + '.csv'
    f = open(name, action)
    f.write('#%s, %s\n' % (xlabel, ylabel))

    aux = 0
    try:
        aux = len(Y[0])
    except:
        pass

    if aux > 0:
        for i in range(len(Y[0])):
            x = X[i]
            f.write('%f' % x)
            for j in range(len(Y)):
                f.write(', %f' % (Y[j][i]))
            f.write('\n')
    else:
        for i,y in enumerate(Y):
            x = X[i]
            f.write('%f, %f\n' % (x,y))

    f.close()
    return 0

def SetupPlot(Title, xlbl, ylbl, grid=False, Block=True, saveflag=0, savename='', sz=12, legend=1):
    # plt.title(Title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    if legend == 1:
        plt.legend(prop={'size': sz})
    plt.grid(grid)
    plt.tight_layout()
    if saveflag == 1:
        plt.savefig(savename)
    plt.show(block=Block)
    plt.clf()
    plt.cla()
    plt.close()

def Populations(angle, temp, energy, X, T, Q, C, Title='', smflag=1, pltflag=1, nu=5, hlrn=0):
    TIMESTEPS = len(T)
    DT = 1
    if hlrn == 1:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]+'HLRN'
    else:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]
    if smflag == 1:
        print("Smoothing Populations")
        ctr = 0
        for k in range(0,TIMESTEPS):
            if k % (TIMESTEPS / 10) == 0:
                print(angle, temp, '\t'+str(ctr)+' %')
                ctr += 10
            C[k] = smooth.GaussSmoothing(TIMESTEPS, k, C, DT, nu=nu)
            Q[k] = smooth.GaussSmoothing(TIMESTEPS, k, Q, DT, nu=nu)
            T[k] = smooth.GaussSmoothing(TIMESTEPS, k, T, DT, nu=nu)
        print(angle, temp, '\t100 %')

    if pltflag == 1:
        print('Plot Populations')
        # fig, ax = plt.subplots(num='a'+angle+'t'+temp+'e'+energy)
        plt.plot(X*2.5e-2, C, 'r-', label = r'$N_C$')
        plt.plot(X*2.5e-2, Q, 'g--', label = r'$N_Q$')
        plt.plot(X*2.5e-2, T, 'b-.', label = r'$N_T$')
        svname = '/home/becker/lammps/newplot/Pop/' + name + 'Population'
        SetupPlot(Title, 'Time / ps', 'Population Fraction', saveflag=g.S_POP, savename=svname+'.pdf', Block=1-g.S_POP)
    WritePlot(X=X, Y=[C,Q,T], name=svname, xlabel='Time / ps', ylabel='Population Fraction C,Q,T', saveflag=g.S_POP, header=True)

def TransitionPopulations(angle, temp, energy, X, QT, TQ, CT, CQ, TC=[], QC=[], Title='', smflag=1, pltflag=1, nu=49, hlrn=0, numpltflag=0):
    plt.clf()
    plt.cla()
    TIMESTEPS = len(QT)
    l = len(QT)
    DT = 1
    # fig, ax = plt.subplots(num='a'+angle+'t'+temp+'e'+energy)
    if hlrn == 1:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]+'HLRN'
    else:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]
    if numpltflag == 1:
        plt.plot(X*2.5e-2, QT, 'b--', label = 'QT')
        plt.plot(X*2.5e-2, TQ, 'C1--', label = 'TQ')
        plt.plot(X*2.5e-2, CT, 'g--', label = 'CT')
        plt.plot(X*2.5e-2, CQ, 'r--', label = 'CQ')
    if smflag == 1:
        start = int(5 * l/30)
        Edge = np.arange(-start-5,l-start-5,1)
        print("Smoothing Transition Populations")
        ctr = 0
        # sf = scipy.signal.savgol_filter
        QT[:] = sf(QT, nu, 3, deriv=0)
        TQ[:] = sf(TQ, nu, 3, deriv=0)
        CT[:] = sf(CT, nu, 3, deriv=0)
        CQ[:] = sf(CQ, nu, 3, deriv=0)
        '''
        for k in range(0,TIMESTEPS):# - int(0.01*l)):
            if k % (TIMESTEPS / 10) == 0:
                print(angle, temp, '\t'+str(ctr)+' %')
                ctr += 10

            QT[k] = smooth.GaussSmoothing(TIMESTEPS, k, QT, dt=DT, nu=nu, edge='', X=Edge)
            TQ[k] = smooth.GaussSmoothing(TIMESTEPS, k, TQ, dt=DT, nu=nu, edge='', X=Edge)
            CT[k] = smooth.GaussSmoothing(TIMESTEPS, k, CT, dt=DT, nu=nu, edge='', X=Edge)
            CQ[k] = smooth.GaussSmoothing(TIMESTEPS, k, CQ, dt=DT, nu=nu, edge='', X=Edge)
        print(angle, temp, '\t100 %')
        '''
    # plot the results
    if pltflag == 1:
        print("Plot Transition Populations")
        #fig, ax = plt.subplots(num='a'+angle+'t'+temp+'e'+energy)
        plt.plot(X*2.5e-2, QT, 'g--', label = r'd$N_{QT}$')
        plt.plot(X*2.5e-2, TQ, 'b-.', label = r'd$N_{TQ}$')
        #ax.plot(X*2.5e-2, CT, 'r--', label = 'CT')
        plt.plot(X*2.5e-2, CQ, 'r-', label = r'd$N_{CQ}$')
        try:
            plt.plot(X*2.5e-2, TC, '-', label = 'TC')
            plt.plot(X*2.5e-2, QC, '-', label = 'QC')
        except:
            dummy=1

        legend = plt.legend()

        svname = '/home/becker/lammps/newplot/Trans/' + name + 'Transition'
        SetupPlot(Title, 'Time / ps', 'Transition Populations', saveflag=g.S_TRANS, savename=svname+'.pdf', Block=1-g.S_TRANS)
    WritePlot(X=X, Y=[QT, TQ, CT, CQ], name=svname, xlabel='Time / ps', ylabel='Transition Populations QT, TQ, CT, CQ', saveflag=g.S_TRANS)
    return QT, TQ, CT, CQ, TC, QC


#tpl: # nb, t1, t2, tm, s1, s2, transition, (energy loss ? ) # tuple with information about bounce events
def TransitionRate(angle, temp, energy, X, Ta, Tb, Tc, Td, lblA='', lblB='', lblC='', lblD='', Title='', smflag=1, pltflag=1, nu=8, ylbl='', avgflag=0, start=0, end=0, hlrn=0):
    plt.cla()
    plt.clf()
    if hlrn == 1:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]+'HLRN'
    else:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]
    if smflag == 1:
        print("Smoothing Transition Rates")

        for k in range(0,len(Ta)):
            Ta[k] = smooth.GaussSmoothing(len(Ta), k, Ta, dt=1, nu=nu)
            Tb[k] = smooth.GaussSmoothing(len(Ta), k, Tb, dt=1, nu=nu)
            Tc[k] = smooth.GaussSmoothing(len(Ta), k, Tc, dt=1, nu=nu)
            Td[k] = smooth.GaussSmoothing(len(Ta), k, Td, dt=1, nu=nu)

    if pltflag == 1:
        print("Plot Transition Rates")
        # fig, ax = plt.subplots(num='a'+angle+'t'+temp+'e'+energy)
        if avgflag == 1:
            avga = stats.AvgRate(start, end, Ta)
            avgb = stats.AvgRate(start, end, Tb)
            avgc = stats.AvgRate(start, end, Tc)
            avgd = stats.AvgRate(start, end, Td)
            Avga = np.array([avga for i in range(len(Ta))])
            Avgb = np.array([avgb for i in range(len(Ta))])
            Avgc = np.array([avgc for i in range(len(Ta))])
            Avgd = np.array([avgd for i in range(len(Ta))])
            plt.plot(X*2.5e-2, Avga, 'g:', label='', linewidth=1)
            plt.plot(X*2.5e-2, Avgb, 'b:', label='', linewidth=1)
            #ax.plot(X*2.5e-2, Avgc, 'r:', label='', linewidth=1)
            plt.plot(X*2.5e-2, Avgd, 'r:', label='', linewidth=1)

        plt.plot(X*2.5e-2, Ta, 'g--', label = lblA)
        plt.plot(X*2.5e-2, Tb, 'b-.', label = lblB)
        #ax.plot(X*2.5e-2, Tc, 'r-', label = lblC)
        plt.plot(X*2.5e-2, Td, 'r-', label = lblD)
        legend = plt.legend()
        '''
        if hlrn == 1:
            plt.axis([0,60,-0.1,1.5])
        else:
            plt.axis([0,40,-0.1,1.5])
        '''
        svname = '/home/becker/lammps/newplot/T/' + name + 'TransitionRate'
        SetupPlot(Title, 'Time / ps', ylbl, saveflag=g.S_T, savename=svname+'.pdf', Block=1-g.S_T)
    WritePlot(X=X, Y=[Ta, Tb, Tc, Td], name=svname, xlabel='Time / ps', ylabel=ylbl + lblA + ', ' + lblB + ', ' + lblC + ', ' + lblD,
        saveflag=g.S_T)
    return Ta, Tb, Tc, Td

def Histogram(emin, emax, nbin, A, B, C, subplts=0, lblA='', lblB='', lblC='', lbl='', Title='', binarr=[], avg=0, std=0, xlbl='', ylbl='',
 acol='black', bcol='red', ccol='blue', totcol='green', saveflag=0, savename='', subscale=1,
 totalplt=1, sm_flag=1, pltflag=1, nu=4, edge='none', edgeA='none', edgeB='none', edgeC='none', writeflag=0, flname='', nb=0, action=-1, scl=1, vertval=[], vertls=[]):
    # define bin number and boundaries (emin, emax, nbin) and create histograms for up to three quantities
    # (A,B,C) as well as the total of A+B+C
    # results can then be plotted, saved or written out
    #
    # there is a further option to mask certain areas of arrays, e.g. to only plot the positive
    # contribution to kinetic energy components


    #TDOD find a better way to create bins, so that 0 meV is always a bin boundary
    bin_len = 3 #TODO
    average = -3000
    stdev = -9000
    if len(binarr) == 0:
        bns = np.linspace(emin, emax, nbin)
    else:
        bns = binarr
    Tot = np.concatenate((A,B,C))
    hTot = np.histogram(Tot, bins=bns)
    hTotNorm = np.histogram(Tot, density=True, bins=bns)
    #norm = hTot[0][nbin//2] / hTotNorm[0][nbin//2]
    if subscale != 0:
        norm = normA = normB = normC = max(hTot[0]) / max(hTotNorm[0])
    else:
        norm = max(hTot[0]) / max(hTotNorm[0])
        normA = normB = normC = 1
    if subplts == 1:
        if subscale != 0:
            hA = np.histogram(A, bins=bns)
            hB = np.histogram(B, bins=bns)
            hC = np.histogram(C, bins=bns)
        else:
            hA = np.histogram(A, bins=bns, density=True)
            hB = np.histogram(B, bins=bns, density=True)
            hC = np.histogram(C, bins=bns, density=True)
        if sm_flag == 1:
            sA = np.zeros([len(hTot[0])])
            sB = np.zeros([len(hTot[0])])
            sC = np.zeros([len(hTot[0])])
            for k in range(0,len(hTot[0])):
                sA[k] = smooth.GaussSmoothing(len(hA[0]), k, hA[0], edge=edgeA, nu=nu, X=hTot[1][:-1])
                sB[k] = smooth.GaussSmoothing(len(hB[0]), k, hB[0], edge=edgeB, nu=nu, X=hTot[1][:-1])
                sC[k] = smooth.GaussSmoothing(len(hC[0]), k, hC[0], edge=edgeC, nu=nu, X=hTot[1][:-1])
            if(pltflag == 1):
                if edgeA == 'positive' or edgeA == 'pos':
                        plt.plot(ma.masked_less(hA[1][:-1],0), scl*(sA / normA), label=lblA, color=acol)
                elif edgeA == 'negative' or edgeA == 'neg':
                        plt.plot(ma.masked_greater(hA[1][:-1],0), scl*(sA / normA), label=lblA, color=acol)
                else:
                        plt.plot(hA[1][:-1], scl*(sA) / normA, label=lblA, color=acol)

                if edgeB == 'positive' or edgeB == 'pos':
                        plt.plot(ma.masked_less(hB[1][:-1],0), scl*(sB / normB), label=lblB, color=bcol)
                elif edgeB == 'negative' or edgeB == 'neg':
                        plt.plot(ma.masked_greater(hB[1][:-1],0), scl*(sB / normB), label=lblB, color=bcol)
                else:
                        plt.plot(hB[1][:-1], scl*(sB) / normB, label=lblB, color=bcol)

                if edgeC == 'positive' or edgeC == 'pos':
                        plt.plot(ma.masked_less(hC[1][:-1],0), scl*(sC / normC), label=lblC, color=ccol)
                elif edgeC == 'negative' or edgeC == 'neg':
                        plt.plot(ma.masked_greater(hC[1][:-1],0), scl*(sC / normC), label=lblC, color=ccol)
                else:
                        plt.plot(hC[1][:-1], scl*(sC) / normC, label=lblC, color=ccol)

        else:
            if(pltflag == 1):
                plt.plot(hA[1][:-1], scl*(hA[0] / normA), label=lblA, color=acol)
                plt.plot(hB[1][:-1], scl*(hB[0] / normB), label=lblB, color=bcol)
                plt.plot(hC[1][:-1], scl*(hC[0] / normC), label=lblC, color=ccol)
    if totalplt == 1:
        if sm_flag == 1:
            sTot = np.zeros([len(hTot[0])])
            for k in range(0, len(hTot[0])):
                sTot[k] = smooth.GaussSmoothing(len(hTot[0]), k, hTot[0], edge=edge, nu=nu, X=hTot[1][:-1])
            if pltflag == 1:
                plt.plot(hTot[1][:-1], scl*sTot / norm, label=lbl, color=totcol, alpha=0.5)

        else:
            if pltflag == 1:
                plt.plot(hTot[1][:-1], scl*(hTot[0] / norm), label=lbl, color=totcol, alpha=0.5)
    if pltflag == 1:
        for v in range(len(vertval)):
            plt.axvline(x=vertval[v], linewidth=1, color='k', linestyle=vertls[v])

        SetupPlot(Title, xlbl, ylbl, grid=True, Block=1-g.S_NRG, saveflag=saveflag, savename=savename)

    if writeflag == 1 and sm_flag == 1 and subplts == 1:
        fl = open(flname,'a')
        fl.write("#action\t bnc\t X\t\t\t total\t\t\t T\t\t\t\t Q\t\t\t\t C\n")
        for i in range(0, len(hTot[0])):
            fl.write("%d\t\t %d\t %e\t %e\t %e\t %e\t %e\n" %(action, nb, hTot[1][i], sTot[i] * scl / norm, sA[i] * scl / norm, sB[i] * scl / norm, sC[i] * scl / norm))

    elif writeflag == 1:
        print('Can only write to file with activated subplots and smoothing')

    if avg == 1:
        average = np.average(hTot[1][:-1], weights=sTot/norm)
    if std == 1 and avg == 1:
        square = np.average(np.square(hTot[1][:-1]), weights=sTot/norm)
        stdev = math.sqrt(square - (average**2) / nbin)# / nbin
    elif std == 1 and avg != 1:
        print("Error! Need Mean to calculate Std.")

    if subplts == 1:
        if sm_flag ==1:
            return hTot[1], sTot*scl, norm, sA*scl, sB*scl, sC*scl, average, stdev
        else:
            return hTot[1], hTot[0]*scl, norm, hA[0]*scl, hB[0]*scl, hC[0]*scl, average, stdev
    else:
        if sm_flag == 1:
            return hTot[1], sTot*scl, norm, [], [], [], average, stdev
        else:
            return hTot[1], hTot[0]*scl, norm, [], [], [], average, stdev


def Solution(angle,temp,energy,N, Time, TimePrime, StateComp, Title='', pltflag=0, maxtime=60):
    # plot both analytical solution (N[0] and N[1]) and compare it to Simulation data
    # (i.e. StateComp[0] & [1])
    if g.HLRN == 1:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]+'HLRN'
    else:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]
    svname = '/home/becker/lammps/newplot/Sol/' + name + 'Solution'
    if pltflag == 1:
        plt.cla()
        plt.clf()
        plt.axis([0, maxtime, 0, max(StateComp[0])+0.05])
        plt.plot(Time*0.025, N[0], 'b', label=r'$N_{T,prediction}$', alpha=0.5)
        plt.plot(TimePrime*0.025, StateComp[0], 'b-.', label=r'$N_{T,num}$')
        plt.plot(Time*0.025, N[1], 'g', label=r'$N_{Q,prediction}$', alpha=0.5)
        plt.plot(TimePrime*0.025, StateComp[1], 'g--', label=r'$N_{Q,num}$')
        SetupPlot(Title, 't / ps', 'Population Fraction', grid=False, Block=1-g.S_SOL, saveflag=g.S_SOL, savename=svname+'.pdf', sz=12, legend=1)
    WritePlot(X=Time*0.025, Y=[N[0], N[1]], name=svname, xlabel='t / ps', ylabel='Analytical Solution for Population Fraction Trapped, Quasi-Trapped',
        saveflag=g.S_SOL)
    WritePlot(X=TimePrime*0.025, Y=[StateComp[0], StateComp[1]], name=svname+'MDdata',
        xlabel='t / ps', ylabel='MD Data for Population Fraction Trapped, Quasi-Trapped',
        saveflag=g.S_SOL)



# def WritePlot(X=[], Y=[], name='', xlabel='', ylabel='', saveflag=True, header=True, action='w'):



###### mainly for angular distribution
def Distr(emin, emax, nbin, Arr, subplts=0, lbl=[], Title='', binarr=[], avg=0, std=0, xlbl='', ylbl='',
 saveflag=0, savename='', subscale=1, nbb=8, numTraj=2000, a=0, t=0, e=0, NB=[],
 totalplt=1, sm_flag=1, pltflag=1, nu=4, edge='pos', writeflag=0, flname='', nb=0, action=-1, scl=1, vertval=[], vertls=[]):
    # similar to function 'Histogram', instead takes only one input (Arr) and has the option to
    # draw vertical lines for reference

    # find a better way to create bins, so that 0 meV is always a bin boundary
    bin_len = 3 #TODO
    average = -3000
    stdev = -9000

    if len(binarr) == 0:
        bns = np.linspace(emin, emax, nbin)
    else:
        bns = binarr
    OutArr = []

    ctr = 0
    for n in range(nbb):
        Aux = []
        for j in range(numTraj):
            if (Arr[n,1,j]) > 0:
                Aux.append(Arr[n,1,j])
                ctr += 1

        #print(Aux)
        Hist = np.histogram(Aux, bins=bns, density=True)
        S = []
        for k in range(0,len(Hist[0])):
            S.append(smooth.GaussSmoothing(len(Hist[1][:-1]), k, Hist[0], edge=edge, nu=nu, X=Hist[1][:-1]))
        if pltflag == 1:
            plt.plot(Hist[1][:-1], S, label=lbl[n])
        #print(Hist[0])
        ctr = 0
        OutArr.append(S)
        if writeflag == 1:
            f = open(flname, 'a')
            for i in range(len(S)):
                f.write("%2d %3d %2.3f %d %f %f\n" %(int(a),int(t),float(e), NB[n], Hist[1][i], S[i]))

    if pltflag == 1:
        for v in range(len(vertval)):
            plt.axvline(x=vertval[v], linewidth=1, color='k', linestyle=vertls[v])

        SetupPlot(Title, xlbl, ylbl, grid=True, Block=1-g.S_ANG, saveflag=g.S_ANG, savename=savename, legend=1 )



    return OutArr

def distrEnergy(Temp, act, Bounces, NRG_Array, NormArr, bins, xlbl='', ylbl='', ax=[], saveflag=0, svname='', pltflag=0, Title='', figname=''):
    # Plot Energy distributions
    # different energy components are defined by different actions (act), e.g. energy loss == act=0
    # for kinetic components a 2D boltzmann distribution will be fitted (act=1 & 2)
    linestyle = ['-', '-.', '--', ':', '-']
    linecolor = ['black', 'black', 'black', 'black', 'grey']
    plt.cla()
    plt.clf()
    # plt.title(Title)
    plt.axis(ax)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
     ##### fit function

    def boltz2(x,t):
        return (10/t)*np.exp(-x/t)
    def boltz3(x,t,a):
        return a * np.sqrt(x) * t**(-1.5) * np.exp(-x/t)

    for i,n in enumerate(Bounces):
        x = NRG_Array[act,n,0,:bins]
        y = NRG_Array[act,n,1,:bins]/NormArr[n]
        N = n+1
        conv = g.kB*1e3/g.e0
        if i < len(Bounces) - 1:
            plt.plot(x, y, ls=linestyle[i], color=linecolor[i], label=str(N))
        else:
            plt.fill_between(x, y, linestyle='-', color='r', alpha=0.35, label=str(N))
            if act == 1:
                gmodel = Model(boltz2)
                yfit = y[:bins]
                xfit = x[:bins]
                if Temp > 120:
                    result = gmodel.fit(yfit, t=Temp * conv, x=xfit)
                    fitparam = result.params['t'].value
                    plt.plot(x, boltz2(x,fitparam), 'b--', label=r'$P_B(E_{||}, T=%3d$ K)' % int(fitparam/conv))
                else:
                    plt.plot(x, boltz2(x,Temp*conv), 'b--', label=r'$P_B(E_{||}, T=%3d$ K)' % int(Temp))
            elif act == 2:
                gmodel = Model(boltz2)
                yfit = y[:bins]
                xfit = x[:bins]
                if Temp > 120:
                    result = gmodel.fit(yfit, t=Temp * conv, x=xfit)
                    fitparam = result.params['t'].value
                    plt.plot(x, boltz2(x,fitparam), 'b--', label=r'$P_B(E_{\bot}, T=%3d$ K)' % int(fitparam/conv))
                else:
                    plt.plot(x, boltz2(x,Temp*conv), 'b--', label=r'$P_B(E_{\bot}, T=%3d$ K)' % int(Temp))
    plt.legend()
    plt.tight_layout()
    if saveflag != 0:
        plt.savefig(svname)
        plt.show(block=False)
        plt.close()
    if pltflag == 1 and saveflag == 0:
        plt.show(block=pltflag)
        plt.close()

def ChooseParams(Temp, angle):
    # caution! all here is hardcoded
    # axes describe the different domains for the energy distributions/histograms to be calculated
    # there is one such list for each bounce for a given temperature
    #
    # then there are certain bounces at which we would like to plot the corresponding distribution
    # these values may depend on the given angle
    #
    # The corresponding arrays to define the axes and Bounces are returned
    if Temp == 80:
        axes =        [[-100,60,0,0.6], [0,60,0,1], [0,130,0,1], [-110, -85, 0, 2], [-120,70,0,0.4], [0,120, 0,0.5]]
        if int(angle) == 0:
                        #delta E        para E      norm E         pot E         tot E          kin E
            Bounces = [[0,1,4,9,26], [0,1,4,9,23], [0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,16], [0,1,4,9,22]]
        elif int(angle) == 30:
            Bounces = [[0,1,4,9,26], [0,1,4,9,23], [0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,17], [0,1,4,9,22]]
        elif int(angle) == 45:
            Bounces = [[0,1,4,9,26], [0,1,4,9,27], [0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,23], [0,1,4,9,22]]
        elif int(angle) == 60:
            Bounces = [[0,1,4,9,26], [0,1,4,9,17,26], [0,1,4,9,17,25], [0,1,4,9,24], [0,1,4,9,25], [0,1,4,9,22]]
    elif Temp == 190: #[0,70,0,1], [0,140,0,0.6]
        axes = [[-100,60,0,0.4], [0,70,0,1], [0,140,0,0.6], [-110, -80, 0, 1.5], [-120,80,0,0.3], [0,120, 0,0.3]]
        if int(angle) == 0:
            Bounces = [[0,1,4,9,25], [0,1,4,9,21], [0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,16], [0,1,4,9,18]]
        elif int(angle) == 30:
            Bounces = [[0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,16], [0,1,4,9,18]]
        elif int(angle) == 45:
            Bounces = [[0,1,4,9,25], [0,1,4,9,23], [0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,16], [0,1,4,9,18]]
        elif int(angle) == 60:
            Bounces = [[0,1,4,9,25], [0,1,4,9,23], [0,1,4,9,25], [0,1,4,9,24], [0,1,4,9,16], [0,1,4,9,18]]
    elif Temp == 300:
        axes = [[-100,60,0,0.3], [0,60,0,0.6], [0,130,0,0.6], [-110, -80, 0, 1.25], [-120,100,0,0.2], [0,120, 0,0.25]]
        if int(angle) == 0:
            Bounces = [[0,1,4,9,24], [0,1,4,9,20], [0,1,4,9,24], [0,1,4,9,17], [0,1,4,9,16], [0,1,4,9,16]]
        elif int(angle) == 30:
            Bounces = [[0,1,4,9,24], [0,1,4,9,20], [0,1,4,9,24], [0,1,4,9,17], [0,1,4,9,16], [0,1,4,9,16]]
        elif int(angle) == 45:
            Bounces = [[0,1,4,9,24], [0,1,4,9,22], [0,1,4,9,24], [0,1,4,9,17], [0,1,4,9,16], [0,1,4,9,16]]
        elif int(angle) == 60:
            Bounces = [[0,1,4,9,24], [0,1,4,9,20], [0,1,4,9,24], [0,1,4,9,17], [0,1,4,9,16], [0,1,4,9,16]]
    return Bounces, axes

'''
    *********************************************
    Plot functions for flux simulations
    *********************************************
'''

# def WritePlot(X=[], Y=[], name='', xlabel='', ylabel='', saveflag=True, header=True, action='w'):
#     if saveflag == False:
#         return -1
#     name = name + '.csv'
#     f = open(name, action)
#     f.write('#%s, %s\n' % (xlabel, ylabel))
#
#     aux = 0
#     try:
#         aux = len(Y[0])
#     except:
#         pass
#
#     if aux > 0:
#         for i in range(len(Y[0])):
#             x = X[i]
#             f.write('%f' % x)
#             for j in range(len(Y)):
#                 f.write(', %f' % (Y[j][i]))
#             f.write('\n')
#     else:
#         for i,y in enumerate(Y):
#             x = X[i]
#             f.write('%f, %f\n' % (x,y))
#
#     f.close()
#     return 0

def MakePlot(saveflag=False, block=False, xlabel='', ylabel='', savepath=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if saveflag == True:
        plt.savefig(savepath + '.pdf')
        # print("saved figure")
        # plt.savefig(savepath + parameter_set_str + "KinEnHeight.pdf")
    plt.show(block=block)
    plt.clf()
    plt.cla()



def PlotDensityHistogram(X=[], Y=[], Label=[], block=False, NumOfTraj=1, xlabel='', ylabel='', saveflag=False, savedir='', writeflag=False):
    area = 1557e-20
    # values in histogram should already have units of 1 / z
    # as values are interpreted as particles per bin
    if writeflag == True:
        WritePlot(X=X[0], Y=Y[0], name=savedir, xlabel=xlabel, ylabel=ylabel, saveflag=writeflag, header=True, action='w')
    for i in range(len(X)):
        plt.plot(X[i], Y[i], label=Label[i])


    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir)

def Scaling(Arr, conversion):
    for i in range(len(Arr)):
        Arr[i] *= conversion
    return Arr

# from complete dataframe (which contains all the simulation data)
# create the density time evolution for bound particles as well as gas particles
def createDensityOverTime(df, NumOfTraj=40, MaxStep=100000):
    print("Compute Density Time Evolution")
    ###### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27
    # TODO Caution! Hard coded
    mass = 40.
    m = mass * au
    area = 1557e-20
    Bound = 5.0 # Angström #TODO

    BoundedParticles = df.loc[df['z'] < Bound, ['step', 'vz', 'pe']]
    GasParticles = df.loc[df['z'] >= Bound, ['step', 'vz', 'pe']]

    maxsteps = MaxStep
    TimeArr = np.arange(0,maxsteps,1000)
    TimeArr = TimeArr * 0.25 / 1000.

    particleCount = []
    partCountGas = []

    for i in range(0,maxsteps,1000):
        TimeResolved = BoundedParticles.loc[BoundedParticles['step'] == i, ['vz']]
        TimeResolvedGas = GasParticles.loc[GasParticles['step'] == i, ['vz']]
        # if TimeResolved['vz'].count() <= 0:
        #     continue
        particleCount.append(TimeResolved['vz'].count())
        partCountGas.append(TimeResolvedGas['vz'].count())

    # TODO somehow add a filter, so as not to count directly reflected particles,
    # as those are never considered to be bound anyway

    # smooth.Compress(particleCount)
    # particleCount = sf(particleCount, 77, 3, deriv=0)
    # partCountGas = sf(partCountGas, 77, 3, deriv=0)




    # bound particles
    conversion = 1. / area * (1e-9)**2 # normalization to unit area, which I chose to be 1 nm^2 for bound particles
    particleCount = Scaling(particleCount, conversion / NumOfTraj)
    # Population = Scaling(Population, conversion) # solution data is already for standard case, which cooresponds to one sim run

    # gas particles
    conversion = 1.0 / (area * 55e-10) # normalization to density in m^-3 for gas particles
    conversion_to_nm = conversion * 1e-27 # convert density to units nm^-3
    partCountGas = Scaling(partCountGas, conversion_to_nm / NumOfTraj)

    return particleCount, partCountGas, TimeArr

# the first few arguments correspond to our rate equation model (and md data)
# while Population2 and TimeArr2 are the solution from the statistical rate theory (with ylabel2 as correct label)
# which aims to extend our model in the stationary regime
def PlotDensityOverTime(xlabel='', ylabel='', Population=[], Stationary=[], Slope=[], mdData=[], mdDataGas=[], TimeArr=[],
                        Population2=[], TimeArr2=[], ylabel2='', pressure=0.0,
                        saveflag=False, savedir='', writeflag=False, block=False, t0=0):


    # find time where data is non-zero
    # this corresponds to the time of the first particles reaching the surface
    l = 0
    for i in range(len(mdData)):
        if mdData[i] > 0:
            l = i
            break

    print("earliest non zero value:", l)
    mdData = sf(mdData, 77, 3, deriv=0)
    mdDataGas = sf(mdDataGas, 77, 3, deriv=0)
    # prepare to write all the data
    if writeflag == True:
        WritePlot(X=TimeArr[:-l], Y=mdData[l:], name=savedir, xlabel=xlabel, ylabel=ylabel, header=True, action='w')
        WritePlot(X=TimeArr[:t0], Y=Population[:t0], name=savedir+'Sol', xlabel=xlabel, ylabel=ylabel, header=True, action='w')
        WritePlot(X=TimeArr[:-l], Y=mdDataGas[l:], name=savedir+'Gas', xlabel=xlabel, ylabel=ylabel2, header=True, action='w')
        WritePlot(X=TimeArr2, Y=Population2, name=savedir+'SRT', xlabel=xlabel, ylabel=ylabel, header=True, action='w')


    # if desired plot first the bound particle densities
    plt.plot(TimeArr[:-l], mdData[l:], 'k', label=str('MD Data'))
    if len(Population) != 0:
        plt.plot(TimeArr[:t0], Population[:t0], 'r', label="MD-RE")
    if len(Population2) != 0:
        plt.plot(TimeArr2, Population2, 'r:', label="SRT")
    if len(Slope) != 0:
        pass
        # Slope = Scaling(Slope, conversion)
        # shift = 100
        # plt.plot(TimeArr[shift:int(len(Slope))+shift], Slope[:int(len(Slope))], '--', color='blue', label="Slope Data")
    if len(Stationary) != 0:
        plt.plot(TimeArr, Stationary, 'r:', label="Stationary Solution")

    # find maximum value for y axis
    maxDens = 0.0
    if(np.max(mdData) > np.max(Population)):
        maxDens = np.max(mdData)
    else:
        maxDens = np.max(Population)
    if(np.max(Population2) > maxDens):
        maxDens = np.max(Population2)

    plt.axis((-10, TimeArr[-l], -0.005, maxDens+0.05))
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir)

    # afterwards plot the gas particle population (or rather its density)
    plt.plot(TimeArr[:-l], mdDataGas[l:], label=r'$N_g$')
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel2, savepath=savedir+'Gas')


def PlotKineticEnergyOverHeight(df, block=False, xlabel='', ylabel='', MaxStep=100000, saveflag=False, savedir='', savename='', writeflag=False):
    print("Compute Kinetic Energy Profile")
    ###### Constants
    # redundant redefinition
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27
    step = MaxStep
    mass = 40
    m = mass * au
    Bound = 5.0
    MaxHeight = df['z'].max()

    stepsize = 0.5
    HeightArr = np.arange(0,MaxHeight-2.,stepsize)
    xKE_In = []
    xKE_Out = []
    yKE_In = []
    yKE_Out = []
    zKE_In = []
    zKE_Out = []
    AvgWindow = 1000000
    lengthArray = len(HeightArr)

    for h in HeightArr:
        VelocityArrIn = df.loc[(df['z'] > h) & (df['z'] <= h+stepsize) & (df['traj'] < 20) &
                                        (df['step'] >= MaxStep-AvgWindow) & (df['vz'] <= 0),
                                        ['vx', 'vy', 'vz']]
        VelocityArrIn['xke'] = 0.5 * m * (VelocityArrIn['vx'] * 100.) ** 2 / kB
        VelocityArrIn['yke'] = 0.5 * m * (VelocityArrIn['vy'] * 100.) ** 2 / kB
        VelocityArrIn['zke'] = 0.5 * m * (VelocityArrIn['vz'] * 100.) ** 2 / kB

        VelocityArrOut = df.loc[(df['z'] > h) & (df['z'] <= h+stepsize) & (df['traj'] < 20) &
                                        (df['step'] >= MaxStep-AvgWindow) & (df['vz'] > 0),
                                        ['vx', 'vy', 'vz']]
        VelocityArrOut['xke'] = 0.5 * m * (VelocityArrOut['vx'] * 100.) ** 2 / kB
        VelocityArrOut['yke'] = 0.5 * m * (VelocityArrOut['vy'] * 100.) ** 2 / kB
        VelocityArrOut['zke'] = 0.5 * m * (VelocityArrOut['vz'] * 100.) ** 2 / kB

        xKE_In.append(VelocityArrIn['xke'].mean())
        xKE_Out.append(VelocityArrOut['xke'].mean())
        yKE_In.append(VelocityArrIn['yke'].mean())
        yKE_Out.append(VelocityArrOut['yke'].mean())
        zKE_In.append(VelocityArrIn['zke'].mean())
        zKE_Out.append(VelocityArrOut['zke'].mean())

    from stats import median
    xKEmean = 0.5 * (median(xKE_In[lengthArray//2:]) + median(xKE_Out[lengthArray//2:]))
    yKEmean = 0.5 * (median(yKE_In[lengthArray//2:]) + median(yKE_Out[lengthArray//2:]))
    zKEmean = 0.5 * (median(zKE_In[lengthArray//2:]) + median(zKE_Out[lengthArray//2:]))
    print("KEmean",(xKEmean + yKEmean + zKEmean) / 3.0)



    if writeflag == True:
        WritePlot(X=HeightArr, Y=[xKE_In, yKE_In, zKE_In], name=savedir+savename+'In', xlabel=xlabel, ylabel=ylabel+' x,y,z', header=True, action='w')
        WritePlot(X=HeightArr, Y=[xKE_Out, yKE_Out, zKE_Out], name=savedir+savename+'Out', xlabel=xlabel, ylabel=ylabel+' x,y,z', header=True, action='w')
    plt.plot(HeightArr, [xKE_In[i] + yKE_In[i] + zKE_In[i] for i in range(len(xKE_In))], label='Kin Energy In')
    plt.plot(HeightArr, [xKE_Out[i] + yKE_Out[i] + zKE_Out[i] for i in range(len(xKE_Out))], label='Kin Energy Out')
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+savename)


def PlotPotentialEnergyOverHeight(df, block=False, xlabel='', ylabel='', MaxStep=100000, saveflag=False, savedir='', savename='', writeflag=False):
    print("Compute Potential Energy Profile")
    ###### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27
    step = MaxStep
    mass = 40
    m = mass * au
    Bound = 5.0
    MaxHeight = float(df['z'].max())
    AvgWindow = 600000

    delta_h = 0.2
    HeightArr = np.arange(0,MaxHeight-2.,delta_h)
    PE = []
    for h in HeightArr:
        Array = df.loc[(df['z'] > h) & (df['z'] <= h+delta_h) & (df['traj'] < 20) & (df['step'] >= MaxStep-AvgWindow), ['pe']]
        PE.append(Array['pe'].mean())


    if writeflag == True:
        WritePlot(X=HeightArr, Y=PE, name=savedir+savename, xlabel=xlabel, ylabel=ylabel, header=True, action='w')
    plt.plot(HeightArr, PE, label='Pot Energy')
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+savename)
    return PE

def PlotPotentialEnergyOverTime(df, block=False, xlabel='', ylabel='', MaxStep=100000, saveflag=False, savedir='', savename='', writeflag=False):
    print("Compute Potential Energy Evolution for Bound Particles")

    Bound = 5.0
    Intermediate = 10.0
    stw = 1000
    PE = []
    PEstd = []
    Time = np.arange(0,MaxStep,stw)
    for t in Time:
        Array = df.loc[(df['step'] == t) & (df['z'] <= Intermediate), ['pe']]
        PE.append(Array['pe'].mean())
        PEstd.append(Array['pe'].std() / (Array['pe'].count()-1))

    if writeflag == True:
        WritePlot(X=Time*0.00025, Y=PE, name=savedir+savename, xlabel=xlabel, ylabel=ylabel, header=True, action='w')
        WritePlot(X=Time*0.00025, Y=PEstd, name=savedir+savename+'std', xlabel=xlabel, ylabel=ylabel+'std', header=True, action='w')
    plt.errorbar(Time*0.00025, PE, yerr=PEstd, label='Pot Energy')
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+savename)
    return PE, PEstd

def PlotKineticEnergyOverTime(df, block=False, xlabel='', ylabel='', MaxStep=100000,
    saveflag=False, savedir='', savename='', writeflag=False, temp_S=-1.0, temp_P=-1.0):
    print("Plot Kinetic Energy Time Evolution")
    ###### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27
    step = MaxStep
    mass = 40
    m = mass * au
    Bound = 5.0
    xKE_Bound = []
    yKE_Bound = []
    zKE_Bound = []
    xKE_Gas = []
    yKE_Gas = []
    zKE_Gas = []
    TimeArr = np.arange(0,MaxStep,10000)
    for t in TimeArr:
        VelocityBound = df.loc[(df['z'] <= Bound) & (df['traj'] < 20) & (df['step'] == t), ['vx', 'vy', 'vz', 'step']]
        VelocityBound['xke'] = 0.5 * m * (VelocityBound['vx'] * 100.) ** 2 / kB
        VelocityBound['yke'] = 0.5 * m * (VelocityBound['vy'] * 100.) ** 2 / kB
        VelocityBound['zke'] = 0.5 * m * (VelocityBound['vz'] * 100.) ** 2 / kB


        VelocityGas = df.loc[(df['z'] > Bound) & (df['traj'] < 20) & (df['step'] == t), ['vx', 'vy', 'vz', 'step']]
        VelocityGas['xke'] = 0.5 * m * (VelocityGas['vx'] * 100.) ** 2 / kB
        VelocityGas['yke'] = 0.5 * m * (VelocityGas['vy'] * 100.) ** 2 / kB
        VelocityGas['zke'] = 0.5 * m * (VelocityGas['vz'] * 100.) ** 2 / kB

        if (VelocityGas['xke'].count() > 0):
            xKE_Gas.append(VelocityGas['xke'].mean())
            yKE_Gas.append(VelocityGas['yke'].mean())
            zKE_Gas.append(VelocityGas['zke'].mean())
        else:
            xKE_Gas.append(0.0)
            yKE_Gas.append(0.0)
            zKE_Gas.append(0.0)
        if (VelocityBound['xke'].count() > 0):
            xKE_Bound.append(VelocityBound['xke'].mean())
            yKE_Bound.append(VelocityBound['yke'].mean())
            zKE_Bound.append(VelocityBound['zke'].mean())
        else:
            xKE_Bound.append(0.0)
            yKE_Bound.append(0.0)
            zKE_Bound.append(0.0)
    # print(KinEnergyBound.describe())
    # return 0
    # KE_Bound = sf(KE_Bound, 11, 3, deriv=0)
    # KE_Gas = sf(KE_Gas, 11, 3, deriv=0)
    # if temp_S > 0:
    #     SurfaceTemp = [temp_S * 2. for i in TimeArr]
    #     plt.plot(TimeArr*0.00025, SurfaceTemp, 'k:')
    # if temp_P > 0:
    #     PlasmaTemp = [temp_P for i in TimeArr]
    #     plt.plot(TimeArr*0.00025, PlasmaTemp, 'r:')
    if writeflag == True:
        WritePlot(X=TimeArr*0.00025, Y=[xKE_Bound, yKE_Bound, zKE_Bound], name=savedir+savename+'Bound', xlabel=xlabel, ylabel=ylabel, header=True, action='w')
        WritePlot(X=TimeArr*0.00025, Y=[xKE_Gas, yKE_Gas, zKE_Gas], name=savedir+savename+'gas', xlabel=xlabel, ylabel=ylabel, header=True, action='w')
    plt.plot(TimeArr*0.00025, [xKE_Bound[i] + yKE_Bound[i] + zKE_Bound[i] for i in range(len(xKE_Bound))], label='Bound Kin Energy')
    plt.plot(TimeArr*0.00025, [xKE_Gas[i] + yKE_Gas[i] + zKE_Gas[i] for i in range(len(xKE_Gas))], label='Gas Kin Energy ')

    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+savename)

def getCoverage(df, maxStep, timeRange, trajnum):
    bound = 5.0
    stw = 1000
    count = 0.0
    countArray = []
    for traj in range(trajnum):
        for t in range(maxStep-timeRange, maxStep, stw*2):
            currentVal = df.loc[(df['step'] == t) & (df['traj'] == traj) & (df['z'] < bound), ['vz']]
            count += currentVal['vz'].count()

    count /= (timeRange / 1000) * trajnum
    return count



def PlotCoverage(df, angle, temp_S, temp_P, pressure, block=False, MaxStep=700000, xlabel='', ylabel='', saveflag=False, savedir='', savename='', writeflag=False):
    print("Plot Surface Coverage")

    # create fcc lattice for comparison
    latticeCoord = [[],[]]
    delta_x = 6.66261 - 1.66565
    delta_y = 2.885
    for x in range(9):
        # 2 lattice base vectors for fcc
        base1 = [1.6656, 0.0]
        base2 = [4.16413, 1.4425]
        base1[0] += x * delta_x
        base2[0] += x * delta_x
        for y in range(12):
            latticeCoord[0].append(base1[0])
            latticeCoord[0].append(base2[0])
            latticeCoord[1].append(base1[1])
            latticeCoord[1].append(base2[1])
            base1[1] += delta_y
            base2[1] += delta_y

    radiusAu = 4.08 * 1.5 # Angstöm
    surfaceAu = math.pi * radiusAu**2

    step = MaxStep
    traj = np.random.randint(0,20)
    coverage = 0.0

    for traj in range(0,20):
        assert(step % 1000 == 0)
        Boundaries = [3.0, 5.0, 8.2, 11.4, 15.0, 20.0]
        Locations = df.loc[(df['step'] ==  step) & (df['traj'] == traj), ['x','y','z','vz']]
        FirstLayer = Locations.loc[(Boundaries[0] < Locations['z']) & (Locations['z'] <= Boundaries[1]), ['x','y','vz']]
        coverage += FirstLayer['x'].count()
    coverage /= 20.

    # print("Coverage:", coverage)

    SecondLayer = Locations.loc[(Boundaries[1] < Locations['z']) & (Locations['z'] <= Boundaries[2]), ['x','y','vz']]
    ThirdLayer = Locations.loc[(Boundaries[2] < Locations['z']) & (Locations['z'] <= Boundaries[3]), ['x','y','vz']]
    HighestLayer = Locations.loc[(Boundaries[4] < Locations['z']) & (Locations['z'] <= Boundaries[5]), ['x','y','vz']]


    sc = plt.scatter(FirstLayer['x'], FirstLayer['y'], label='First Layer', c=FirstLayer['vz'])
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$v_z$ / $10^{2}$ $\frac{m}{s}$', rotation=270, labelpad=20)
    plt.scatter(latticeCoord[0], latticeCoord[1], alpha=0.2, color='grey', s=surfaceAu)
    name = savename + "FirstLayer"
    WritePlot(X=FirstLayer['x'].values, Y=FirstLayer['y'].values, name=savedir+name, xlabel=xlabel, ylabel=ylabel, header=True, action='w', saveflag=writeflag)
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)

    sc = plt.scatter(SecondLayer['x'], SecondLayer['y'], label='Second Layer', c=SecondLayer['vz'])
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$v_z$ / $10^{2}$ $\frac{m}{s}$', rotation=270, labelpad=20)
    plt.scatter(FirstLayer['x'], FirstLayer['y'], label='First Layer', alpha=0.5, color='red')
    plt.scatter(latticeCoord[0], latticeCoord[1], alpha=0.2, color='grey', s=surfaceAu)
    name = savename + "SecondLayer"
    WritePlot(X=SecondLayer['x'].values, Y=SecondLayer['y'].values, name=savedir+name, xlabel=xlabel, ylabel=ylabel, header=True, action='w', saveflag=writeflag)
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)

    sc = plt.scatter(ThirdLayer['x'], ThirdLayer['y'], label='Third Layer', c=ThirdLayer['vz'])
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$v_z$ / $10^{2}$ $\frac{m}{s}$', rotation=270, labelpad=20)
    plt.scatter(SecondLayer['x'], SecondLayer['y'], label='Second Layer', alpha=0.5, color='red')
    plt.scatter(latticeCoord[0], latticeCoord[1], alpha=0.2, color='grey', s=surfaceAu)
    name = savename + "ThirdLayer"
    WritePlot(X=ThirdLayer['x'].values, Y=ThirdLayer['y'].values, name=savedir+name, xlabel=xlabel, ylabel=ylabel, header=True, action='w', saveflag=writeflag)
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)

    sc = plt.scatter(HighestLayer['x'], HighestLayer['y'], label='Highest Layer', c=HighestLayer['vz'])
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$v_z$ / $10^{2}$ $\frac{m}{s}$', rotation=270, labelpad=20)
    plt.scatter(latticeCoord[0], latticeCoord[1], alpha=0.2, color='grey', s=surfaceAu)
    name = savename + "HighestLayer"
    WritePlot(X=HighestLayer['x'].values, Y=HighestLayer['y'].values, name=savedir+name, xlabel=xlabel, ylabel=ylabel, header=True, action='w', saveflag=writeflag)
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)

    return coverage

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
        svname = '/home/becker/lammps/newplot/Pop/' + name + 'Population.pdf'
        SetupPlot(Title, 'Time / ps', 'Population Fraction', saveflag=g.S_POP, savename=svname, Block=1-g.S_POP)

def TransitionPopulations(angle, temp, energy, X, QT, TQ, CT, CQ, TC=[], QC=[], Title='', smflag=1, pltflag=1, nu=49, hlrn=0, numpltflag=0):
    TIMESTEPS = len(QT)
    l = len(QT)
    DT = 1
    fig, ax = plt.subplots(num='a'+angle+'t'+temp+'e'+energy)
    if hlrn == 1:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]+'HLRN'
    else:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]
    if numpltflag == 1:
        ax.plot(X*2.5e-2, QT, 'b--', label = 'QT')
        ax.plot(X*2.5e-2, TQ, 'C1--', label = 'TQ')
        ax.plot(X*2.5e-2, CT, 'g--', label = 'CT')
        ax.plot(X*2.5e-2, CQ, 'r--', label = 'CQ')
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
        ax.plot(X*2.5e-2, QT, 'g--', label = r'd$N_{QT}$')
        ax.plot(X*2.5e-2, TQ, 'b-.', label = r'd$N_{TQ}$')
        #ax.plot(X*2.5e-2, CT, 'r--', label = 'CT')
        ax.plot(X*2.5e-2, CQ, 'r-', label = r'd$N_{CQ}$')
        try:
            ax.plot(X*2.5e-2, TC, '-', label = 'TC')
            ax.plot(X*2.5e-2, QC, '-', label = 'QC')
        except:
            dummy=1

        legend = ax.legend()

        svname = '/home/becker/lammps/newplot/Trans/' + name + 'Transition.pdf'
        SetupPlot(Title, 'Time / ps', 'Transition Populations', saveflag=g.S_TRANS, savename=svname, Block=1-g.S_TRANS)

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
        svname = '/home/becker/lammps/newplot/T/' + name + 'TransitionRate.pdf'
        SetupPlot(Title, 'Time / ps', ylbl, saveflag=g.S_T, savename=svname, Block=1-g.S_T)

    return Ta, Tb, Tc, Td

#TODO
def Histogram(emin, emax, nbin, A, B, C, subplts=0, lblA='', lblB='', lblC='', lbl='', Title='', binarr=[], avg=0, std=0, xlbl='', ylbl='',
 acol='', bcol='', ccol='', totcol='', saveflag=0, savename='', subscale=1,
 totalplt=1, sm_flag=1, pltflag=1, nu=4, edge='none', edgeA='none', edgeB='none', edgeC='none', writeflag=0, flname='', nb=0, action=-1, scl=1, vertval=[], vertls=[]):
    # find a better way to create bins, so that 0 meV is always a bin boundary
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


def EnergyDistribution():
    return 0

def Solution(angle,temp,energy,N, Time, TimePrime, StateComp, Title='', pltflag=0, maxtime=60):
    if g.HLRN == 1:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]+'HLRN'
    else:
        name = 'a'+angle+'t'+temp+'e'+energy.split('.')[0]
    svname = '/home/becker/lammps/newplot/Sol/' + name + 'Solution.pdf'
    if pltflag == 1:
        plt.cla()
        plt.clf()
        plt.axis([0, maxtime, 0, max(StateComp[0])+0.05])
        plt.plot(Time*0.025, N[0], 'b', label=r'$N_{T,prediction}$', alpha=0.5)
        plt.plot(TimePrime*0.025, StateComp[0], 'b-.', label=r'$N_{T,num}$')
        plt.plot(Time*0.025, N[1], 'g', label=r'$N_{Q,prediction}$', alpha=0.5)
        plt.plot(TimePrime*0.025, StateComp[1], 'g--', label=r'$N_{Q,num}$')
        SetupPlot(Title, 't / ps', 'Population Fraction', grid=False, Block=1-g.S_SOL, saveflag=g.S_SOL, savename=svname, sz=12, legend=1)


###### mainy for angular distribution
def Distr(emin, emax, nbin, Arr, subplts=0, lbl=[], Title='', binarr=[], avg=0, std=0, xlbl='', ylbl='',
 saveflag=0, savename='', subscale=1, nbb=8, numTraj=2000, a=0, t=0, e=0, NB=[],
 totalplt=1, sm_flag=1, pltflag=1, nu=4, edge='pos', writeflag=0, flname='', nb=0, action=-1, scl=1, vertval=[], vertls=[]):
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

###### Flux plot func

def MakePlot(saveflag=False, block=False, xlabel='', ylabel='', savepath=''):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    if saveflag == True:
        plt.savefig(savepath)
        # print("saved figure")
        # plt.savefig(savepath + parameter_set_str + "KinEnHeight.pdf")
    plt.show(block=block)
    plt.clf()
    plt.cla()

def PlotDensityHistogram(X=[], Y=[], Label=[], block=False, NumOfTraj=1, xlabel='', ylabel='', saveflag=False, savedir=''):
    home = str(Path.home())
    savedir = home + "/lammps/flux/plot/"
    area = 1557e-20
    for i in range(len(X)):
        plt.plot(X[i], Y[i]/NumOfTraj/area*(1e-9)**2, label=Label[i])
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir)

def PlotDensityOverTime(df, block=False, NumOfTraj=40, MaxStep=100000, xlabel='', ylabel='', Population=[], saveflag=False, savedir=''):
    print("Compute Density Time Evolution")
    ###### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27
    mass = 40
    m = mass * au
    area = 1557e-20
    Bound = 5.0 # Angström #TODO

    BoundedParticles = df.loc[df['z'] < Bound, ['step', 'vz', 'pe']]
    # BoundedParticles = df.loc[df['z'] < Bound, ['vz', 'pe', 'step']]
    # BoundedParticles['zKE'] = 0.5 * m * (BoundedParticles['vz'] * 100.) ** 2 / e0
    # BoundedParticles = BoundedParticles.loc[(BoundedParticles['pe'] + BoundedParticles['zKE'] < 0), ['vz', 'step']]


    maxsteps = MaxStep
    TimeArr = np.arange(0,maxsteps,1000)
    TimeArr = TimeArr * 0.25 / 1000.
    # 1000 steps == 0.25*1000 fs == 0.25 ps

    particleCount = []
    for i in range(0,maxsteps,1000):
        TimeResolved = BoundedParticles.loc[BoundedParticles['step'] == i, ['vz']]
        # TimeResolved = TrappedParticles.loc[TrappedParticles['step'] == i, ['vz', 'pe']]
        particleCount.append(TimeResolved['vz'].count())

    # TODO somehow add a filter, so as not to count directly reflected particles,
    # as those are never considered to be bound
    # smooth.Compress(particleCount)
    particleCount = sf(particleCount, 77, 3, deriv=0)
    # for k in range(0, len(particleCount)):
    #     particleCount[k] = smooth.GaussSmoothing(len(particleCount), k, particleCount, dt=1, nu=7)

    particleCount = particleCount / NumOfTraj

    particleCount = particleCount / area * (1e-9)**2 # normalization to unit area, which I chose to be 1 nm^2

    # TODO Caution! Hard coded
    pressure = 1.0
    temp_P = 300

    plt.plot(TimeArr, particleCount, label='MD data')
    if len(Population) != 0:
        plt.plot(TimeArr, Population / area * (1e-9)**2, label="Analytical Solution")
    # xlabel = 'time / ps'
    # ylabel = 'Number of bound particles'
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir)

def PlotKineticEnergyOverHeight(df, block=False, xlabel='', ylabel='', MaxStep=100000, saveflag=False, savedir=''):
    print("Compute Kinetic Energy Profile")
    ###### Constants
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
    KE = []
    for h in HeightArr:
        VelocityArr = df.loc[(df['z'] > h) & (df['z'] <= h+stepsize) & (df['traj'] < 2) & (df['step'] >= MaxStep-10000), ['vx', 'vy', 'vz']]
        VelocityArr['ke'] = 0.5 * m * ( (VelocityArr['vx'] * 100.) ** 2
                                        + (VelocityArr['vy'] * 100.) ** 2
                                        + (VelocityArr['vz'] * 100.) ** 2) / kB
        KE.append(VelocityArr['ke'].mean())

    plt.plot(HeightArr, KE, label='Kin Energy')

    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir)


def PlotKineticEnergyOverTime(df, block=False, xlabel='', ylabel='', MaxStep=100000, saveflag=False, savedir=''):
    print("Plot Kinetic Energy Time Evolution")
    ###### Constants
    kB = 1.3806503e-23
    e0 = 1.60217662e-19
    au = 1.66053904e-27
    step = MaxStep
    mass = 40
    m = mass * au
    Bound = 5.0
    KE_Bound = []
    KE_Gas = []
    TimeArr = np.arange(0,MaxStep,5000)
    for t in TimeArr:
        VelocityBound = df.loc[(df['z'] <= Bound) & (df['traj'] < 2) & (df['step'] == t), ['vx', 'vy', 'vz', 'step']]
        VelocityBound['ke'] = 0.5 * m * ( (VelocityBound['vx'] * 100.) ** 2
                                        + (VelocityBound['vy'] * 100.) ** 2
                                        + (VelocityBound['vz'] * 100.) ** 2) / kB
        VelocityGas = df.loc[(df['z'] > Bound) & (df['traj'] < 2) & (df['step'] == t), ['vx', 'vy', 'vz', 'step']]
        VelocityGas['ke'] = 0.5 * m * ( (VelocityGas['vx'] * 100.) ** 2
                                        + (VelocityGas['vy'] * 100.) ** 2
                                        + (VelocityGas['vz'] * 100.) ** 2) / kB

        KE_Gas.append(VelocityGas['ke'].mean())
        KE_Bound.append(VelocityBound['ke'].mean())

    # print(KinEnergyBound.describe())
    # return 0
    # KE_Bound = sf(KE_Bound, 11, 3, deriv=0)
    # KE_Gas = sf(KE_Gas, 11, 3, deriv=0)
    plt.plot(TimeArr*0.00025, KE_Bound, label='Bound Kin Energy')
    plt.plot(TimeArr*0.00025, KE_Gas, label='Gas Kin Energy ')
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir)

def PlotCoverage(df, block=False, MaxStep=700000, xlabel='', ylabel='', saveflag=False, savedir='', parameter_set_str=''):
    print("Plot Surface Coverage")
    step = MaxStep
    traj = 1
    assert(step % 1000 == 0)
    Boundaries = [3.0, 5.0, 8.2, 11.4, 15.0, 20.0]
    Locations = df.loc[(df['step'] ==  step) & (df['traj'] == traj), ['x','y','z']]
    FirstLayer = Locations.loc[(Boundaries[0] < Locations['z']) & (Locations['z'] <= Boundaries[1]), ['x','y']]
    SecondLayer = Locations.loc[(Boundaries[1] < Locations['z']) & (Locations['z'] <= Boundaries[2]), ['x','y']]
    ThirdLayer = Locations.loc[(Boundaries[2] < Locations['z']) & (Locations['z'] <= Boundaries[3]), ['x','y']]
    HighestLayer = Locations.loc[(Boundaries[4] < Locations['z']) & (Locations['z'] <= Boundaries[5]), ['x','y']]
    plt.scatter(FirstLayer['x'], FirstLayer['y'], label='First Layer')

    name = parameter_set_str + "FirstLayer.pdf"
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)
    plt.scatter(SecondLayer['x'], SecondLayer['y'], label='Second Layer')
    name = parameter_set_str + "SecondLayer.pdf"
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)
    plt.scatter(ThirdLayer['x'], ThirdLayer['y'], label='Third Layer')
    name = parameter_set_str + "ThirdLayer.pdf"
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)
    plt.scatter(HighestLayer['x'], HighestLayer['y'], label='Highest Layer')
    name = parameter_set_str + "HighestLayer.pdf"
    MakePlot(saveflag=saveflag, block=block, xlabel=xlabel, ylabel=ylabel, savepath=savedir+name)

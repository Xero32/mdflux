import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lmfit import Model

def within(y, x, dx):
    if (x-dx < y and y <= x+dx):
        return 1
    else:
        return 0

def mapParams(angle, temp_S, temp_P, pressure, niaFlag):
    # map param space to a hex string/number
    # where each parameter is unambiguously encoded by a hex digit
    param = "0x"

    if within(angle, 0, 1):
        param += "0"
    elif within(angle, 15,1):
        param += "1"
    elif within(angle, 30,1):
        param += "2"
    elif within(angle, 45,1):
        param += "3"
    elif within(angle, 60,1):
        param += "4"
    else:
        param += "F"

    if within(temp_S, 80, 1):
        param += "0"
    elif within(temp_S, 190, 1):
        param += "1"
    elif within(temp_S, 300, 1):
        param += "2"
    else:
        param += "F"

    if within(temp_P, 80, 1):
        param += "0"
    elif within(temp_P, 190, 1):
        param += "1"
    elif within(temp_P, 300, 1):
        param += "2"
    else:
        param += "F"

    if within(pressure, 0.5, 0.01):
        param += "0"
    elif within(pressure, 1.0, 0.01):
        param += "1"
    elif within(pressure, 2.0, 0.01):
        param += "2"
    elif within(pressure, 4.0, 0.01):
        param += "3"
    elif within(pressure, 8.0, 0.01):
        param += "4"
    elif within(pressure, 12.0, 0.01):
        param += "5"
    elif within(pressure, 16.0, 0.01):
        param += "6"
    elif within(pressure, 20.0, 0.01):
        param += "7"
    else:
        param += "F"

    if niaFlag:
        param += "1"
    else:
        param += "0"

    return param

def decodeParams(input):
    for i,x in enumerate(input):
        if i < 2:
            continue

        if i == 2:
            if x == "0":
                angle = 0
            elif x =="1":
                angle = 15
            elif x =="2":
                angle = 30
            elif x =="3":
                angle = 45
            elif x =="4":
                angle = 60
            else:
                print("Angle unspecified!")
                return -1,-1,-1,-1, False
        if i == 3:
            if x == "0":
                temp_S = 80
            elif x == "1":
                temp_S = 190
            elif x == "2":
                temp_S = 300
            else:
                print("Surface temperature unspecified!")
                return -1,-1,-1,-1, False
        if i == 4:
            if x == "0":
                temp_P = 80
            elif x == "1":
                temp_P = 190
            elif x == "2":
                temp_P = 300
            else:
                print("Plasma temperature unspecified!")
                return -1,-1,-1,-1, False

        if i == 5:
            if x == "0":
                pressure = 0.5
            elif x == "1":
                pressure = 1.0
            elif x == "2":
                pressure = 2.0
            elif x == "3":
                pressure = 4.0
            elif x == "4":
                pressure = 8.0
            elif x == "5":
                pressure = 12.0
            elif x == "6":
                pressure = 16.0
            elif x == "7":
                pressure = 20.0
            else:
                print("Pressure unspecified!")
                return -1,-1,-1,-1, False

        if i == 6:
            if x == "1":
                niaFlag = True
            else:
                niaFlag = False

    return angle, temp_S, temp_P, pressure, niaFlag




# TODO
# Add information about
#   potential energy (bound)
#   potential energy (bound + precursor state)
#   potential energy (bulk)
#   kinetic energy (bound)
#   kinetic energy (precursor)
#   kinetic energy (bulk)
#   density (bound)
#   density (precursor)
#   density (bulk)
#   pressure (bound)
#   pressure (precursor)
#   pressure (bulk) ### take pressure data from p_zz pressure tensor component

def main():
    home = str(Path.home())
    fname = home + '/lammps/mdThetaData.dat'
    f = open(fname,'r')
    data = []
    for line in f:
        try:
            angle, temp_S, temp_P, pressure, cov, pe = line.split()
            data.append([
                float(angle),
                float(temp_S),
                float(temp_P),
                float(pressure),
                float(cov),
                float(pe)
                ])
        except:
            continue
    f.close()
    path = home + '/lammps/flux/'

    columns = ['angle', 'temp_S', 'temp_P', 'pressure', 'cov', 'pe']
    df = pd.DataFrame(data=data, columns=columns)
    df.describe()
    maxcov = df['cov'].max()
    print(maxcov)
    df['cov'] /= maxcov
    Temp_S = [80., 190., 300.]
    Temp_P = 300.
    Temp_P2 = 190.
    dfList = []
    for i in range(len(Temp_S)):
        dfList.append(df.loc[(df['temp_S'] == Temp_S[i]) & (df['temp_P'] == Temp_P), ['pressure', 'cov', 'pe']])

    theta = np.arange(0.0,1.0,0.01)
    pext = np.arange(0.5,20.0,0.2)

    # plot langmuir isotherms
    for i in range(len(Temp_S)):
        plt.plot(dfList[i]['pressure'], dfList[i]['cov'], label=str(Temp_S[i]) + ' K')

    # fit langmuir isotherm to obtain alpha
    alpha = 0.05
    def pressure_fct(p, alpha):
        return alpha * p / (1. + alpha * p)

    xfit = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    # hardcoded for 30 300 300 p
    yfit = df.loc[(df['temp_S'] == 190) & (df['temp_P'] == Temp_P), ['cov']].values

    gmodel = Model(pressure_fct, independent_vars=['p'], param_names=['alpha'])
    gmodel.set_param_hint('alpha', value=0.05)
    fitparams = gmodel.make_params()
    result = gmodel.fit(yfit, fitparams, p=xfit)
    fitresult = result.params['alpha'].value
    print(result.fit_report())
    plt.plot(xfit, pressure_fct(xfit, fitresult), label="Langmuir Fit, alpha=%f" %(fitresult))

    alpha = 0.3
    plt.plot(pext, pressure_fct(pext, alpha), label='Reference, alpha=%f' %(alpha))
    plt.legend()
    plt.xlabel("p / atm")
    plt.ylabel("theta")
    plt.tight_layout()
    plt.savefig(path + "theta_over_p.pdf")
    plt.show()
    plt.cla()
    plt.clf()
    # 3. compute Langmuir isotherms: theta(p) = alpha*p / (1 + alpha*p)
    # 4. find alpha [maybe alpha(theta)]: alpha(theta) = (theta/p) / (1 + theta)

    # plot pressure graph: p(theta)
    # p(theta) * alpha = theta / (1 - theta)

    for i in range(len(Temp_S)):
        plt.plot(dfList[i]['cov'], dfList[i]['pressure'], label=str(Temp_S[i]) + ' K')

    plt.plot(theta, theta / (alpha *  (1. - theta)), label='reference')
    plt.axis([0,1,0,25])
    plt.legend()
    plt.xlabel("theta")
    plt.ylabel("p / atm")
    plt.tight_layout()
    plt.savefig(path + "p_over_theta.pdf")
    plt.show()
    plt.cla()
    plt.clf()

    for i in range(len(Temp_S)):
        plt.plot(dfList[i]['cov'], dfList[i]['pe'], label=str(Temp_S[i]) + ' K')
    plt.legend()
    plt.xlabel("theta")
    plt.ylabel("pe / eV")
    plt.tight_layout()
    plt.savefig(path + "pe_over_theta.pdf")
    plt.show()
    plt.clf()
    plt.cla()

    for i in range(len(Temp_S)):
        plt.plot(dfList[i]['pressure'], dfList[i]['pe'], label=str(Temp_S[i]) + ' K')
    plt.legend()
    plt.xlabel("p / atm")
    plt.ylabel("pe / eV")
    plt.tight_layout()
    plt.savefig(path + "pe_over_p.pdf")
    plt.show()
    plt.clf()
    plt.cla()

if __name__ == "__main__":
    main()

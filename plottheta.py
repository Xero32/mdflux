import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

home = str(Path.home())
fname = home + '/lammps/param.dat'
f = open(fname,'r')
data = []
for line in f:
    angle, temp_S, temp_P, pressure, cov, pe = line.split()
    data.append([
        float(angle),
        float(temp_S),
        float(temp_P),
        float(pressure),
        float(cov),
        float(pe)
        ])
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
p = np.arange(0.0,20.0,0.2)

# plot langmuir isotherms
for i in range(len(Temp_S)):
    plt.plot(dfList[i]['pressure'], dfList[i]['cov'], label=str(Temp_S[i]) + ' K')

alpha = 0.05 # use later as fit parameter
pressure_fct = alpha * p / (1. + alpha * p)
plt.plot(p, pressure_fct, label='reference')
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

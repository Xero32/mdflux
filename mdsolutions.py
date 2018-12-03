import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
##### Constants
kB = 1.3806503e-23
e0 = 1.60217662e-19
pi = 3.14159265359
au = 1.66053904e-27
atm = 101325
NA = 6.02214086e23
hbar = 1.0545718e-34
R = 8.3144598
###### VdW Constants for Argon
a = 1.355       # l^2 bar /mol^2
a = a * 1e-7 / (NA**2)
b = 0.03201     # l/mol
b = b * 1e-6 / NA
###### system specific constansts
zSurface = 11.8
area = 1557e-20
max_cov = 121.75
substrate_atoms = 216
cov = 1.0 #TODO set coverage
avgHeight = 55. # average height from spawn point to surface

'''
Setup for analytical solution according to paper by Elliot, Ward (1997)
'''

# pressure in bulk phase
# evolution over time
# so far pss, p0 and tau are fit parameters
# but should in principle be extracted from pressure tensor or p = nkT
def pressureExp(t, pss, p0, tau):
    return pss - (pss - p0) * np.exp(-t/tau)

# p: time dependent pressure
# M: adsorption sites on surface
# M0: number of surface atoms in uppermost layer
# covEq: equilibrium coverage
# area: surface area in simulation box
# m: mass of adsorbate particle
# T: plasma temperature
# alpha: fit parameter from langmuir equation
#        can be interpreted as 1/p0
# t0: starting time, t: time array
# theta0: initial value
def integrateTheta(p, M, M0, covEq, area, m, T, alpha, dt, t0, t, theta0):
    if(False):
        area = 1557e-20
        dt = 0.25
        t = np.arange(0.0, 800.0, dt)
        t_0 = 400 # timesteps after which this solution becomes valid
        m = 40.0 * au
        M = 121.
        M0 = 216.
        theta0 = 3.2 / M0
        covEq = 5.5
        T = 300.
        alpha = 0.05
        p = pressureExp(t, 4.38060634, 0.97202180, 109.458062)


    M /= area
    M0 /= area
    theta0 /= area
    sigma = 1.0 / M
    theta_M = M / M0
    theta_E = covEq / M0 / area # maybe?
    v = np.sqrt(2.0 * pi * m * kB * T)
    v_inv = 1.0 / v

    print("theta0 =", theta0)
    print("theta_M =", theta_M)
    print("theta_E =", theta_E)

    assert(len(p) == len(t))
    # theta = np.full((len(t)), 0.0)
    theta = [0.0 for i in range(len(t))]
    theta[t_0] = theta0
    print(theta[t_0])
    for i,_ in enumerate(t):

        if(i > t_0):
            j = i-1
            aux = (theta_M - theta[j]) * p[i] * alpha / theta[j]
            # convert to SI units
            # because [dt] = ps
            theta[i] = p[i] * atm * (theta_M - theta_E) * sigma * v_inv * (aux - 1.0 / aux) * dt * 1e-12 + theta[j]


    plt.plot(t, theta)

def ChemPotentialGas(T, p):
    mass = 40.
    m = mass * au
    g = 1. # degeneracy
    Lambda = np.sqrt(2.* pi * hbar**2 / (m * kB * T))
    Phi = T**(3./2.) * Lambda**3 / g
    mu = kB * T * np.log(p * Phi)
    return mu, Phi


def ChemPotentialSurface(T, theta, b, beta):
    b = 0.0
    # maybe introduce taylor expanded beta(theta) for potential
    mu = kB * T * np.log(theta / (1. - theta) / np.exp((b - beta)/kB*T)) # check correct use of theta!
    return mu

def CalcExchangeRate(p, T, theta, maxcov):
    mass = 40.
    m = mass * au
    M = maxcov / area # adsorption sites per unit area
    sigma = 1. / M # geometrical size of adsorption size == adsorption cross section
    K = p / np.sqrt(2.*pi*m*kB*T) * (1. - theta * maxcov) * sigma
    return K

def Pss(theta_e, theta_M, b, beta, T):
    return theta_e / ((theta_M - theta_e) * 1. * np.exp((b-beta)/(kB*T)))

def Theta(time, Pss, theta_M, theta_e, sigma, m, T, b, beta):
    theta = np.full(len(time), 0.0)
    dt = Time[1] - Time[0] # this assumes linear time range
    expo = np.exp((b-beta)/(kB*T))
    for i in range(1, len(theta)):

        j = i - 1
        aux0 = Pss * (theta_M - theta_e) * sigma / (np.sqrt(2.*pi*m*kB*T))
        aux1 = (theta_M - theta[j]) * Pss * expo / theta[j]
        aux2 = theta[j] / ( (theta_M - theta[j]) * Pss * expo )
        theta[i] += (aux0 * (aux1 - aux2)) * dt
    return theta





'''
Setup for analytical solution according to Ward, Findlay (1982)
'''

def zeroPressure():
    m = 40.0 * au
    T = 300.0
    beta = 1.0 / (kB * T)
    E = -0.085 * e0   # approx potential energy of adsorbed particle
    Z = 1.0 + np.exp(-beta * E)
    print(Z)
    lam = 2.0 * pi * hbar / np.sqrt(2.0*pi * m * kB * T)
    p_0 = 1.0 / (Z * lam * lam * lam * beta)
    print(p_0 / atm)


def PartitionFunction(u0, T, omega):
    denom = (1.0 - np.exp(-hbar * omega / (R * T)))
    print(denom)
    q = (np.exp(-hbar * omega / (2.0 * kB * T) )) / denom
    q = np.exp(-u0 / (kB*T)) * q**3
    return q

def PartitionFunctionSingle(V,T):
    return V * (2.0 * pi * kB * T / (h**2)) ** (3.0/2.0)

def ChemPotSurface(q, theta, T):
    mu = kB * T * np.log(theta / ((1. - theta) * q))
    return mu

def ChemPotGas(T, p):
    # p = 1.0 * atm
    mass = 40.0
    m = mass * au
    lam = 2.0 * pi * hbar / np.sqrt(2.0*pi * m * kB * T)
    Phi = kB * T * np.log(lam**3 / (kB*T))
    mu = Phi + kB * T * np.log(p)
    return mu, Phi

def CalcAlpha(q, Phi, T):
    return q * np.exp(Phi / (kB * T))

def CalcEverything():
    # try to obtain a sensible result for alpha
    T = 300
    p = 1.0
    p = 1.0 * atm
    u0 = -0.087563 # eV # u0 for 30 300 300 1.0
    u0 = u0 * e0 # J
    omega = 0.5e34

    q = PartitionFunction(u0, T, omega)
    print(q)
    q = 20000000
    mu, Phi = ChemPotGas(T,p)
    print(mu, Phi)
    alpha = CalcAlpha(q, Phi, T)
    print(alpha)
    print((1.0/alpha) / atm)

def CalcK(M, alpha, p, N_0, N):
    # take K as fit parameter to surface coverage plot
    M_prime = area * M
    eta = alpha * p
    Z = -eta / (2.*(eta**2 - -1.)**2)
    Theta = (eta**2 - 1.)*(N/M) - 1.*eta**2*(N/M)+eta**2
    Theta_0 = (eta**2 - 1.)*(N_0/M) - 1.*eta**2*(N_0/M)+eta**2

    aux0 = (eta**2 - 1.) * (N/M) - eta*(eta+1.)
    aux1 = (eta**2 +1.)*(N/M) - eta*(eta - 1.)
    phi = aux0 / aux1

    aux0 = (eta**2 - 1.) * (N_0/M) - eta*(eta+1.)
    aux1 = (eta**2 +1.)*(N_0/M) - eta*(eta - 1.)
    phi_0 = aux0 / aux1
    print("aux0:", asdaux0)
    print("aux1:", aux1)
    print("phi_0:",phi_0)


    K = 0.0
    K += ((eta**2 - 1.) * np.log(Theta/Theta_0) + 2.*theta * np.log(phi/phi_0))
    K += 2 * (eta**2 - 1.) * (N/M - N_0/M)
    K *= Z * M
    return K


def SurfaceAtoms(Time, alpha, M, p, start, N0):
    # alleged solution to the integral
    # Theta = K / area * (alpha * p * Time * (M-N[j]) / N[j] - N[j] * Time / (alpha * p * (M-N[j]) ) )

    # M: max coverage
    # alpha: fit parameter; alpha = q*exp(Phi/kT) where Phi is some thermodynamical fct
    N = np.full(len(Time), 0.0)
    N[start] = N0
    # hardcoded equilibration coverage for p = 1 atm
    N_eq = 5.5
    # K = CalcK(M, alpha, p, 0.0, N_eq)
    K = 1.0
    dt = Time[1] - Time[0] # this assumes linear time range
    dt *= 0.00025
    for i in range(start+1, len(Time)):
        j = i-1
        aux0 = K / area
        aux1 = (M - N[j]) * alpha * p / N[j]
        aux2 = N[j] / ((M - N[j]) * alpha * p)
        N[i] = aux0 * (aux1 - aux2) * dt
    return N

def IntegrateTheta(time,K, pressure=1., N0=0.5, start=24, alpha=0.05):
    # so far this function doesn't give good results for e.g. temp_S = 190 K

    N = np.full(len(time), 0.0)
    # N0 = 0.4875271260300431
    N_eq = 5.5 # deprecated
    M = 120.75 # maxcov, hardcoded!!!
    # start = 48
    # start = 10
    start = int(start)
    N[int(start)] = N0
    area = 1557e-20 #area hardcoded
    # area = 1.
    # alpha = 0.05 # obtained as manual fit to langmuir isotherms

    p = pressure
    conversion = 1. #/ area * (1e-9)**2
    dt = (time[1] - time[0]) * 0.00025
    for i in range(start+1, len(time)):
        j = i-1
        aux0 = K / area
        aux1 = (M - N[j]) * alpha * p / N[j]
        aux2 = N[j] / ((M - N[j]) * alpha * p)
        N[i] = aux0 * (aux1 - aux2) * dt + N[j]
    return N * conversion

# time = np.arange(0,1e7,1000)
# K = 1.0e-21
# N = IntegrateTheta(time, K)
# import matplotlib.pyplot as plt
# plt.plot(time*0.00025, N)


'''
Analytical Solution of particle populations according to the Rate Equation Model
'''

from flux import CalcFlux

def getFlux(pressure, temp_P, area, m):
    pressureSI = pressure * atm
    incidentTemp = temp_P
    incidentEnergy = incidentTemp * kB # = kB * T_inc
    incidentmeV = incidentEnergy / e0 * 1e3
    velocity = np.sqrt(2.*incidentmeV*e0/(m*1000.))
    density0 = pressureSI / (kB*temp_P)
    particleflux_fs, time_betw = CalcFlux(m, temp_P, velocity, pressureSI, area, density0, a=a, b=b)
    return particleflux_fs, time_betw

# when we have to arrays which essentially dexcribe the same quantity
# but differ in  their resolution (e.q. time resolution)
# we may have to convert from one resolution to another
#
# here we aim to convert from a high resolution (A)
# to a low resolution (B)
def InterpolateSlope(A, sB):
    #TODO Work on that!
    sA = len(A)-1
    assert(sA < 250)
    assert(sB < 200)
    # intermediate solution has a length which is divisible by both input lengths
    # this trait can be used later to cut up the array to the deired resolution
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

# find the boundaries for the non equilibrated data
# i.e. where the data is greater than zero (when a particle has already arrived on the surface)
# up to when the predefined equilibration time is reached, i.e. when the integrated contribution of non-eq. particles
# has become stationary (see definition of 'NonEqTerm' in fct 'constructCompleteSolution')
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

    return earliest, latest

def OpenSingleTrajData(fname):
    f = open(fname, 'r')
    data = []
    for i, line in enumerate(f):
        if '#' in line:
            continue
        data.append(float(line.split('=')[1]))
    print("Successfully read data from %s" % fname)
    f.close()
    return data

def unpackData(data, dt):
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

# open file which contains all information about the transition rates
# from the single particle solutions
def openSingleFile(angle, energy, temp_S):
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
        # we cannot resolve the first time period up until the equilibration time
        # therefore we need the pure md data in this section
        f1 = open(f1name, 'r')
        f2 = open(f2name, 'r')
        T, Q, dataTime = InitPopulations(f1, f2)
    except:
        print("Could not open %s and %s" % (f1name,f2name))
        return [],[],[],[]
    try:
        data = OpenSingleTrajData(fname)
        return data, T, Q, dataTime
    except:
        print("Could not open %s" % fname)
        return [],[],[],[]

# lambda1, lambda2, c1, c2, R21, R22, N1, N2, te
# are values from the single particle solution
# particleflux_fs is the value of particles per femtosecond on the surface
# tArr is the time axis
# shift is a shift in time by the average time it takes a particle to eventually reach the surface
def constructFluxSolution(lambda1, lambda2, c1, c2, R21, R22, N1, N2, te, particleflux_fs, time_betw, tArr, shift, t_0):
    # compute constants
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

    # for the fluxterms we need to regard the time (i.e. shift) it takes a particle to reach the surface
    tArrShift = tArr - shift
    fluxterm1 = (1. - np.exp(lambda1 * tArrShift)) * C1 / (C0 * np.abs(lambda1)) * (c11 + c21)
    fluxterm2 = (1. - np.exp(lambda2 * tArrShift)) * C2 / (C0 * np.abs(lambda2)) * (c12 + c22)

    # the singleterms are the same as in the pure single particle solution
    singletermT1 = c1 * (lambda1 - R22) * np.exp(lambda1 * tArr)
    singletermT2 = c2 * (lambda2 - R22) * np.exp(lambda2 * tArr)
    singletermQ1 = c1 * R21 * np.exp(lambda1 * tArr)
    singletermQ2 = c2 * R21 * np.exp(lambda2 * tArr)

    return fluxterm1, fluxterm2, singletermT1, singletermT2, singletermQ1, singletermQ2, phi

# after computing the single and flux terms in the function above
# now we add the shift factor for non-equilibrated particles on the surface
def constructCompleteSolution(fluxterm1, fluxterm2, singletermT1, singletermT2, singletermQ1, singletermQ2, T, Q, dataTime, dt, stw, phi):
    # initialize population
    Population = (fluxterm1 + fluxterm2)

    # create the term which corresponds to particles
    # that have gone to the surface but have not yet equilibrated
    NonEqTerm = [0.0 for i in range(len(Population))]

    # get delta t from the md data
    single_delta_t = (dataTime[1] - dataTime[0])
    single_delta_t0 = single_delta_t / 6.53
    flux_delta_t = dt * stw

    # use this delta t to integrate the contribution of non-eq. particles
    for i in range(1,len(T)):
        NonEqTerm[i] += (T[i] + Q[i]) * phi * single_delta_t0 + NonEqTerm[i-1]

    # after the equilibration time, all other particles are said to
    # always contribute the same amount to the adsorbate density (on average)
    for i in range(len(T),len(Population)):
        NonEqTerm[i] = NonEqTerm[i-1]


    # convert the slope of the NonEqTerm array, since we have different time resolution in the single particle and flux simulations
    earliest, latest = FindBoundaries(NonEqTerm)
    # how many steps are equal to tE in each simulation
    Single_eq_steps = latest-earliest
    ratio = single_delta_t / flux_delta_t
    Flux_eq_steps = np.ceil(Single_eq_steps * ratio)
    # in the single particle solution we had a higher time resolution (we wrote out data every 100 steps)
    # in the flux simulations the time resolution is much coarser (data written out every 1000 steps)
    NewNonEqTerm = InterpolateSlope(NonEqTerm[earliest:latest+1], int(Flux_eq_steps))
    del NonEqTerm

    # add remaining terms to Population
    Population += (singletermT1 + singletermT2 + singletermQ1 + singletermQ2)
    # ... including the non-eq. particle contribution
    for i in range(0, len(Population)):
        # remember that after the equilibration time NewNonEqTerm is constant!
        Population[i] += NewNonEqTerm[-1]

    # finally return our analytical solution and NewNonEqTerm as the pure md data
    return Population, NewNonEqTerm


def analyticalSolution(angle, temp_S, energy, pressure, temp_P, m, maxtime, cov):
    area = 1557e-20
    dt = 0.00025
    t_0 = 6.53
    stw = 1000 # StepsToWrite

    pressureSI = pressure * atm
    te = 12. / dt # conversion to timesteps

    particleflux_fs, time_betw = getFlux(pressure, temp_P, area, m)

    data, T, Q, dataTime = openSingleFile(angle, energy, temp_S)
    lambda1, lambda2, c1, c2, R21, R22, N1, N2, te, T_QT, T_CT, T_TQ, T_CQ = unpackData(data, dt)

    tArr = np.arange(0,maxtime,stw) * dt / t_0 # conversion to t_0, accounting for units in lambda: [lambda] = 1/t0

    avgHeight = 55. # angström
    avgVelo = np.sqrt(2. * kB * temp_P / m)
    avgTime = avgHeight / avgVelo * 1e2
    shift = avgTime / t_0
    tArr -= (te * dt / t_0)

    # create terms from analytical rate equation model
    fluxterm1, fluxterm2, singletermT1, singletermT2, singletermQ1, singletermQ2, phi = constructFluxSolution(
                                                    lambda1, lambda2, c1, c2, R21, R22, N1, N2, te, particleflux_fs, time_betw, tArr, shift, t_0)
    # add remaining terms and finalize the analytical solution
    population, mdData = constructCompleteSolution(fluxterm1, fluxterm2, singletermT1, singletermT2, singletermQ1, singletermQ2, T, Q, dataTime, dt, stw, phi)
    
    return population, mdData

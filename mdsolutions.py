import numpy as np
##### Constants
kB = 1.3806503e-23
e0 = 1.60217662e-19
pi = 3.14159265359
au = 1.66053904e-27
atm = 101325
NA = 6.02214086e23
hbar = 1.0545718e-34
R = NA * kB
###### VdW Constants for Argon
a = 1.355       # l^2 bar /mol^2
a = a * 1e-7 / (NA**2)
b = 0.03201     # l/mol
b = b * 1e-6 / NA
###### system specific constansts
area = 15.57e-20
substrate_atoms = 216
maxcov = 122.
cov = 1.0 #TODO set coverage
theta = cov / (substrate_atoms)
theta_M = maxcov / (substrate_atoms / area)

'''
Setup for analytical solution according to paper by Elliot, Ward (1997)
'''

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
Setup for analytical solution according to Ward, Finlay (1982)
'''

def PartitionFunction(u0, T):
    q = np.exp(-u0 / R / T)
    return q

def ChemPotSurface(q, theta, T):
    mu = kB * T * np.log(theta / ((1. - theta) * q))
    return mu

def ChemPotGas(T, p):
    mass = 40.
    m = mass * au
    g = 1. # degeneracy
    Lambda = np.sqrt(2.* pi * hbar**2 / (m * kB * T))
    Phi = T**(3./2.) * Lambda**3 / g
    mu = kB * T * np.log(p * Phi)
    return mu, Phi

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
    print("aux0:", aux0)
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

def IntegrateTheta(time,K):
    N = np.full(len(time), 0.0)
    N0 = 0.4875271260300431
    N_eq = 5.5
    M = 120.75
    start = 48*2
    N[start] = N0
    area = 1557e-20
    alpha = 0.05
    p = 1. * atm
    dt = (time[1] - time[0]) * 0.00025
    for i in range(start+1, len(time)):
        j = i-1
        aux0 = K / area
        aux1 = (M - N[j]) * alpha * p / N[j]
        aux2 = N[j] / ((M - N[j]) * alpha * p)
        N[i] = aux0 * (aux1 - aux2) * dt + N[j]
    return N


'''
pressure tensor yay
'''

def heaviside(x):
    if x > 0: return 1.
    elif x == 0: return 0.5
    else: return 0.0

# Lennard Jones parameters
ArEpsilon = 1.67e-21 / e0
ArSigma = 3.4000
r = np.arange(0.0, 60., 0.5)
F = 24. * ArEpsilon * ( ArSigma**12 * r**(-13) - 2. * ArSigma**6 * r**(-7) )
def PressureTensor(df, grid):
    # remember to account for interaction/force according to weights
    # weights are determined by the proportion of the connecting vector inside each unit volume

    # potential component
    p_u = np.full(len(grid), 0.0)

    for i,z in enumerate(grid):
        pass

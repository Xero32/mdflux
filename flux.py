## express flux phi as
# phi = 1/gamma * 1/t0
# with t0 = 6.53 fs

import numpy as np
import getvdwroots as vdw
###### Constants
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

def GetGamma(area, t0, mass, temp, target_pressure):
    global kB, au
    unit_pressure = np.sqrt(2.*mass*au*kB*temp) / (area * t0)
    # unit_pressure *= 2. ## if impingement is treated completely elastic
    gamma = unit_pressure / target_pressure
    return gamma, unit_pressure

def GetGamma0(mass, temp, velocity, pressure, t0, area, density=0, a=0, b=0):
    m = mass * au
    # velocity = np.sqrt(8.*kB*temp / (pi*m))             # expected velocity value
    # velocity = np.sqrt(kB*temp / (2.*pi*m))             # mean speed in 1 direction
    # flux = pressure * velocity / (kB*temp)
    # velocity = np.sqrt(2.*kB*temp / (m))               # most probable speed // corresponds to input velocity in lammps script

    if density == 0:
        density = pressure / (kB*temp)
    else:
        vdwVolume = vdw.NewtonIter(100000, 1./density, temp, pressure, a, b)
        vdwDensity = 1./vdwVolume
        density = vdwDensity
    print("Density: ", density, "m^-3")
    flux = density * velocity
    particle_flux = flux * area
    fs = 1e-15
    g = 1./flux * 1./fs
    # g = np.sqrt(mass*au*kB*temp / (2.*pi)) * 1. / (pressure*t0)
    # g = velocity / (kB*temp) / (pressure * t0)       # how many particles per unit area and unit time
    g0 = g/area                                             # how many particles strike our test area per unit time
    return g, g0

def CalcFlux(mass, temp, velocity, pressure, area, density, a=a, b=b):

    vdwVolume = vdw.NewtonIter(100000, 1./density, temp, pressure, a, b)
    vdwDensity = 1./vdwVolume
    density = vdwDensity
    flux = density * velocity
    particle_flux = flux * area
    particle_flux_fs = particle_flux * 1e-15
    time_between_deposits = 1./particle_flux_fs
    return particle_flux_fs, time_between_deposits

def main():
    global a,b
    area = 572e-20  # Alex' platinum area
    area = 1452e-20 # my gold area
    area = 1557e-20
    mass = 40.
    t0 = 6.53e-15
    temp = 300.
    target_pressure = 1. * atm
    fs = 1e-15

    IdealDensity = target_pressure / (kB*temp)
    # velocity = np.sqrt(2.*incidentmeV*e0/(mass*au*1000.))/100.
    velocity = np.sqrt(2.*kB*temp/mass/au)
    gamma, gamma0 = GetGamma0(mass, temp, velocity, target_pressure, t0, area, density=IdealDensity, a=a, b=b)
    print("Area: ", area, "m^2")
    # print("Unit pressure: ", unit_pressure, " Pa")
    print("Gamma: ", gamma, "1/m*m")
    print("Gamma0: ", gamma0, "fs between depositions")
    flux = 1./gamma * 1./fs
    particle_flux = 1./gamma0 * 1./fs

    BaseFlux, Time_Between = CalcFlux(mass, temp, velocity, target_pressure, area, IdealDensity)
    print("Base Flux: ", BaseFlux, "1/fs")
    InvBaseFlux = 1./BaseFlux
    print("Inverted Base Flux:", InvBaseFlux, "fs")

    print("Flux: ", flux, "1/(m*m*s)")
    print("Particle Flux: ", particle_flux, "1/s")

    # m = mass * au
    # pressure = target_pressure
    # velocity = np.sqrt(8.*kB*temp / (pi*m))             # expected velocity value
    # print(velocity)
    # velocity = np.sqrt(kB*temp / (2.*pi*m))             # mean speed in 1 direction
    # print(velocity)
    # velocity = np.sqrt(2.*kB*temp / (m))     # most probable speed // corresponds to input velocity in lammps script
    # print(velocity)
    # J = pressure / (kB*temp) * velocity
    # dN = J * area * t0
    # gamma = 1./dN
    # print(gamma)

if __name__ == "__main__":
    main()

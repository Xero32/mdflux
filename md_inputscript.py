#### configures a LAMMPS input script
import numpy as np
import math
import random
import shlex
import subprocess
import time
import argparse

def InpParams(parser):
    parser.add_argument("a", help='Angle', default='30', nargs='?')
    parser.add_argument("ts", help='Surface Temperature', default='300', nargs='?')
    parser.add_argument("tp", help='Plasma Temperature', default='300', nargs='?')
    parser.add_argument("p", help='Plasma Pressure', default='10', nargs='?')

    args = parser.parse_args()
    if args.a:
        angle = float(args.a)
    if args.ts:
        temp_S = int(args.ts)
    if args.tp:
        temp_P = int(args.tp)
    if args.p:
        pressure = float(args.p)

    return angle, temp_S, temp_P, pressure

parser = argparse.ArgumentParser()
incidentAngle, temp_S, temp_P, pressure = InpParams(parser)

####### Simulation Setup
duration = 100 # t0
# temp_S = 190 # K, surface temperature               range(80,190,300)
# temp_P = 300 # K, plasma temperature                range(190,300)
# pressure = 10   # datm, gas/plasma pressure
# incidentAngle = 0.52 # radians
Num_of_Simulations = 200 # Number of iterations for statistical handling
logFlag = False
Cluster = 'hlrn'
beam = False    # False: random direction of incident particles with fixed angle to surface normal
                # makes use of one degree of freedom
                # True: sets up all incident particles parallel to each other
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

##### Simulation Parameters
unit_time = 0.00653 # ps # just for reference, not relevant for simulation-setup
delta_t = 0.00025   # ps
delta_t_fs = 0.25   # fs
delta_t_sec = delta_t * 1e-12
duration = unit_time * 1e3 * duration
stepnum = duration / delta_t
incidentTemp = temp_P
incidentEnergy = incidentTemp * kB # = kB * T_inc
incidentmeV = incidentEnergy / e0 * 1e3
pressure /= 10.



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

###### function for volume/particle from VdW-equation and its derivative
        # used in Newton iteration; find roots of this function
def function(v, T, P, a, b):
    return v**3 - (b + kB*T / P) * v**2 + a / P * v - a * b / P

def derivative(v, T, P, a, b):
    return 3. * v**2 - 2. * (b + kB*T / P) * v + a / P
######

def NewtonIter(maxIter, v0, T, P, a, b):
    # setup Newton iteration to find the Van-der-Waals corrected density (and volume per particle)
    epsilon = 1e-100
    ideal = v0
    flag = False
    i = 0
    # for i in range(maxIter):
    while(flag == False and i < maxIter):
        y = function(v0, T, P, a, b)
        yprime = derivative(v0, T, P, a, b)

        if (np.abs(yprime) < epsilon):
            break

        v1 = v0 - y/yprime

        if(abs(v1-v0) <= epsilon * abs(v1)):
            flag = True
            print("Newton-Iteration succesful\n")
            break
        else:
            v0 = v1

        i += 1
    if (flag == False):
        print("Newton-Iteration ran for %d steps\n" % maxIter)
    if (ideal/v1 > 1.25 or ideal/v1 < 0.75):
        print("Caution! Van der Waals value differs from ideal gas law by more than 25%.\n")
    return v1

def CalcFlux(mass, temp, velocity, pressure, area, density, a=a, b=b):
    # Calculate the flux for given plasma temperature and pressure
    # onto the simulated surface area
    # returns: particles/fs and time[fs] between particle depositions for the given area
    vdwVolume = NewtonIter(100000, 1./density, temp, pressure, a, b)
    vdwDensity = 1./vdwVolume
    density = vdwDensity
    flux = density * velocity
    particle_flux = flux * area
    particle_flux_fs = particle_flux * 1e-15
    time_between_deposits = 1./particle_flux_fs

    return particle_flux_fs, time_between_deposits

processors = ([6,4,1])
boundary = ['p','p','f']
lattice_const = 4.08
substrateZmax = lattice_const * 3.
substrateYmax = lattice_const * 9. * np.sqrt(3) / np.sqrt(2)
substrateXmax = lattice_const * 12. / np.sqrt(2)
xInsert = 0.5 * substrateXmax
yInsert = 0.5 * substrateYmax
zInsertMin = substrateZmax + 50.
zInsertMax = substrateZmax + 60.
zBoxMax = zInsertMax
zRemove = substrateZmax + 50.
typeNewAtom = 2
mass = 40.

velocityProjectile = -np.sqrt(2.*incidentmeV*e0/(mass*au*1000.))/100. # gives velocity in Angström/ps (metal units in LAMMPS)


area = 1557e-20
# area = 572e-20 # area for Alex' platinum sample
# t0 = 6.53e-15
target_pressure = pressure * atm
IdealDensity = target_pressure / (kB*temp_P)

flux_per_fs, time_between_fs = CalcFlux(mass, temp_P, np.abs(velocityProjectile)*100., target_pressure, area, IdealDensity, a=a, b=b)
fluxsteps = int(time_between_fs / delta_t_fs)
mindistance = 4.0 # minimal distance between particles upon insertion # used in LAMMPS deposit function

###### create filename for given parameters
_temp_S = str(temp_S)
_temp_P = str(temp_P)
_pr = str(pressure)
_T_in = str(incidentTemp)
_E_in = str(incidentEnergy)
_pressure = NoPunctuation(pressure,1)
_angle = NoPunctuation(incidentAngle, 2)
parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"
filename = parameter_set_str + ".in"
ScriptName = filename
print(filename)
f = open(filename, 'w')

def WriteGeneral(f, processors, boundary):
    f.write("# GENERAL\n\n")
    if logFlag == True:
        f.write('log             ./scriptlog.lammps\n')     # TODO remove for use on computing clusters
    else:
        f.write('log none\n')
    f.write("processors %d %d %d\n" %(tuple(processors)))
    f.write("units metal\n")
    f.write("boundary %s %s %s\n" %(tuple(boundary)))
    f.write("atom_style atomic\natom_modify map array sort 10000 0.0\n")

def WriteVariables(
    f, lattice_const, substrateXmax, substrateYmax, substrateZmax,
    xInsert, yInsert, zInsertMin, zInsertMax, zBoxMax, zRemove,
    typeNewAtom, velocityProjectile, incidentmeV,
    temp_S, temp_P, pressure, incidentAngle, fluxsteps, mindistance):

    f.write('\n\n# VARIABLES\n\n')
    f.write('variable lattice_const equal "%.3f"\n' % lattice_const)
    f.write('variable substrateZmax equal "%.3f"\n' % substrateZmax)
    # f.write('variable substrateXmax equal "v_latticeConstant * 9 * sqrt(3) / sqrt(2)"\n')
    # f.write('variable substrateYmax equal "v_latticeConstant * 12 / sqrt(2)"\n')
    f.write('variable substrateXmax equal "%.5f"\n' % (lattice_const * 9. * math.sqrt(3.) / math.sqrt(2.)))
    f.write('variable substrateYmax equal "%.5f"\n' % (lattice_const * 12. / math.sqrt(2.)))
    f.write('variable xInsert equal "%f"\n' % xInsert)
    f.write('variable yInsert equal "%f"\n' % yInsert)
    f.write('variable zInsertMin equal "%.3f"\n' % zInsertMin)
    f.write('variable zInsertMax equal "%.3f"\n' % zInsertMax)
    f.write('variable zBoxMax equal "%.2f"\n' % zBoxMax)
    f.write('variable zRemove equal "%.2f"\n' % zRemove)
    f.write('variable typeNewAtom equal "%d"\n' % typeNewAtom)
    f.write('variable incidentmeV equal "%.3f"\n' % incidentmeV)
    f.write('variable velocityProjectile equal "-sqrt(v_incidentmeV)*0.6947"\n')
    f.write('variable SurfaceTemp equal "%d"\n' % temp_S)
    f.write('variable PlasmaTemp equal "%d"\n' % temp_P)
    f.write('variable PlasmaPressure equal "%.3f"\n' % pressure)
    f.write('variable incidentAngle equal "%.3f"\n' % incidentAngle)
    f.write('variable StepsToDeposit equal "%d"\nvariable MinDistance equal "%.1f"\n' % (fluxsteps,mindistance)) #TODO work on WaitTime between particle insertions

    ##NB give output folder in lammps run command

def WriteRegionsGroups():
    f.write('\n\n# REGIONS\n\n')
    f.write('region simulationBoxRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} 0.0 ${zBoxMax}\n')
    f.write('region substrateAtomsRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} 0.0 ${substrateZmax}\n')
    f.write('region mobileSubstrateAtomsRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} 5.0 ${substrateZmax}\n')
    f.write('region insertRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} ${zInsertMin} ${zInsertMax}\n')
    f.write('region bulkRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} ${substrateZmax} ${zRemove}\n')
    f.write('create_box 2 simulationBoxRegion\n')
    f.write('\n\n# GROUPS\n\n')
    f.write('group substrateGroup region substrateAtomsRegion\n')
    f.write('group mobileSubstrateAtomsGroup region mobileSubstrateAtomsRegion\n')
    f.write('group adAtomGroup type 2\n')
    f.write('print "--- Incident Angle: $(v_incidentAngle) rad, Surface Temperature: $(v_SurfaceTemp) K, Plasma Temperature: $(v_PlasmaTemp) K, Incident Energy: $(v_incidentmeV) meV, Pressure: $(v_PlasmaPressure) atm"\n')


def WriteInteraction(
    ljcut=12.0, borncut=13.0,
    AuEpsilon=0.2294, AuSigma=2.629,
    ArAuA=3592.5, ArAuRho=0.34916, ArAuSigma=0.0, ArAuC=44.99, ArAuD=-2481.30, ArAuCut=13.0,
    ArEpsilon=0.0104, ArSigma=3.4,
    AuMass=197, ArMass=40):
    f.write('\n\n# INTERACTION\n\n')
    f.write('pair_style hybrid lj/cut %.1f born %.1f\n' %(ljcut, borncut))
    f.write('pair_coeff 1 1 lj/cut %f %f\n' %(AuEpsilon, AuSigma))
    f.write('pair_coeff 1 2 born %f %f %f %f %f %f\n' %(ArAuA, ArAuRho, ArAuSigma, ArAuC, ArAuD, ArAuCut))
    f.write('pair_coeff 2 2 lj/cut %f %f\n' %(ArEpsilon, ArSigma))
    f.write('fix bottomWall all wall/reflect zlo 0.0\n')
    f.write('mass 1 %d\nmass 2 %d\n' %(AuMass, ArMass))

def WriteThermostat():
    f.write('\n\n# THERMOSTAT\n\n')
    f.write('thermo_modify   lost ignore flush yes\n'+
    'fix             substrateLV mobileSubstrateAtomsGroup langevin $(v_SurfaceTemp) $(v_SurfaceTemp) 1.0 $(v_rand)\n'+
    'fix             substrateNVE mobileSubstrateAtomsGroup nve\n')  #TODO remove abortion condition

    f.write('thermo          1000\n'
    'compute         temperatureOfMobileAtoms mobileSubstrateAtomsGroup temp\n'+
    'thermo_style    custom step cpu c_temperatureOfMobileAtoms pe etotal \n'+
    'thermo_modify   lost ignore flush yes\n')
    #TODO get variable of ALL argon atoms, not just one!!
    # 'variable        numberOfAtoms equal count(all)\n'+
    # 'variable        numberAdAtoms equal count(adAtomGroup)\n'+
    # 'variable        xCoordinate equal x[v_numberOfAtoms]\n'+
    # 'variable        yCoordinate equal y[v_numberOfAtoms]\n'+
    # 'variable        zCoordinate equal z[v_numberOfAtoms]\n'+
    # 'variable        xVelo equal vx[v_numberOfAtoms]\n'+
    # 'variable        yVelo equal vy[v_numberOfAtoms]\n'+
    # 'variable        zVelo equal vz[v_numberOfAtoms]\n')

def WriteRun(fluxsteps, NumOfAd, tstep=0.00025, step_num=1000000, beam=False):

    f.write('\n\n# RUN\n\n')
    f.write('timestep %f\n' %tstep)
    f.write('fix adAtomNVEfix adAtomGroup nve\n')
    f.write("compute PotEnergy adAtomGroup pe/atom\n\n")
    f.write("# LOOP\n\n")
    f.write("variable imax equal %d\n" % Num_of_Simulations)
    f.write("variable i loop ${imax}\n")
    f.write("label LOOP_START\n\n")

    f.write('read_data ${inFolder}/thermal${SurfaceTemp}Au.dat add merge\n')
    f.write('group substrateGroup region substrateAtomsRegion\n')
    f.write('group mobileSubstrateAtomsGroup region mobileSubstrateAtomsRegion\n')
    f.write('group adAtomGroup type 2\n')
    f.write('delete_atoms group adAtomGroup\n')
    f.write('delete_atoms region bulkRegion\n')

    # f.write('dump DumpID all custom 1000 ${outFolder}/%dfluxAll${i}.lammpstrj type id proc x y z ix iy iz vx vy vz fx fy fz\n' % temp_S)
    f.write("dump ArDump adAtomGroup custom 1000 ${outFolder}/flux${i}.lammpstrj type id x y z ix iy iz vx vy vz c_PotEnergy\n\n\n")

    #TODO remove for runs on computing cluster


    for ctr in range(0,NumOfAd):

        if beam == False:
            A = random.randint(0,10000)
            f.write('\nvariable rndm equal random(-100.0,100.0,%d)\n' % A)
            f.write('variable veloX equal $(v_velocityProjectile*sin(v_incidentAngle)*sin(0.0628*v_rndm))\n')
            f.write('variable veloY equal $(v_velocityProjectile*sin(v_incidentAngle)*cos(0.0628*v_rndm))\n')
            f.write('variable veloZ equal $(v_velocityProjectile*cos(v_incidentAngle))\n')
        else:
            A = 100.0
            f.write('variable veloX equal $(v_velocityProjectile*sin(v_incidentAngle)*sin(0.0628*%f))\n' % A)
            f.write('variable veloY equal $(v_velocityProjectile*sin(v_incidentAngle)*cos(0.0628*%f))\n' % A)
            f.write('variable veloZ equal $(v_velocityProjectile*cos(v_incidentAngle))\n')

        B = random.randint(0,100000)
        f.write('fix DepositAtoms adAtomGroup deposit 1 2 $(v_StepsToDeposit) %d region insertRegion &\n' % B)
        f.write('vx $(v_veloX) $(v_veloX) &\n'+
        'vy $(v_veloY) $(v_veloY) &\n'+
        'vz $(v_veloZ) $(v_veloZ) &\n'+
        'near %d attempt 10 id next\n' % mindistance)


        #TODO create output file/dump
        #TODO write routine to analyze output file
        #

        # TODO choose one of the following methods to delete desorbed particles far beyond the surface
        # f.write('variable RemoveCoord equal "v_zCoordParticle >= v_zRemove"\n'+
        # f.write('variable RemoveVelo equal "v_zVelocityParticle > 0"\n'+
        # 'variable RemoveCond equal $(v_RemoveCoord) && $(v_RemoveVelo)\n'+
        # 'if "$(v_xCoordParticle) >= $(v_zRemove)" && "$(v_zVelocityParticle) > 0" then group DeleteAtom == $(v_ParticleID)\n') #find particle ID

        # f.write('if "$(v_xCoordinate) >= $(v_zRemove)" && "$(v_zVelo) > 0" then group DeleteAtom == $(v_numberOfAtoms)\n') #TODO Caution!!!! find correct particle ID!
        # f.write('fix EvaporateAtoms DeleteAtom evaporate 1 1000000 insertRegion 2\n')


        # f.write('fix outputData all print 100 "$(step*dt) $(v_xCoordinate) $(v_yCoordinate) $(v_zCoordinate) $(v_xVelo) $(v_yVelo) $(v_zVelo) $(pe)" file ${outFolder}/${i}.dat screen no title "time [0]    x [1]    y [2]    z [3]    vx [4]    vy [5]    vz [6]      pe [7]"\n')
        # f.write('dump adAtomDump adAtomGroup custom 100 ${outFolder}/${i}.lammpstrj id c_PotEnergy\n') #NB maybe choose lammpstrj as default output file

        C = random.randint(-20,20)
        f.write('run %d\n\n' % (fluxsteps + C))

    f.write("delete_atoms group all\n")
    f.write("undump ArDump\n")
    # f.write("undump DumpID\n")
    f.write("next i\n")
    f.write("jump SELF LOOP_START\n")


def GenerateCommand(cluster, rand, filename):
    ## used for actual execution of LAMMPS simulation
    if cluster == "local":
        processors = [2,2,1]
        proc_number = processors[0] * processors[1] * processors[2]
        inFolder = "/home/becker/lammps"
        outFolder = "/home/becker/lammps/test/" + parameter_set_str
        command = str("mpirun -np %d lammps-daily -var rand %d -var outFolder %s -var inFolder %s -in %s/%s\n" %(proc_number, rand, outFolder, inFolder, inFolder, filename))
        filename = str("filename=%s/%s" % (inFolder, filename))

    elif cluster == "hlrn" or cluster == "HLRN":
        processors = [6,4,1]
        proc_number = processors[0] * processors[1] * processors[2]
        inFolder = str("${HOME}/%s" % parameter_set_str) #TODO
        outFolder = "${WORK}" #TODO
        command = str("aprun -n %d lmp_mpi -var rand %d -var outFolder %s -var inFolder %s -in ${HOME}/%s\n" %(proc_number, rand, outFolder, inFolder, filename))
        filename = str("filename=%s/%s" % (inFolder, filename))
    elif cluster == "rzcluster" or cluster == "rz":
        processors = [4,3,1]
        proc_number = processors[0] * processors[1] * processors[2]
        inFolder = "" #TODO
        outFolder = "" #TODO
        command = str("mpirun -np %d lammps-daily -var rand %d -var outFolder %s -var inFolder %s -in %s/%s\n" %(proc_number, rand, outFolder, inFolder, inFolder, filename))
        filename = str("filename=%s/%s" % (inFolder, filename))
            #NB set variables 'rand', 'inFolder', and 'outFolder' in lammps execution command line

    return processors, proc_number, command, filename, inFolder, outFolder

def main():
    global fluxsteps, mindistance, ScriptName, beam, Cluster
    infolder = "/home/becker/lammps"
    proc, proc_number, cmd, namevar, infolder, outfolder = GenerateCommand(Cluster, random.randint(0,10000), ScriptName)
    WriteGeneral(f, proc, boundary)
    WriteVariables(
        f, lattice_const, substrateXmax, substrateYmax, substrateZmax,
        xInsert, yInsert, zInsertMin, zInsertMax, zBoxMax, zRemove,
        typeNewAtom, velocityProjectile, incidentmeV,
        temp_S, temp_P, pressure, incidentAngle, fluxsteps, mindistance)

    WriteRegionsGroups()

    # LJ-potential coefficients
    ArEpsilon = 1.67e-21 / e0
    ArSigma = 3.4000
    WriteInteraction(ArEpsilon=ArEpsilon, ArSigma=ArSigma)
    WriteThermostat()
    NumOfAd = int(stepnum/fluxsteps)

    WriteRun(fluxsteps, NumOfAd, tstep=delta_t, step_num=stepnum, beam=beam)
    args = shlex.split(cmd)  # just in case

    #TODO maybe try to create a shell variable containing the path/to/inputfile string
    # and then execute it in the following line in the job script
    # ID = p.pid
    # print(ID)
    return cmd, args, infolder, outfolder

if __name__ == "__main__":
    cmd, args, infolder, outfolder = main()
    print("mkdir %s;\n" % outfolder)
    print(cmd)

    # calling the command directly causes the writing to input script to abort halfway through
    # p = subprocess.run(cmd, shell=True)

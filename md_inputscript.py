#### configures a LAMMPS input script
import numpy as np

k_B = 1.38064852e-23
e0 = 1.60217662e-19

temp_S = 300 # K, surface temperature
temp_P = 300 # K, plasma temperature
pressure = 1.0 # atm, gas/plasma pressure
incidentTemp = 300
incidentAngle = 0.52
incidentEnergy = incidentTemp * k_B # = k_B * T_inc
incidentmeV = incidentEnergy / e0 * 1e3


def NoPunctuation(d, places):
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

processors = ([6,4,1])
boundary = ['p','p','f']
lattice_const = 4.08
substrateZmax = lattice_const * 3.
substrateYmax = lattice_const * 9. * np.sqrt(3) / np.sqrt(2)
substrateXmax = lattice_const * 12. / np.sqrt(2)
xInsert = 0.5 * substrateXmax
yInsert = 0.5 * substrateYmax
zInsertMin = substrateZmax + 50.
zInsertMax = substrateZmax + 70.
zBoxMax = zInsertMax + 5.
zRemove = substrateZmax + 50.
typeNewAtom = 2
velocityProjectile = -np.sqrt(incidentEnergy)*0.6947  # factor assumes argon mass = 40 u

fluxsteps = 260 # how many steps between particle depositions #TODO
mindistance = 6.0 # minimal distance between particles upon insertion #TODO

_temp_S = str(temp_S)
_temp_P = str(temp_P)
_pr = str(pressure)
_T_in = str(incidentTemp)
_E_in = str(incidentEnergy)
_pressure = NoPunctuation(pressure,1)
_angle = NoPunctuation(incidentAngle, 2)
filename = "A" + _angle + "_E" + _T_in + "K_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm.in"

print(filename)
f = open(filename, 'w')

def WriteGeneral(f, processors, boundary):
    f.write("# GENERAL\n\n")
    f.write("processors %d %d %d\n" %(tuple(processors)))
    f.write("units metal\n")
    f.write("boundary %s %s %s\n" %(tuple(boundary)))
    f.write("atom_style atomic\natom_modify map array sort 10000 0.0\n")

def WriteVariables(
    f, lattice_const, substrateXmax, substrateYmax, substrateZmax,
    xInsert, yInsert, zInsertMin, zInsertMax, zBoxMax, zRemove,
    typeNewAtom, velocityProjectile, incidentmeV,
    temp_S, temp_P, pressure, incidentAngle):
    f.write('\n\n# VARIABLES\n\n')
    f.write('variable lattice_const equal "%.3f"\n' % lattice_const)
    f.write('variable substrateZmax equal "%.3f"\n' % substrateZmax)
    f.write('variable substrateXmax equal "v_latticeConstant * 9 * sqrt(3) / sqrt(2)"\n')
    f.write('variable substrateYmax equal "v_latticeConstant * 12 / sqrt(2)"\n')
    f.write('variable xInsert equal "%f"\n' % xInsert)
    f.write('variable yInsert equal "%f"\n' % yInsert)
    f.write('variable zInsertMin equal "%.3f"\n' % zInsertMin)
    f.write('variable zInsertMax equal "%.3f"\n' % zInsertMax)
    f.write('variable zBoxMax equal "%.2f"\n' % zBoxMax)
    f.write('variable zRemove equal "%.2f"\n' % zRemove)
    f.write('variable typeNewAtom equal "%d"\n' % typeNewAtom)
    f.write('variable incidentmeV equal "%.3f"\n' % incidentmeV)
    f.write('variable velocityProjectile "-sqrt(v_incidentmeV)*0.6947"\n')
    f.write('variable SurfaceTemp equal "%d"\n' % temp_S)
    f.write('variable PlasmaTemp equal "%d"\n' % temp_P)
    f.write('variable PlasmaPressure equal "%.3f"\n' % pressure)
    f.write('variable incidentAngle equal "%.3f"\n' % incidentAngle)
    f.write('variable StepsToDeposit equal "%d"\nvariable MinDistance equal "%.1f"\n' % (fluxsteps,mindistance)) #TODO work on WaitTime between particle insertions
    f.write('print "--- Incident Angle: $(v_incidentAngle) rad, Surface Temperature: $(v_SurfaceTemp) K, Plasma Temperature: $(PlasmaTemp) K, Incident Energy: $(v_incidentmeV) meV, Pressure: $(v_PlasmaPressure) atm"\n')
    ##NB give output folder in lammps run command

def WriteRegionsGroups():
    f.write('\n\n# REGIONS\n\n')
    f.write('region simulationBoxRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} 0.0 ${zBoxMax}\n')
    f.write('region substrateAtomsRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} 0.0 ${substrateZmax}\n')
    f.write('region mobileSubstrateAtomsRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} 5.0 ${substrateZmax}\n')
    f.write('region insertRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} ${zInsertMin} ${zInsertMax}\n')
    f.write('region bulkRegion block 0.0 ${substrateXmax} 0.0 ${substrateYmax} ${substrateZmax} ${zRemove}\n')
    f.write('create_box 2 simulationRegion\n')
    f.write('\n\n# GROUPS\n\n')
    f.write('group substrateGroup region substrateAtomsRegion\n')
    f.write('group mobileSubstrateAtomsGroup region mobileSubstrateAtomsRegion\n')
    f.write('group adAtomGroup type 2\n')

def WriteInteraction(
    ljcut=12.0, borncut=13.0,
    AuEpsilon=0.2294, AuSigma=2.629,
    ArAuA=3592.5, ArAuRho=0.34916, ArAuSigma=0.0, ArAuC=44.99, ArAuD=-2481.30, ArAuCut=13.0,
    ArEpsilon=0.01, ArSigma=3.2,
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
    'fix             substrateLV mobileSubstrateAtomsGroup langevin ${SurfaceTemp} ${SurfaceTemp} 1.0 ${rand}\n'+
    'fix             substrateNVE mobileSubstrateAtomsGroup nve\n'+

    'variable        positionCheck equal "v_zCoordinate > v_substrateZmax + 15.0"\n'+
    'fix             stoppingConditionFix all halt 1 v_positionCheck != 0 error continue\n'+  #TODO remove abortion condition

    #dump            unscaledDump all custom 100 ./all.lammpstrj id type x y z ix iy iz vx vy vz
    #log             ./log.lammps
    'log             none\n'+

    'thermo          1000\n'+
    'compute         temperatureOfMobileAtoms mobileSubstrateAtomsGroup temp\n'+
    'thermo_style    custom step cpu c_temperatureOfMobileAtoms pe etotal \n'+
    'thermo_modify   lost ignore flush yes\n'+
    #TODO get variable of ALL argon atoms, not just one!!
    'variable        numberOfAtoms equal count(all)\n'+
    'variable        xCoordinate equal x[v_numberOfAtoms]\n'+
    'variable        yCoordinate equal y[v_numberOfAtoms]\n'+
    'variable        zCoordinate equal z[v_numberOfAtoms]\n'+
    'variable        xVelo equal vx[v_numberOfAtoms]\n'+
    'variable        yVelo equal vy[v_numberOfAtoms]\n'+
    'variable        zVelo equal vz[v_numberOfAtoms]\n')

def WriteRun(tstep=0.00025, step_num=1000000):
    f.write('\n\n# RUN\n\n')
    f.write('timestep %f\n' %tstep)
    f.write('fix adAtomNVEfix adAtomGroup nve\n')
    f.write('compute PotEnergy adAtomGroup pe/atom\n')
    f.write('read_data ${inFolder}/thermal${temperature}.dat add merge\n')
    f.write('fix DepositAtoms adAtomGroup deposit 1 2 v_StepsToDeposit 123456 region insertRegion &\n'+
    '   vx $(v_velocityOfProjectile*sin(v_incidentAngle)*sin(0.0628*v_i)) &\n'+
    '   vy $(v_velocityOfProjectile*sin(v_incidentAngle)*cos(0.0628*v_i)) &\n'+
    '   vz $(v_velocityOfProjectile*cos(v_incidentAngle)) &\n'+
    'near v_MinDistance attempt 10 id next\n')

    # TODO choose one of the following methods to delete desorbed particles far beyond the surface
    f.write('variable RemoveCoord equal "v_zCoordParticle >= v_zRemove"\n'+
    'variable RemoveVelo equal "v_zVelocityParticle > 0"\n'+
    'variable RemoveCond equal $(v_RemoveCoord) && $(v_RemoveVelo)\n'
    'if "$(v_xCoordParticle) >= $(v_zRemove)" && "$(v_zVelocityParticle) > 0" then group DeleteAtom == $(v_ParticleID)\n')
    f.write('fix EvaporateAtoms DeleteAtom evaporate 1 1000000 insertRegion 2\n')
    f.write('fix outputData all print 100 "$(step*dt) $(v_xCoordinate)  $(v_yCoordinate) $(v_zCoordinate) $(v_xVelo) $(v_yVelo) $(v_zVelo) $(pe)" file ${outFolder}/${i}.dat screen no title "time [0]    x [1]    y [2]    z [3]    vx [4]    vy [5]    vz [6]      pe [7]"\n')
    f.write('dump adAtomDump adAtomGroup custom 100 ${outFolder}/${i}.lammpstrj c_PotEnergy\n')
    f.write('run "%d"\n' % step_num)

def main():
    WriteGeneral(f, processors, boundary)
    WriteVariables(
        f, lattice_const, substrateXmax, substrateYmax, substrateZmax,
        xInsert, yInsert, zInsertMin, zInsertMax, zBoxMax, zRemove,
        typeNewAtom, velocityProjectile, incidentmeV,
        temp_S, temp_P, pressure, incidentAngle)
    #NB set variables 'rand', 'inFolder', and 'outFolder' in lammps execution command line
    WriteRegionsGroups()
    WriteInteraction()
    WriteThermostat()
    WriteRun()

if __name__ == "__main__":
    main()

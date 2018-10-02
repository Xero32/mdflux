import numpy as np
import argparse


#Parameters: given as input!
angle = 0.52    # radians
pressure = 1.0    # atm
temp_P = 300    # K, plasma temperature
temp_S = 80    # K, surface temperature
# flags
vary_pressure = True
vary_temp_S = False
vary_temp_P = False
vary_angle = False


def NoPunctuation(d, places=2):
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

if vary_temp_P == True:
    if vary_temp_S == True:
        jobname = str('flux_A%s_P%s.job' %(NoPunctuation(angle), NoPunctuation(pressure, places=1)))
    else:
        jobname = str('flux_A%s_TS%d_P%s.job' %(NoPunctuation(angle), temp_S, NoPunctuation(pressure, places=1)))
elif vary_temp_S == True:
    jobname = str('flux_A%s_TP%d_P%s.job' %(NoPunctuation(angle), temp_P, NoPunctuation(pressure, places=1)))
elif vary_pressure == True:
    jobname = str('flux_A%s_TS%d_TP%d.job' %(NoPunctuation(angle), temp_S, temp_P))
f = open(jobname, 'w')


def WriteHeader(cluster, walltime, nodes, ppn):
    if cluster == 'hlrn' or cluster == 'HLRN':
        f.write('#PBS -N mdArAu\n'+
        '#PBS -N mdArAu\n'+
        '#PBS -j oe\n'+
        '#PBS -q mppq\n')
        f.write('#PBS -l nodes=%d:ppn=%d\n' %(nodes, ppn))
        f.write('#PBS -l walltime=%d:00:00\n' % walltime)
        f.write('#PBS -A shp00015\n\n')

        f.write('cd $PBS_O_WORKDIR\n')
        f.write('cd $WORK\n')
        f.write('module load lammps\n')
        f.write('pwd\n\n')

    elif cluster == 'rz' or cluster =='rzcluster':
        f.write("")
        #TODO

def WriteParameters(temp_P, temp_S, angle, pressure):
    f.write('PlasmaTemp=%d\n' % temp_P)
    f.write('SurfaceTemp=%d\n' % temp_S)
    f.write('Angle=%f\n' % angle)
    f.write('Pressure=%f\n\n' % pressure)

def WriteLoop(cluster, Temp_S=[], Temp_P=[], Angle=[], Pressure=[]):
    if len(Temp_P) != 0:
        f.write('for T_P in ')
        for tp in Temp_P:
            f.write('%d ' % tp)
        f.write('; do\n')
    if len(Temp_S) != 0:
        f.write('for T_S in ')
        for ts in Temp_S:
            f.write('%d ' % ts)
        f.write('; do\n')
    f.write('\n')

    f.write('nano=$(date +%N)\n'+
    'seed=$((nano%100000))\n')
    f.write('scriptname=A%s_TS${T_S}K_TP${T_P}K_p%sdatm\n' % (NoPunctuation(angle), NoPunctuation(pressure, places=1)))
    f.write('echo start HLRN run for $scriptname\n')
    if cluster == 'hlrn' or cluster == 'HLRN':
        f.write('FOLDER=${WORK}/${scriptname}/${PBS_JOBID}\n')
        f.write('mkdir -p ${FOLDER}\n\n')
        f.write('aprun -n 24 lmp_mpi -screen none -var rand ${seed} -var outFolder ${FOLDER} -var inFolder ${HOME} -in ${HOME}/${scriptname}.in &\n')

    elif cluster == 'rzcluster' or cluster == 'rz':
        f.write('FOLDER=${WORK}/${scriptname}\n')
        f.write('mkdir -p ${FOLDER}\n\n')
        f.write('mpirun -np 24 lmp_mpi -screen none -var rand ${seed} -var outFolder ${FOLDER} -var inFolder ${HOME} -in ${HOME}/${scriptname}.in &\n')

    f.write('done\ndone\nwait\n')
    f.write(
            'end=$(date +%s)\n'+
            'echo Beende hlrn run\n'+
            'secs=$((end-start))\n')
    f.write('echo Laufzeit:%s\n')
    f.write("printf '%dd:%dh:%dm:%ds' $(($secs/86400)) $(($secs%86400/3600)) $(($secs%3600/60)) $(($secs%60))\n")

def WritePressureLoop(cluster, angle, temp_S, temp_P, Pressure=[]):
    f.write('T_S=%d\n' % temp_S)
    f.write('T_P=%d\n\n' % temp_P)
    f.write('for p in ')
    for p in Pressure:
        f.write('%d ' % p)
    f.write('; do\n\n')

    f.write('nano=$(date +%N)\n'+
    'seed=$((nano%100000))\n')
    f.write('scriptname=A%s_TS${T_S}K_TP${T_P}K_p${p}datm\n' % NoPunctuation(angle))
    f.write('echo start HLRN run for $scriptname\n')
    if cluster == 'hlrn' or cluster == 'HLRN':
        f.write('FOLDER=${WORK}/${scriptname}/${PBS_JOBID}\n')
        f.write('mkdir -p ${FOLDER}\n\n')
        f.write('aprun -n 24 lmp_mpi -screen none -var rand ${seed} -var outFolder ${FOLDER} -var inFolder ${HOME} -in ${HOME}/${scriptname}.in &\n')

    elif cluster == 'rzcluster' or cluster == 'rz':
        f.write('FOLDER=${WORK}/${scriptname}\n')
        f.write('mkdir -p ${FOLDER}\n\n')
        f.write('mpirun -np 24 lmp_mpi -screen none -var rand ${seed} -var outFolder ${FOLDER} -var inFolder ${HOME} -in ${HOME}/${scriptname}.in &\n')

    f.write('done\nwait\n')
    f.write(
            'end=$(date +%s)\n'+
            'echo Beende hlrn run\n'+
            'secs=$((end-start))\n')
    f.write('echo Laufzeit:%s\n')
    f.write("printf '%dd:%dh:%dm:%ds' $(($secs/86400)) $(($secs%86400/3600)) $(($secs%3600/60)) $(($secs%60))\n")


def main():
    WriteHeader('hlrn', 12, 4, 24)
    WriteParameters(temp_P, temp_S, angle, pressure)
    # WriteLoop('hlrn', Temp_S=[190,300], Temp_P=[190,300])
    WritePressureLoop('hlrn', 0.52, 80, 300, [10, 40, 80, 160])

if __name__ == "__main__":
    main()

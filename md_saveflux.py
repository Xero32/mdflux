### md_saveflux.py
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
import argparse
import os
import time

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
angle, temp_S, temp_P, pressure = InpParams(parser)

""" Outline:

Read .lammpstrj file:
    save timestep in line following "ITEM: TIMESTEP"

    then save trajectory information from the line following
        "ITEM: ATOMS type id x y z ix iy iz vx vy vz c_PotEnergy"

    Create pandas dataframe with series
        time, id, x, y, z, ix, iy, iz, vx, vy, vx, pe

        make sure that time information is spread across multiple particles if needed

        Optional: particle tracking, to reassure that particle IDs are unique
        However, we are mainly interested in the density vs z-coordinate,
        so that individual trajectories are not of interest

    Analyze density profile in z direction at multiple timesteps (via pyplot, gnuplot,...?)

    Evaluate boundary between bounded and continuum states,
    count particles in bounded states in each timestep
    then graph number_of_bounded vs time

    maybe: account for incident particles inside 'bounded' territory by checking their vz component

    interesting analysis:
        outgong vs incoming particle flux
        resolve density for outgoing/incoming flux -> resolve this over time and you get saturation time
        machine learning for how lattice sites influence adsorption behavior ?

"""


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

#TODO pass parameters as input values in shell
# temp_S = 300
# temp_P = 300
# pressure = 1.0
# angle = 0.52

# construct directory/file name
_temp_S = str(temp_S)
_temp_P = str(temp_P)
_pr = str(pressure)
_pressure = NoPunctuation(pressure,1)
_angle = NoPunctuation(angle, 2)
parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"

HOME = "/home/becker"
infolder = HOME + "/lammps/flux/111/hlrn/" + parameter_set_str
jobs = (2118671, 2118672, 2126365, 2126363)
trajectories = np.arange(1,10)

# column_names = ['step', 'id', 'x','y','z','ix','iy','iz','vx','vy','vz','pe']
# df = pd.DataFrame(columns=column_names)

relative_step= id= x= y= z= ix= iy= iz= vx= vy= vz= pe = np.nan
 # df1 = pd.DataFrame([[relative_step, id,
 #                        x, y, z,
 #                        ix, iy, iz,
 #                        vx, vy, vz, pe]],
 #                        columns=column_names)

# for debugging:
trj = 1
jb = 2118671

def work(trj):
    column_names = ['traj', 'step', 'id', 'x','y','z','ix','iy','iz','vx','vy','vz','pe']
    # df = pd.DataFrame(columns=column_names)
    print(trj)
    for jb in jobs:
        ctr = 0
        start_time_step = 0
        directory = infolder + "/" + str(jb) + ".bbatch.hsn.hlrn.de/"
        if os.path.isdir(directory) == True:
            print(directory)
            traj_name = infolder + "/" + str(jb) + ".bbatch.hsn.hlrn.de/flux" + str(trj) + ".lammpstrj"
            f = open(traj_name, 'r')
        else:
            continue

        flag_timestep = False
        start = time.time()
        for i, line in enumerate(f):
            if 'ITEM:' in line:
                if 'TIMESTEP' in line:
                    flag_timestep = True
                    continue
                else:
                    continue

            elif float(line.split()[0]) != 2 and flag_timestep == False:
                continue

            if flag_timestep == True:
                tstep = line.split()[0]
                if i < 2:
                    start_time_step = tstep
                flag_timestep = False

            else:
                if len(line.split()) == 12:
                    type, id, x, y, z, ix, iy, iz, vx, vy, vz, pe = line.split()
                    pe = float(pe) * 2.
                else:
                    continue
                relative_step = int(tstep)-int(start_time_step)


                if ctr == 0:
                    df = pd.DataFrame([[int(trj), relative_step, int(id),
                            float(x), float(y), float(z),
                            int(ix), int(iy), int(iz),
                            float(vx), float(vy), float(vz), float(pe)]],
                            columns=column_names)
                    ctr += 1

                else:
                    df1 = pd.DataFrame([[int(trj), relative_step, int(id),
                            float(x), float(y), float(z),
                            int(ix), int(iy), int(iz),
                            float(vx), float(vy), float(vz), float(pe)]],
                            columns=column_names)
                    ctr += 1
                    df = pd.concat([df,df1], ignore_index=True)
            end = time.time()

        print("Iteration took " + str(end-start) + " seconds")
    if df.empty:
        print("Error! Cannot create Dataframe")
        return pd.DataFrame(colums=column_names)
    else:
        return df

max_traj = 24
num_jobs = 8
arg_inst = np.arange(1,max_traj+1)
results = Parallel(n_jobs=num_jobs, verbose=1, backend="threading")(map(delayed(work), arg_inst))
pddf = results[0]

for i in range(1,max_traj):
    pddf = pd.concat([pddf, results[i]], ignore_index=True)

column_names = ['traj', 'step', 'id', 'x','y','z','ix','iy','iz','vx','vy','vz','pe']

csv_file = "/home/becker/lammps/flux/" + parameter_set_str + ".csv"
pddf.to_csv(path_or_buf=csv_file, columns=column_names,
            index=False, mode='w')

### md_saveflux.py
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
import argparse
import os, sys
import time
from pathlib import Path

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

def Readfile(name, offset=0, procNum=4):
    print("Readfile")
    jobs = (2118671, 2118672, 2126365, 2126363)
    home = str(Path.home())
    infolder = home + "/lammps/flux/111/hlrn/" + name
    all_data = []
    for jb in jobs:
        path = infolder + "/" + str(jb) + ".bbatch.hsn.hlrn.de/"
        every_nth_file = []
        try:
            all_files = [f for f in os.listdir(path) if f.endswith('.lammpstrj')]
            for i, filename in enumerate(all_files):
                if (i-offset) % procNum == 0:
                    every_nth_file.append(filename)
        except:
            continue

        for f in every_nth_file:
            with open(path + "/" + f) as file:
                lines = file.read().split('\n')
            all_data.append(lines)
            del lines

    # print(len(all_data))
    return all_data

def Savedata(all_data, offset, procNum):
    print("Savedata")
    column_names = ['traj', 'step', 'id', 'x','y','z','ix','iy','iz','vx','vy','vz','pe']
    ctr = 0
    start_time_step = 0
    flag_timestep = False
    for trj, traj in enumerate(all_data):
        start = time.time()
        trajNum = trj * procNum + offset
        print(trajNum)

        start_time_step = int(traj[1])

        for i, line in enumerate(traj):
            try:
                aux = int(line.split()[0])
            except:
                continue

            if (aux % 1000 == 0 and len(line.split()) == 1):
                tstep = aux
            elif (len(line.split()) == 12):
                type, id, x, y, z, ix, iy, iz, vx, vy, vz, pe = line.split()
                pe = float(pe) * 2.

                relative_step = int(tstep)-int(start_time_step)

                if ctr == 0:
                    df = pd.DataFrame([[int(trajNum), relative_step, int(id),
                            float(x), float(y), float(z),
                            int(ix), int(iy), int(iz),
                            float(vx), float(vy), float(vz), float(pe)]],
                            columns=column_names)
                    ctr += 1

                else:
                    df1 = pd.DataFrame([[int(trajNum), relative_step, int(id),
                            float(x), float(y), float(z),
                            int(ix), int(iy), int(iz),
                            float(vx), float(vy), float(vz), float(pe)]],
                            columns=column_names)
                    ctr += 1
                    df = pd.concat([df,df1], ignore_index=True)
            else:
                continue

        end = time.time()
        print("Duration:", end-start, "seconds. Process:", offset)

    return df

def DivideAndConquer(name, off, pn): # Inputs are: parameter_set_str, offset, number of parallel processes
    dataArr = Readfile(name, off, pn)
    df = Savedata(dataArr, off, pn)
    return df

def main():
    parser = argparse.ArgumentParser()
    angle, temp_S, temp_P, pressure = InpParams(parser)
    home = str(Path.home())
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"

    num_jobs = 4
    Offset = np.arange(0,num_jobs)
    # Serial execution:
    # df = DivideAndConquer(parameter_set_str, 0, 1)

    # Parallel execution:
    data = Parallel(n_jobs=num_jobs, verbose=1, backend="threading")(delayed(Readfile)(name=parameter_set_str, offset=i, procNum=num_jobs) for i in Offset)
    frame = Parallel(n_jobs=num_jobs, verbose=1, backend="multiprocessing")(delayed(Savedata)(all_data=data[i], offset=i, procNum=num_jobs) for i in Offset)

    pddf = results[0]
    for i in range(1,num_jobs+1):
        pddf = pd.concat([pddf, frame[i]], ignore_index=True)
    # results = Parallel(n_jobs=num_jobs, verbose=1, backend="threading")(delayed(DivideAndConquer)(name=parameter_set_str, off=i, pn=num_jobs) for i in Offset)
    # TODO so far parallelization takes up to 30 seconds, whereas simple/single execution takes around 16 seconds
    print(pddf.describe())
    print(frame[0].describe())

    #  might be deprecated
    # relative_step= id= x= y= z= ix= iy= iz= vx= vy= vz= pe = np.nan

if __name__ == "__main__":
    main()

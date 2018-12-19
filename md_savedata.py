### md_saveflux.py
import numpy as np
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
    parser.add_argument("--beam", help="Choose whether Plasma Particles are to be parallel to each other", action='store_true')
    parser.add_argument("--nia", help="Save data for the non-interacting case", action='store_true')
    parser.add_argument("--pt", help="Save data for Argon on Platinum surface", action='store_true')

    args = parser.parse_args()
    if args.a:
        angle = float(args.a)
    if args.ts:
        temp_S = int(args.ts)
    if args.tp:
        temp_P = int(args.tp)
    if args.p:
        pressure = float(args.p)
    if args.beam:
        beamFlag = True
    else:
        beamFlag = False
    if args.nia:
        niaFlag = True
    else:
        niaFlag = False
    if args.pt:
        ptFlag = True
    else:
        ptFlag = False

    return angle, temp_S, temp_P, pressure, beamFlag, niaFlag, ptFlag

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

def Readfile(name, offset=0, procNum=4, max_i=1000, start_i=0):
    # This function reads '.lammpstrj' file data to simple arrays for further processing
    # Data is fully extracted from files to lessen memory interaction time
    print("Readfile")
    assert(procNum != 0)
    start = time.time()
    jobs = (2165358, 2151092, 2150174, 2118671, 2118672, 2135869, 2126363, 2135868, 2126365, 2135867, 2188758, 2203105, 2242959, 2242958, 2242957, 2244571)
    home = str(Path.home())
    infolder = home + "/lammps/flux/111/hlrn/" + name
    # infolder = home + "/lammps/flux/beam/" + name
    all_data = []
    for jb in jobs:
        path = infolder + "/" + str(jb) + ".bbatch.hsn.hlrn.de/"
        every_nth_file = []
        try:
            list_of_files = os.listdir(path)
            all_files = [f for f in list_of_files if f.endswith('.lammpstrj')]
            # Introduce splitting due to potential memory overload at low temp/high pressure
            i = start_i
            if len(list_of_files) < start_i+max_i:
                maximum = len(list_of_files) # - 1    # to omit last, incomplete traj file
            else:
                maximum = start_i+max_i

            while (i < maximum):
                if (i-offset) % procNum == 0:
                    every_nth_file.append(all_files[i])
                i += 1
        except:
            continue

        for f in every_nth_file:
            with open(path + "/" + f) as file:
                lines = file.read().split('\n')
            all_data.append(lines)
            del lines

        #end for (traj files)
        del every_nth_file

    #end for (jobs)
    print(time.time()-start, "seconds")

    return all_data

def Savedata(all_data, offset, procNum, start_traj=0):
    # This function extracts useful data from fct "Readfile", i.e. all_data
    # All data should be inside the RAM, significantly reducing computing time
    # A List of Trajectory data (i.e. completedata), which consists of multiple lines (timesteps) is returned
    print("Savedata")
    assert(procNum != 0)
    if start_traj != 0:
        offset = start_traj
    start = time.time()
    column_names = ['traj', 'step', 'id', 'x','y','z','ix','iy','iz','vx','vy','vz','pe']
    ctr = 0
    start_time_step = 0
    flag_timestep = False
    completedata = []
    trj = 0
    for traj in all_data:
        trajdata = []
        trajNum = trj * procNum + offset
        trj += 1
        start_time_step = int(traj[1])

        for line in traj:
            try:
                aux = int(line.split()[0])
            except:
                continue

            if (aux % 1000 == 0 and len(line.split()) == 1):
                tstep = aux
            elif (len(line.split()) == 12):
                type, id, x, y, z, ix, iy, iz, vx, vy, vz, pe = line.split()
                pe = float(pe) * 2.
                rs = int(tstep)-int(start_time_step)

                trajdata.append([int(trajNum), rs, int(id),
                                float(x), float(y), float(z),
                                int(ix), int(iy), int(iz),
                                float(vx), float(vy), float(vz), float(pe)])


            else:
                continue
        # end for (lines)
        completedata.append(trajdata)
        del trajdata

    # end for (traj)
    print(time.time()-start, "seconds")
    return completedata


def WriteListToCSV(Array, name, action, header=True):
    # Write all data from previous fcts "Readfile" and "Savedata" to csv file
    # The csv file can then be read by another module for data analysis
    print("Write to CSV")
    start = time.time()
    home = home = str(Path.home())
    csv_file = home + "/lammps/flux/" + name + ".csv"

    with open(csv_file, action) as f:
        if header == True:
            f.write("traj,step,id,x,y,z,ix,iy,iz,vx,vy,vz,pe\n")
        for traj in Array[:]:
            for line in traj:
                f.write("%d, %d, %d, %f, %f, %f, %d, %d, %d, %f, %f, %f, %f\n" %(tuple(line)))

    print(time.time()-start, "seconds")
    return 0

def DivideAndConquer(name, off, pn, max_i=1000, start_i=0): # Inputs are: parameter_set_str, offset, number of parallel processes
    dataArr = Readfile(name, off, pn, max_i, start_i)
    arr = Savedata(dataArr, off, pn, start_traj=start_i)
    del dataArr
    return arr

def SetParamsName(angle, temp_S, temp_P, pressure):
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"
    return parameter_set_str

def main():
    parallel = False
    parser = argparse.ArgumentParser()
    angle, temp_S, temp_P, pressure, beamFlag, niaFlag, ptFlag = InpParams(parser)
    home = str(Path.home())
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    if niaFlag:
        parameter_set_str = "nia"
    elif ptFlag:
        parameter_set_str = "Pt"
    else:
        parameter_set_str = ""

    parameter_set_str += SetParamsName(angle, temp_S, temp_P, pressure)
    if parallel == False:
        # Serial execution:
        # Split the work into multiple iterations to prevent memory overflow
        if (temp_S <= 190 and pressure >= 8.0):
            print("low temp, high pressure")
            maxI = 4

            arr = DivideAndConquer(parameter_set_str, 0, 1, max_i=maxI)
            WriteListToCSV(arr, parameter_set_str, 'w')
            del arr
            for i in range(1,5):
                startI = i * maxI
                arr = DivideAndConquer(parameter_set_str, 0, 1, max_i=maxI, start_i=startI)
                WriteListToCSV(arr, parameter_set_str, 'a', header=False)
                del arr
        elif (temp_S <= 190 and pressure > 2.0):
            print("low temp, medium pressure")
            maxI = 8
            startI = 0
            arr = DivideAndConquer(parameter_set_str, 0, 1, max_i=maxI)
            WriteListToCSV(arr, parameter_set_str, 'w')
            del arr
            for i in range(1,11):
                startI = i * maxI
                arr = DivideAndConquer(parameter_set_str, 0, 1, max_i=maxI, start_i=startI)
                WriteListToCSV(arr, parameter_set_str, 'a', header=False)
                del arr


        else:
            print("high temp, low pressure")
            maxI=500
            print(parameter_set_str)
            arr = DivideAndConquer(parameter_set_str, 0, 1, max_i=maxI)
            WriteListToCSV(arr, parameter_set_str, 'w')
            del arr
    else:
        # TODO
        # try and implement parallel execution with above splitting to avoid memory overflow
        # for small filesizes (high temp, low pressure) straightforward parallelization already works
        #

        assert(temp_S >= 190)
        assert(pressure <= 2.0)

        num_jobs = 4
        Offset = np.arange(0,num_jobs)

        data = Parallel(n_jobs=num_jobs, verbose=1, backend="threading")(
            delayed(Readfile)(name=parameter_set_str, offset=i, procNum=num_jobs) for i in Offset)
        array = Parallel(n_jobs=num_jobs, verbose=1, backend="multiprocessing")(
            delayed(Savedata)(all_data=data[i], offset=i, procNum=num_jobs) for i in Offset)
        arr = array[0]
        for i in range(1, num_jobs):
            arr.extend(array[i])
        WriteListToCSV(arr, parameter_set_str, 'w')
        del arr


if __name__ == "__main__":
    main()

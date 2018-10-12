# multipleplot.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

#####
Block = False
saveflag = True
quantity_dir = 'dens/'
# quantity = 'DensTimeGas'
quantity = sys.argv[2]
angle = 0.52
# temp_S = 300
temp_S = int(sys.argv[1])
temp_P = 300
pressure = -1.0 # set as variable for comparison
Params = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


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

def SetParamsName(angle, temp_S, temp_P, pressure):
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = ''
    if angle >= 0:
        parameter_set_str += "A" + _angle + "_"
    if temp_S >= 0:
        parameter_set_str += "TS" + _temp_S + "K_"
    if temp_P >= 0:
        parameter_set_str += "TP" + _temp_P + "K_"
    if pressure >= 0:
        parameter_set_str += "p" + _pressure + "datm"

    # parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"
    return parameter_set_str

def OpenFiles(files=[]):
    all_data = []
    header_data = []
    for file in files:
        try:
            with open(file, 'r') as f:
                lines = f.read().split('\n')
            all_data.append(lines)
            del lines
        except:
            continue
    return all_data


def ReadData(all_data):
    dataArr = []
    headerArr = []
    for filedata in all_data:

        xarray = []
        yarray = []
        for i, line in enumerate(filedata):
            if line.startswith("#"):
                auxline = line.split('#')[1]
                headerArr.append(auxline.split(','))

            else:
                try:
                    linedata = line.split(',')
                    xarray.append(float(linedata[0]))
                    yarray.append(float(linedata[1]))

                except:
                    _ = 1

        dataArr.append([xarray, yarray])
        del xarray
        del yarray

    return dataArr, headerArr

def PlotData(dataArr, headerArr, Params=[], block=False, save=False, write=False, savename=''):
    from plot import MakePlot, WritePlot

    for i in range(len(dataArr)):
        plt.plot(dataArr[i][:][0], dataArr[i][:][1], label=Params[i])

    MakePlot(saveflag=save, block=block, xlabel=headerArr[0][0], ylabel=headerArr[0][1], savepath=savename)







def main():
    global angle, temp_S, temp_P, pressure
    global Block, saveflag, quantity
    global Params

    ParamsLbl = []
    path = []
    home = home = str(Path.home())

    for par in Params:
        name = SetParamsName(angle, temp_S, temp_P, par)
        path.append(home + "/lammps/flux/plot/" + quantity_dir + name + quantity + '.csv')
        ParamsLbl.append(str(par) + " atm")
    data = OpenFiles(path)
    dataArr, headerArr = ReadData(data)

    newname = SetParamsName(angle, temp_S, temp_P, pressure)
    savename = home + "/lammps/flux/plot/" + quantity_dir + newname + "Compare" + quantity
    PlotData(dataArr, headerArr, Params=ParamsLbl, block=Block, save=saveflag, savename=savename)

if __name__ == "__main__":
    main()

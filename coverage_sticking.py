import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process, Value, Array
# from md_evalflux import SetParamsName
from pathlib import Path

def SetParamsName(angle, temp_S, temp_P, pressure):
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
    _temp_S = str(temp_S)
    _temp_P = str(temp_P)
    _pr = str(pressure)
    _pressure = NoPunctuation(pressure,1)
    _angle = NoPunctuation(angle, 2)
    parameter_set_str = "A" + _angle + "_TS" + _temp_S + "K_TP" + _temp_P + "K_p" + _pressure + "datm"
    return parameter_set_str

count = 0

# obtain the initital sticking probability of plasma atoms  on the surface
# for a given time moment (i.e. within a certain time range)
#
# caution! uses recursion
#
# df: dataframe with all trajectory data
# timeMin and timeMax define the time interval in which we want to search
# duration: duration of a collision event with direct reflection
# traj: current trajectory in which we look for collision events
# lastID: since we call this function recursively, we want to make sure that we don't count
#           the same reflection twice, hence we assure that the next found particle has a different ID than lastID
# particleCounter: number of all particles thet meet the requirements
#       (i.e. is recently spawned with downward  velocity in given trajectory)
def getSticking(df, timeMin, timeMax, duration, traj, lastID, particleCounter, reflectionCounter, depth):
    global count
    count += 2
    depth += 1

    bound = 5.0
    spawnMin = 48 # insertRegion starts 50 AA above surface, take some extra tolerance and make it 48
    spawnMax = 60
    cutoff = 12.0 # cutoff radius of Born-Mayer potential for Ar-Au interaction
    stw = 1000
    # duration = 80*stw # approx 20 ps from top to directly reflected should be a good value for our simulation


    # for one trajectory ('traj') find the particle with the highest ID
    # which has just been spawned.
    # then track this particle and decide whether it has reached a height > bound
    # after the defined duration


    # first get all particles in the time interval
    # within the spawn region with downward velocity
    particlesTracked = df.loc[(timeMin <= df['step']) & (df['step'] < timeMax) &
                        (df['traj'] == traj) & (df['z'] > spawnMin) & (df['vz'] < 0),
                        ['id', 'z', 'vz', 'pe','step']]

    # get particles within cutoff radius for Ar-Au interaction
    # particlesTracked = df.loc[(timeMin <= df['step']) & (df['step'] < timeMax) &
    #                     (df['traj'] == traj) & (df['z'] < spawnMin) & (df['vz'] < 0),
    #                     ['id', 'z', 'vz', 'pe','step']]
    # for this approach duration should be around 20*stw


    # if it happens to be that there is no particle recently spawned
    # we want to do the same check some time later
    if (particlesTracked['z'].count() == 0):
        timeNew = timeMin + 2 * stw
        # assure that we are still within our time range
        if(timeNew < timeMax):
            return getSticking(df, timeNew, timeMax, duration, traj, lastID, particleCounter, reflectionCounter, depth)
        else:
            # when we have left the time range, finally return the counter values
            return particleCounter, reflectionCounter, depth
    else:


        # scan the particles at the earliest time moment, and obtain the newest particle at this instance
        getMaxID = particlesTracked.loc[(particlesTracked['step'] == timeMin), ['id']]
        maxID = getMaxID['id'].max()
        countID = getMaxID['id'].count()

        if(maxID != lastID and countID != 0):
            # we have found a particle that meets our requirements,
            # and we have made sure that we don't double count the previous particle
            # add one to the total
            particleCounter += 1

            # if we have detected the same particle twice, simply omit this step
            # define timeNew and call the function again,
            # hoping that there has been another particle spawned

            # use this ID to track one specific particle
            particleToTrack = particlesTracked.loc[particlesTracked['id'] == maxID, ['id', 'z', 'vz', 'pe', 'step']]

            # need to construct a new dataframe from the original one to see where the particle is later on
            # but now we can use the information about the particle ID to track
            # and the approximate time where we have to look
            checkReflected = df.loc[(df['id'] == maxID) & (df['step'] == timeMin+duration) & (df['traj'] == traj) &
                                (df['z'] > bound) & (df['vz'] > 0),
                                ['id', 'z', 'vz', 'pe', 'step']]
            # if after the predestined duration the particle's height is above the bound region with an upward velocity
            # we consider the particle to be directly reflected
            if checkReflected['id'].count() != 0:
                reflectionCounter += 1


        timeNew = timeMin + 2*stw
        return getSticking(df, timeNew, timeMax, duration, traj, maxID, particleCounter, reflectionCounter, depth)

def getStickingData(angle, temp_S, temp_P, pressure):
    home = str(Path.home())
    columns = ['prob', 'dens']
    df = pd.DataFrame(columns=columns)
    covMeanDF = pd.DataFrame(columns=['t','dens'])

    # prob = np.asarray([])
    # dens = np.asarray([])


    paramsString = SetParamsName(angle, temp_S, temp_P, pressure)

    try:
        dir = '/lammps/'
        stickingName = home + dir + 'sticking' + paramsString + '.txt'
        stickingFile = open(stickingName, 'r')

        dir = "/lammps/flux/plot/dens/"
        coverageName = home + dir + paramsString + "DensTime.csv"
        coverageFile = open(coverageName, 'r')
    except:
        # print("Error")
        # print(temp_S)
        # print(pressure)
        # print(df.describe())

        return df

        #TODO
        # change values from coverage df
        # to average over time period!
    try:
        stickingdf = pd.read_csv(stickingFile, delim_whitespace=True,
            dtype={'t':np.float64, 'prob':np.float64}, names=['t','prob'],
            skiprows=1, header=None)
    except:
        stickingdf = pd.read_csv(stickingFile, sep=",",
            dtype={'t':np.float64, 'prob':np.float64}, names=['t','prob'],
            skiprows=1, header=None)

    coveragedf = pd.read_csv(coverageFile, sep=",",
        dtype={'t':np.float64, 'dens':np.float64}, names=['t', 'dens'],
        skiprows=1, header=None)


    stickingFile.close()
    coverageFile.close()


    delta_t = np.float64(stickingdf['t'][1] - stickingdf['t'][0])



    count = 0
    for t in stickingdf['t']:
        covValues = coveragedf.loc[(t-0.5*delta_t <= coveragedf['t']) & (coveragedf['t'] < t+0.5*delta_t), ['t','dens']]
        covMean = covValues['dens'].mean()
        auxList = [t, covMean]


        covMeanDF = covMeanDF.append(pd.Series(auxList, index=['t', 'dens']), ignore_index=True)


    # join both data sets, so that we get the sticking and coverage at the same time
    # Caution: this works only if both timescales are multiples of each other so that
    # one point in time is exactly contained in both dataframes.
    #
    # Especially since we compare floating point numbers, this may not always behave correctly
    # in that case, revert to a solution, where a time point from stickingdf is contained in a
    # time interval of coveragedf
    # fulldf = stickingdf.set_index('t').join(coveragedf.set_index('t'))

    # UDATE: implemented the aforementioned averaging:
    fulldf = stickingdf.set_index('t').join(covMeanDF.set_index('t'))

    df = df.append(fulldf)

    return df

def main():
    angle = 0.52
    temp_P = 300
    pressureList = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    dfList = []
    df = pd.DataFrame(columns=['prob', 'dens'])
    tempList = [80, 190, 300]
    markerList = ['o', 's', 'v']
    # hard coded. took sticking prob from single sim data
    singleStickingList = [0.7621509824198551, 0.6628029504741833, 0.5889830508474577]

    for i, temp_S in enumerate(tempList):
        for pressure in pressureList:
            dfList.append(getStickingData(angle, temp_S, temp_P, pressure))


    colorList = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    maxVal = 9

    for i, temp in enumerate(tempList):
        for j, pressure in enumerate(pressureList):
            idx = j + i * len(pressureList)
            if dfList[idx]['dens'].count() == 0:
                continue

            if j == 0:
                label = str(temp)
            else:
                label = ""
            plt.scatter(dfList[idx]['dens'], dfList[idx]['prob'], label=label, marker=markerList[i], c=colorList[i])
            # , c=colorList[i])
            # compareRange = np.arange(dfList[idx]['dens'].min(), dfList[idx]['dens'].max(), 0.1)
            compareRange = np.arange(0, maxVal+1, 1)

        plt.plot(compareRange, [singleStickingList[i] for j,_ in enumerate(compareRange)], ':', c=colorList[i])
    plt.xlabel(r"$\rho$ / nm$^{-2}$")
    plt.ylabel("Sticking Probability")
    plt.axis((0,maxVal, 0.45, 0.96))
    plt.legend()
    plt.tight_layout()
    plt.savefig("stickCovDependent.pdf")
    plt.show()







if __name__ == "__main__":
    main()

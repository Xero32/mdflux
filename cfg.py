#cfg.py
import matplotlib
from pathlib import Path
TIMESTEPS = 0
TRAJ_COUNTER = 0
HLRN_COUNTER = 0
HLRN = 0
NUM_OF_JOBS = 0
jobs = ()
nbnc = 0
HOME = str(Path.home())
# HOME = '/home/becker'

kB = 1.3806503e-23
e0 = 1.60217662e-19
pi = 3.14159265359

TIMESTEPS_RZ = 120000 // 100
TIMESTEPS_HLRN = 160000 // 100
NUM_OF_TRAJ_RZ = 200
NUM_OF_TRAJ = 200
matplotlib.rcParams.update({'font.size': 16})
## Plot, Save Write flags
#Plot

P_POP = 0
P_TRANS = 1
P_T = 1
P_T_AVG = 0 # set if averaged transition rates shall be plotted
P_ANG = 0
P_SOL = 1
P_TBETW = 0
P_DECR = 1
P_NRG = 0
#set all: if not needed, comment line
P_POP = P_TRANS = P_T = P_SOL = P_ANG = P_TBETW = P_DECR = P_NRG = 0

#Save
S_POP = 0
S_TRANS = 0
S_T = 1
S_SOL = 0
S_ANG = 0
S_TBETW = 0
S_DECR = 0
S_NRG = 0
#set all: if not needed, comment line
S_POP = S_TRANS = S_T = S_SOL = S_ANG = S_TBETW = S_DECR = S_NRG = 1

#Write
W_POP = 0
W_TRANS = 0
W_T = 0
W_SOL = 0
W_ANG = 0
W_TBETW = 0
W_DECR = 0
W_NRG = 0
W_STICK = 0
W_PARAM = 0
#Smooth
G_POP = 1
G_TRANS = 1
G_T = 1
G_SOL = 0
G_ANG = 1
G_TBETW = 0
G_DECR = 0
G_NRG = 1

#when we want to save, we need to call plot functions
if S_POP != 0:
    P_POP = 1
if S_TRANS != 0:
    P_TRANS = 1
if S_T != 0:
    P_T = 1
if S_SOL != 0:
    P_SOL = 1
if S_ANG != 0:
    P_ANG = 1
if S_TBETW != 0:
    P_TBETW = 1
if S_DECR != 0:
    P_DECR = 1
if S_NRG != 0:
    P_NRG = 1

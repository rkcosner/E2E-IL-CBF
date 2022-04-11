import matplotlib.pyplot as plt 
import numpy as np 
from controller_util import * 
from tqdm import tqdm 

path = "/home/ryan/Documents/TripleCamera/AirSim/PythonClient/cdc22/experiments/plots0_m4_mpi"
x_traj = np.load(path + "/x_traj.np.npy")

hs = []
for x in tqdm(x_traj):
    x[1,0] -=3
    _, h = raceTrackBarrierBits(x)
    hs.append(h)


hs = np.array(hs)
fig = plt.figure()
plt.plot(hs)
fig.savefig("plots/hs.png")
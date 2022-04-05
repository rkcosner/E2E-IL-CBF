import numpy as np
import csv
import matplotlib.pyplot as plt
from cosner_utils.utils import getEllipse, saveFigure
import pandas as pd
from cosner_utils.ode_sim import  *

path = "/home/rkcosner/Documents/Research/CDC22/data/dataOut/inverted_pendulum_ics/l2_ics"

test = InvertedPendulum(np.array([[0],[0]]))


if __name__ == "__main__":
    ex, ey = getEllipse()

    fig = plt.figure()
    plt.plot(ex,ey, 'g')
    N = 86
    for i in range(86):
        csv = pd.read_csv(path +"/ic" +str(i) + "/csvs/states.csv", header = None, usecols=[0,1])
        x = csv[0].tolist()
        y = csv[1].tolist()
        plt.plot(x,y, 'y')


    for i in range(86):
        csv = pd.read_csv(path + "/ic" + str(i) + "/csvs/states.csv", header=None, usecols=[0, 1])
        x = csv[0].tolist()
        y = csv[1].tolist()
        plt.plot(x[0], y[0], 'b^')

    plt.title("Trajectories for gridded x_0s using IL controller \n with samples only on the boundary")
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")

    plt.show()

    saveFigure(fig, "ip_trop_ics.pdf", "figures")


    fig = plt.figure()
    minh = 100
    for i in range(86):
        csv = pd.read_csv(path +"/ic" +str(i) + "/csvs/states.csv", header = None, usecols=[0,1])
        x = csv[0].tolist()
        y = csv[1].tolist()
        hs = []
        for i in range(len(x)):
            state = np.array([[x[i]], [y[i]]])
            hs.append(test.getSafetyVal(state)[0,0])
            if test.getSafetyVal(state)[0,0] < minh:
                minh = test.getSafetyVal(state)[0,0]
        ts = np.linspace(0, 1, 101 )
        plt.plot(ts[0:-1], hs, color=[0.75, 0.75, 1], linewidth=2)


    i = 4
    csv = pd.read_csv(path + "/ic" + str(i) + "/csvs/states.csv", header=None, usecols=[0, 1])
    x = csv[0].tolist()
    y = csv[1].tolist()
    hs = []
    print(x[0], y[0])
    for i in range(len(x)):
        state = np.array([[x[i]], [y[i]]])
        hs.append(test.getSafetyVal(state)[0, 0])
        if test.getSafetyVal(state)[0, 0] < minh:
            minh = test.getSafetyVal(state)[0, 0]
    ts = np.linspace(0, 1, 101)
    plt.plot(ts[0:-1], hs, color="b", linewidth=2)


    plt.hlines(ts[0], ts[-1], 0, linestyle="--", color="r", linewidth=2)
    plt.ylim((-0.1, 1.15))
    plt.title("min h = " + str(minh))
    plt.savefig("ip_h_traj.pdf")

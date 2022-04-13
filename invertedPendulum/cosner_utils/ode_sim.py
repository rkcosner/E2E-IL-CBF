import os

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from cosner_utils import utils
from controllers import K_CBF_SOCP, raceTrackBarrierBits
pi = np.pi

class System:
    def __init__(self, x0, n, m):
        self.x0 = x0
        self.t0 = 0
        self.dt = 0.001
        self.tend = 10
        self.state_dim = n
        self.input_dim = m

        self.x_traj = []
        self.u_traj = []
        self.t_traj = []
        self.h_traj = []

    def reset(self):
        self.x_traj = []
        self.u_traj = []
        self.t_traj = []
        self.h_traj = []

    def drift_f(self, x):
        n = self.state_dim
        f = np.zeros((n,1))
        return f

    def input_matrix_g(self, x):
        n = self.state_dim
        m = self.input_dim
        g = np.zeros((n,m))
        return g

    def controller(self, x):
        m = self.input_dim
        input = np.zeros((m,1))
        return input

    def dynamics(self, t, x):
        n = self.state_dim
        x = np.reshape(x, (n,1))
        f = self.drift_f(x)
        g = self.input_matrix_g(x)
        u, _ = self.controller(x)
        xdot = f + g@u
        return xdot

    def simulate(self):
        dt = self.dt
        ode_integrator = ode(self.dynamics)
        ode_integrator.set_initial_value(self.x0, self.t0)
        while ode_integrator.successful() and ode_integrator.t < self.tend:

            ode_integrator.integrate(ode_integrator.t+dt)
            ## data logging
            t = ode_integrator.t
            x = ode_integrator.integrate(ode_integrator.t)
            u, _ = self.controller(ode_integrator.integrate(ode_integrator.t))
            h = self.getSafetyVal(x)
            self.t_traj.append(t)
            self.x_traj.append(x[:, 0].tolist())
            self.u_traj.append(u[:, 0].tolist())
            self.h_traj.append(h[0, 0].tolist())

            ###############
            print("Sim time: ", t)
            if  h[0,0] < -0.001:
                print("\t safety violation, h = ", h[0,0])


    def simulateEulerStep(self):
        dt = self.dt
        t = 0
        x = self.x0
        while t < self.tend:
            ## data logging
            u,_ = self.controller(x)
            h = self.getSafetyVal(x)
            self.t_traj.append(t)
            self.x_traj.append(x[:, 0].tolist())
            self.u_traj.append(u[:, 0].tolist())
            self.h_traj.append(h[0, 0].tolist())
            ###############
            x += dt*self.dynamics(t,x)
            t += dt

            print("Sim time: ", t)
            if  h[0,0] < -0.001:
                print("\t safety violation, h = ", h[0,0])


    def plot(self, save_path):
        n = self.state_dim
        m = self.input_dim

        x_traj = np.array(self.x_traj)
        state_fig, state_axs = plt.subplots(n)
        for i in range(n):
            state_axs[i].plot(self.t_traj, x_traj[:,i])
            state_axs[i].set_ylabel("state " + str(i))
        state_fig.suptitle("States")
        state_axs[n-1].set_xlabel("Time [sec]")

        u_traj = np.array(self.u_traj)
        input_fig, input_axs = plt.subplots(m)
        if m > 1:
            for i in range(m):
                input_axs[i].plot(self.t_traj, u_traj[:,i])
                input_axs[i].set_ylabel("input " + str(i))
            input_axs[m - 1].set_xlabel("Time [sec]")
        else:
            input_axs.plot(self.t_traj, u_traj[:, 0])
            input_axs.set_ylabel("input " + str(0))
            input_axs.set_xlabel("Time [sec]")
        input_fig.suptitle("Inputs")

        plt.figure()
        plt.title("CBF val")
        plt.plot(self.t_traj, self.h_traj)
        plt.hlines(0, self.t_traj[0], self.t_traj[-1], linestyle = '--')
        plt.xlabel("time")
        plt.show()

        plt.show()

        os.system("mkdir " + save_path)
        utils.saveFigure(state_fig, "state_subplots.png", save_path)
        utils.saveFigure(input_fig, "input_subplots.png", save_path)

    def saveData(self, save_path):
        os.system("mkdir " + save_path + "/csvs/")
        np.savetxt(save_path+"/csvs/states.csv", self.x_traj, delimiter=",")
        np.savetxt(save_path+"/csvs/inputs.csv", self.u_traj, delimiter=",")
        np.savetxt(save_path+"/csvs/times.csv", self.t_traj, delimiter=",")
        np.savetxt(save_path+"/csvs/times.csv", self.h_traj, delimiter=",")

class Unicycle(System):
    def __init__(self, x0):
        """    print(len(test.t_traj))
    print(test.x_traj)
        System States are:
            [x,y,theta]
        System Inputs are:
            [v, omega]
        """
        System.__init__(self, x0, 3, 2)

    def input_matrix_g(self, x):
        n = self.state_dim
        m = self.input_dim
        g = np.zeros((n,m))
        g[0,0] = np.cos(x[2,0])
        g[1,0] = np.sin(x[2,0])
        g[2,1] = 1
        return g

    def controller(self, x):
        m = self.input_dim
        u = np.zeros((m,1))
        u[0,0] = 1
        u[1,0] = 0.2
        return u

    def raceTrackController(self, x):
        m = self.input_dim
        u = np.zeros((m,1))
        u[0,0] = 1
        u[1,0] = 0.5
        return u

    def moduloSpin(self):
        x_traj = np.array(self.x_traj)
        x_traj[:,2] = x_traj[:,2]%(2*np.pi)
        self.x_traj = x_traj.tolist()

    def plotPlanar(self, save_path):
        n = self.state_dim
        m = self.input_dim

        x_traj = np.array(self.x_traj)
        fig = plt.figure()
        plt.plot(x_traj[:,0], x_traj[:,1])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Planar Motion")
        plt.show()
        utils.saveFigure(fig, "planar_states.png", save_path)

    def getSafetyVal(self, x):
        bits = raceTrackBarrierBits(x)
        hs = []
        for b in bits:
            hs.append(b[2])

        h = np.array([[min(hs)]])
        return h

class InvertedPendulum(System):
    def __init__(self, x0):
        """    print(len(test.t_traj))
            print(test.x_traj)
        System States are:
            [x,y,theta]
        System Inputs are:
            [v, omega]
        """
        System.__init__(self, x0, 2, 1)

    def drift_f(self, x):
        n = self.state_dim
        f = np.zeros((n,1))
        f[0,0] = x[1,0]
        f[1,0] = np.sin(x[0,0])
        return f

    def input_matrix_g(self, x):
        n = self.state_dim
        m = self.input_dim
        g = np.zeros((n,m))
        g[0,0] = 0
        g[1,0] = 1
        return g

    def controller(self, x):
        m = self.input_dim
        u = np.zeros((m,1))
        k = 0.75
        u[0,0] = -k*x[0,0]
        return u, True

    def controllerDes(self, x):
        m = self.input_dim
        u = np.zeros((m,1))
        k = 0.75
        u[0,0] = -k*x[0,0]
        return u, True

    def fbLinController(self, x):
        m = self.input_dim
        u = np.zeros((m,1))
        Kp = 1
        Kd = 1
        u[0,0] = -np.cos(x[0,0]) - (Kp*x[0,0] + Kd*x[1,0])
        return u, True

    def ip_clf(self, x):
        m = 1
        u = np.zeros((m, 1))
        k1 = 2
        k2 = 2

        # P = np.array([[np.sqrt(3), 1], [1, np.sqrt(3)]])
        u[0, 0] = -np.cos(x[0, 0]) - k1 * x[0, 0] - k2 * x[1, 0]

        return u, True

    def cbfqp(self, x):
        m = self.input_dim
        u = np.zeros((m,1))
        P = np.array([[np.sqrt(3), 1], [1, np.sqrt(3)]])
        # V = x'Px
        c = np.sqrt(3)*(45*pi/180)**2 # 45 degrees
        alpha = 1
        g = self.input_matrix_g(x)
        f = self.drift_f(x)
        h = c - x.T@P@x
        Lgh = -2*x.T@P@g
        Lfh = -2*x.T@P@f
        if Lfh >= -alpha*h:
            return u, True
        else:
            return Lgh.T/(Lgh@Lgh.T)*(-Lfh - alpha*h), True


    def getSafetyVal(self, x):
        P = np.array([[np.sqrt(3), 1], [1, np.sqrt(3)]])
        c = np.sqrt(3)*(45*pi/180)**2 # 45 degrees
        h = c - x.T@P@x
        return h

    def trop(self, x):
        m = self.input_dim
        u = np.zeros((m,1))
        P = np.array([[np.sqrt(3), 1], [1, np.sqrt(3)]])
        # V = x'Px
        c = np.sqrt(3)*(45*pi/180)**2 # 45 degrees
        alpha = 1
        g = self.input_matrix_g(x)
        f = self.drift_f(x)
        h = c - x.T@P@x
        Lgh = -2*x.T@P@g
        Lfh = -2*x.T@P@f
        u_des, _ = self.controllerDes(x)
        u, flag = K_CBF_SOCP([[Lfh, Lgh, h]], u_des, alpha=1, sigma=2, MRCBF_add=0.027, MRCBF_mult=0.04)

        return u, flag


    def moduloSpin(self):
        x_traj = np.array(self.x_traj)
        x_traj[:,0] = (x_traj[:,0]+pi)%(2*pi) - pi
        self.x_traj = x_traj.tolist()



    def plotPlanar(self, save_path, ellipse_x = [], ellipse_y = []):
        n = self.state_dim
        m = self.input_dim

        x_traj = np.array(self.x_traj)
        fig = plt.figure()
        plt.plot(x_traj[:,0], x_traj[:,1])
        plt.plot(x_traj[0,0], x_traj[0,1], '^')
        plt.plot(ellipse_x, ellipse_y, 'g')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r"$\dot{\theta}$")
        plt.xlim((-1.3,1.3))
        plt.ylim((-1.3,1.3))
        plt.title("Phase Portrait")
        plt.show()
        utils.saveFigure(fig, "planar_states.png", save_path)




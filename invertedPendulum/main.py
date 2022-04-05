import numpy as np
from datetime import datetime
from cosner_utils.utils import *
from cosner_utils.ode_sim import *
from cosner_utils.plotting_utils import *
from cosner_utils.pendulum_visualizer import *
from torchvision.utils import save_image
from controllers import *
from ipMobileNetV2 import *
import torch
from tqdm import tqdm
import csv

def printInfo(timestamp):
    print("Hi!")
    print("Running the CDC 2022 script with datetime label :", timestamp)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    Set up Data Logging
    """
    timestamp = getTimeStamp()
    printInfo(timestamp)

    # Run Race Car
    x0 = np.array([[-0.5], [-1.84154943], [0.0]])
    test = Unicycle(x0)
    # test.controller = raceTrackController
    # test.controller = raceTrackCBFQP
    test.controller = raceTrackTROP
    if False:
        # Set Up Unicycle Logging
        data_output_path = createDataOutputFolder("racecar", timestamp)

        # Run Unicycle Simulation
        test.tend = 15
        test.dt = 0.005
        test.simulate = test.simulateEulerStep
        test.simulate()
        test.moduloSpin()
        test.plot(data_output_path)
        test.plotPlanar(data_output_path)
        test.saveData(data_output_path)

        # Plot track
        x_traj = np.array(test.x_traj)
        plotTrack(x_traj[:,0], x_traj[:,1], l, w, data_output_path)


    # Sample for Car
    if False:
        socp_failure = 0
        r = l/4/pi

        r_density = 0.1
        thetasCar = np.linspace(0, 2 * pi, int(2*pi/r_density))

        # Inner Boundary
        ls = np.linspace(0, l, int(l/r_density))
        xs = []
        ys = []
        samplePoints = []

        for lStep in ls:
            for thC in thetasCar:
                if lStep <=l/4:
                    dotval = np.sqrt(turn_delta * (np.sin(thC) + 1) + (r ) ** 2) - (r)
                    x = lStep - l/8
                    y = -l/4/pi-dotval
                elif lStep <= l/2:
                    thT = (lStep-l/4)/l*4*pi - pi/2
                    dotval = np.sqrt(turn_delta * (np.sin(thC+thT) + 1) + (r ) ** 2) - (r )
                    x = l/8 + (l/4/pi+dotval)*np.cos(thT)
                    y = (l/4/pi+dotval)*np.sin(thT)
                elif lStep <= 3*l/4:
                    dotval = np.sqrt(turn_delta * (np.sin(thC+pi) + 1) + (r ) ** 2) - (r )
                    x = l/2 - lStep + l/8
                    y = l/4/pi+dotval
                elif lStep <= l:
                    thT = (lStep-l*3/4)/l*4*pi + pi/2
                    dotval = np.sqrt(turn_delta * (np.sin(thC+thT+pi) + 1) + (r) ** 2) - (r )
                    x = -l/8 + (l/4/pi+dotval)*np.cos(thT)
                    y =(l/4/pi+dotval)*np.sin(thT)
                state = np.array([[x], [y], [thC]])
                xs.append(x)
                ys.append(y)
                samplePoints.append([x,y,thC])

        # Outer Boundary
        ls = np.linspace(0, l/2 + 2*l/4+2*w*pi, int((l/2 + 2*l/4+2*w*pi)/r_density))

        for lStep in ls:
            for thC in thetasCar:
                if lStep <=l/4:
                    dotval = np.sqrt(turn_delta * (np.sin(thC) + 1) + (r +w) ** 2) - (r+w)
                    x = lStep - l/8
                    y = -l/4/pi-w-dotval
                elif lStep <= l/4 + l/4+w*pi:
                    thT = (lStep-l/4)/(l/4+w*pi)*pi - pi/2
                    dotval = np.sqrt(turn_delta * (np.sin(thC+thT) + 1) + (r +w) ** 2) - (r +w)
                    x = l/8 + (l/4/pi+w-dotval)*np.cos(thT)
                    y = (l/4/pi+w-dotval)*np.sin(thT)
                elif lStep <=  l/2 + l/4+w*pi:
                    dotval = np.sqrt(turn_delta * (np.sin(thC+pi) + 1) + (r +w) ** 2) - (r +w)
                    x = l/4 + l/4+w*pi - lStep + l/8
                    y = l/4/pi+w-dotval
                elif lStep <= l/2 + 2*l/4+2*w*pi:
                    thT = (lStep - (l/2 + l/4+w*pi))/(l/4+w*pi)*pi + pi/2
                    dotval = np.sqrt(turn_delta * (np.sin(thC+thT+pi) + 1) + (r+w) ** 2) - (r +w)
                    x = -l/8 + (l/4/pi + w  -dotval)*np.cos(thT)
                    y =(l/4/pi+w -dotval)*np.sin(thT)
                state = np.array([[x], [y], [thC]])
                xs.append(x)
                ys.append(y)
                samplePoints.append([x,y,thC])

                state = np.array([[x], [y], [thC]])
                u, flag = raceTrackTROP(state)
                if not flag:
                    socp_failure += 1

        plt.figure()
        plt.plot(xs, ys, '.')
        thetas = np.linspace(0,2*pi,1000)
        rs = outer_loop(thetas,l_fulltrack, w)
        xs, ys = polar_to_xy(thetas, rs)
        plt.plot(xs, ys)
        rs = outer_loop(thetas,l_fulltrack, 0)
        xs, ys = polar_to_xy(thetas, rs)
        plt.plot(xs, ys)
        plt.show()
        print("SOCP failures : ", socp_failure)
        print("Failures are occuring when parallel to boundary")

        print("saving data to ./data/dataOut/racecarSamplePoints.np")
        samplesOut = np.array(samplePoints)
        np.save("./data/dataOut/racecarSamplePoints_r"+str(r_density), samplesOut)

    # Add Controller to Datset
    if False:

        path = "./data/dataIn/"
        csv_filename = "data.csv"
        csv_filename_new = "controller_data.csv"
        socp_failure = 0

        y_offset = 0

        os.system("touch " + path + csv_filename_new)
        with open(path + csv_filename_new, "w", newline='\n') as newcsvfile:
            fieldnames = ["i", "x", "y", "theta", "u1", "u2"]
            writer = csv.DictWriter(newcsvfile, fieldnames=fieldnames)
            writer.writeheader()

            with open(path + csv_filename, newline='\n') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row_idx, row in tqdm(enumerate(reader)):
                    if row_idx > 0:
                        state = np.array([[float(row[1])], [float(row[2]) - y_offset], [float(row[3])]])
                        u, flag = raceTrackTROP(state)
                        if not flag:
                            socp_failure += 1
                        writer.writerow(
                            {"i": row[0],  "x": row[1], "y": float(row[2]) - y_offset, "theta": row[3],
                             "u1": u[0, 0], "u2": u[1, 0]})

        print("SOCP failures : ", socp_failure)


    # Run Inverted Pendulum
    if True:
        x0 = np.array([[-0.3], [0]])
        test = InvertedPendulum(x0)
        ip = PendulumEnv()
        convert_tensor = transforms.ToTensor()
        data_output_path = createDataOutputFolder("inverted_pendulum", timestamp)
        model_path = "/home/rkcosner/Documents/Research/CDC22/data/trainedModels/model_20220323_223709_9.pth"
        e2e_model = MobileNetV2(width_mult = 1, states = 1, vel_layer_width = 200)
        e2e_model.load_state_dict(torch.load(model_path))
        e2e_model.eval()

        if False:
            # Set Up IP Logging

            # Run IP Simulation
            test.controller = ip.trop#test.trop#test.cbfqp #test.ip_clf # test.fbLinController
            test.tend = 10
            test.simulate()
            test.moduloSpin()
            test.plot(data_output_path)
            test.plotPlanar(data_output_path)
            test.saveData(data_output_path)
        # Create Inverted Pendulum Data Set
        if False:
            c = np.sqrt(3)*(45*pi/180)**2 # 45 degrees
            x_min = -45/180*pi-0.1765
            x_max = -x_min
            x_now = x_min
            x = []
            y = []

            r = 0.009

            x_last = x_now
            y_last =  (-2*x_now+np.sqrt(4*np.sqrt(3)*(c - np.sqrt(3)*x_now**2) + 4*x_now**2))/2/np.sqrt(3)
            x.append(x_last)
            y.append(y_last)
            rs = []
            i = 0
            while x_now < x_max:
                y_now =  (-2*x_now+np.sqrt(4*np.sqrt(3)*(c - np.sqrt(3)*x_now**2) + 4*x_now**2))/2/np.sqrt(3)

                if ((x_now-x_last)**2 + (y_now-y_last)**2) > r**2:
                    x.append(x_now)
                    y.append(y_now)
                    rs.append(np.sqrt((x_now-x_last)**2 + (y_now-y_last)**2))
                    x_last = x_now
                    y_last = y_now
                x_now += 0.00005
                i+=1

            x = np.array(x)
            x = np.hstack([x,-x])
            y_bottom = -np.array(y)
            y = np.hstack([y,y_bottom])

            plt.figure()
            plt.plot(x,y, '.')
            plt.show()
            # Visualize
            path = "./data/trainingData/training_set_r_" + str(max(rs))
            os.system("mkdir " + path )
            for i in range(len(x)):
                theta = x[i]
                ip.state = np.array([np.sin(theta), np.cos(theta), y[i]])
                ip.last_u = None
                frame = ip.render(mode="rgb_array")
                img = Image.fromarray(frame)
                img = preprocess(img)
                save_image(img, path + "/img_"+ str(i).zfill(4) + ".png")

            os.system("mkdir " + path + "/csv")
            with open(path + "/csv/data.csv", "w", newline="\n") as csvfile:
                fieldnames = ["i", "theta", "theta_dot", "u"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for i in range(len(x)):
                    theta = x[i]
                    theta_dot = y[i]
                    state = np.array([[theta], [theta_dot]])
                    u = test.trop(state)
                    writer.writerow({"i": str(i).zfill(4), "theta": x[i], "theta_dot": y[i], "u": u[0,0]})

            # compare results
            if False:

                def e2e_controller(x):
                    theta = x[0,0]
                    theta_dot = x[1,0]
                    ip.state = np.array([np.sin(theta), np.cos(theta), theta_dot])
                    ip.last_u = None
                    frame = ip.render(mode="rgb_array")
                    img = Image.fromarray(frame)
                    img = preprocess(img)
                    save_image(img, "current_img.png")

                    img_gen = Image.open("current_img.png")
                    img_gen = convert_tensor(img_gen)

                    theta_dot = torch.tensor([theta_dot], dtype=torch.float32).unsqueeze(0)
                    u = e2e_model(img_gen.unsqueeze(0), theta_dot)
                    u = u.cpu().detach().numpy()
                    return u


                err = []
                ues = []
                uls = []

                for i in range(len(x)):
                    state = np.array([[x[i]], [y[i]]])
                    ue = test.trop(state)
                    ul = e2e_controller(state)

                    ues.append(ue[0,0])
                    uls.append(ul[0,0])

                plt.figure()
                plt.plot(ues)
                plt.plot(uls)
                plt.show()
        # Test Inverted Pendulum
        if False:


            convert_tensor = transforms.ToTensor()


            def e2e_controller(x):
                theta = x[0]
                theta_dot = x[1]
                print("angle =", theta * 180 / pi)
                ip.state = np.array([np.sin(theta), np.cos(theta), theta_dot])
                ip.last_u = None
                frame = ip.render(mode="rgb_array")
                img = Image.fromarray(frame)
                img = preprocess(img)
                # save_image(img, "current_img.png")
                #
                # img = Image.open("current_img.png")
                # img = convert_tensor(img)
                theta_dot = torch.tensor([theta_dot], dtype=torch.float32)
                u = e2e_model(img.unsqueeze(0), theta_dot)
                u = u.cpu().detach().numpy()
                # print("input = ", u)
                return u

            test.controller = e2e_controller #test.trop#
            test.tend = 5
            test.dt = 0.01
            test.simulate = test.simulateEulerStep
            test.simulate()
            test.moduloSpin()
            test.plot(data_output_path)
            test.plotPlanar(data_output_path)
            test.saveData(data_output_path)
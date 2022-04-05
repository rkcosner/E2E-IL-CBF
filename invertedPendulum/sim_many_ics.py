import numpy as np
from datetime import datetime
from cosner_utils.utils import *
from cosner_utils.ode_sim import *
from cosner_utils.plotting_utils import *
from cosner_utils.pendulum_visualizer import *
from cosner_utils.utils import getEllipse
from torchvision.utils import save_image
from controllers import *
from ipMobileNetV2 import *
import torch
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

    # Run Inverted Pendulum
    x0 = np.array([[-0.3], [0]])
    test = InvertedPendulum(x0)
    ip = PendulumEnv()
    convert_tensor = transforms.ToTensor()
    data_output_path = createDataOutputFolder("inverted_pendulum_ics", timestamp)
    model_path = "/home/rkcosner/Documents/Research/CDC22/data/trainedModels/bestl2_3_23.pth"
    e2e_model = MobileNetV2(width_mult=1, states=1, vel_layer_width=200)
    e2e_model.load_state_dict(torch.load(model_path))
    e2e_model.eval()


    convert_tensor = transforms.ToTensor()


    def e2e_controller(x):
        theta = x[0, 0]
        theta_dot = x[1, 0]
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

    test.controller = test.trop#e2e_controller #test.trop#
    test.tend = 1
    test.dt = 0.01
    test.simulate = test.simulateEulerStep

    xs = np.linspace(-1,1,13)
    ys = np.linspace(-1,1,13)


    # Get ellipse
    ex, ey = getEllipse()


    iter = 0
    for x in xs:
        for y in ys:
            if test.getSafetyVal(np.array([[x],[y]]))>=0:
                os.system("mkdir " + data_output_path +"/ic" + str(iter))
                x0 = np.array([[x], [y]])
                test.x0 = x0
                test.reset()
                test.simulate()
                test.moduloSpin()
                test.plotPlanar(data_output_path+"/ic" + str(iter), ex, ey)
                test.saveData(data_output_path+"/ic" + str(iter))

                iter +=1
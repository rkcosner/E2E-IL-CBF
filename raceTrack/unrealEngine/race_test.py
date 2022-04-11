import airsim #pip install airsim
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import time
import os 
from datetime import datetime
import csv 
from tqdm import tqdm 
pi = np.pi

import torch 
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from plotting_utils import * 
from myMobileNetV2 import * 
from controller_util import * 
convert_tensor = transforms.ToTensor()


SAVE_RGB = True
OVERWRITE_DATA = False
END2END = True 
ONEOFF = True 
oneoff_idx = 1

if not END2END: 
    prepend = "trop"
else: 
    prepend = ""

"""
Load the model
"""
model_path = "/home/ryan/Documents/CDC22/notebooks/SaveMobNetModels/20220325_145128/model_20220325_145128_17.pth"
end2endController = MobileNetV2(width_mult = 1, states=2)
end2endController.load_state_dict(torch.load(model_path))
end2endController.eval()

# Makes the file output size waaaaay smaller
TORCH_IMAGE = True

print("Setting up preprocessing")
#
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


print("Connecting to Airsim")

# for car use CarClient() 
client = airsim.MultirotorClient()

def tic(): 
    return time.time()

def toc(t, statement):
    elapsed = time.time() - t
    print(statement, " ", elapsed)

def convertresponse(response,i ): 
    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(response.height, response.width, 3)
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    if SAVE_RGB: 
        # img = cv2.cvtColor(img*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./plots/imgs_traj/converted_img_rgb"+str(i) + ".png", img_rgb)
    saveFigure(fig, "imgs_traj/img_current.png", ".")
    plt.close()
    if TORCH_IMAGE: 
        img_rgb = Image.fromarray(np.uint8(img_rgb)).convert("RGB")
        img_rgb = preprocess(img_rgb)
    return img_rgb 

def get_image(): 
    responses = client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.Scene, False, False)])
    return responses[0], responses[0].time_stamp


l = 20 
w = 2 

r = 2
x_offset = 0
y_offset = 0
theta_offset = 0


T = 15
dt = 1.0/60



def eulerStep(state, u ): 
    theta = state[2,0]
    g = np.array([
        [np.cos(theta), 0], 
        [np.sin(theta), 0], 
        [0,             1]
    ])
    
    dstate_dt = g@u.T
    new_state = state + dt*dstate_dt

    return new_state


def getICpoints(): 

    xs = []
    ys = []

    N = int(l/4/r)

    offset_from_edge = 0.25
    ws = np.linspace(0+offset_from_edge,w-offset_from_edge,3)
    for width in ws: 
        # Bottom Line 
        x = -l/8 + x_offset
        for i in range(N+1):
            xs.append(x)
            ys.append(-l/4/pi-width + y_offset)
            x += r
        
        # Top Line 
        x = -l/8 + x_offset
        for i in range(N+1):
            xs.append(x)
            ys.append(l/4/pi+width + y_offset)
            x += r


        # Right Side 
        N_arcs = int((l/4 + w*pi)/r)
        x = l/8 + x_offset
        y = -l/4/pi - width +y_offset
        for i in range(N_arcs+1):
            xs.append(x)
            ys.append(y) 
            x = l/8 + (l/4/pi + width)  * np.sin(float(i)*pi/N_arcs)+x_offset
            y = 0   - (l/4/pi + width)  * np.cos(float(i)*pi/N_arcs)+y_offset

        # Left Side
        x = -l/8  + x_offset
        y = -l/4/pi - width + y_offset
        for i in range(N_arcs+1):
            xs.append(x)
            ys.append(y) 
            x = -l/8 - (l/4/pi + width )* np.sin(float(i)*pi/N_arcs)+x_offset
            y =  0   - (l/4/pi + width )* np.cos(float(i)*pi/N_arcs)+y_offset

    return xs, ys 


xs, ys = getICpoints()

x = 6
y = 2
theta = 0
state = np.array([[x],[y],[theta]])


for ic_idx, x in enumerate(xs):

    print("point number ", ic_idx, " of ", len(xs))
    y = ys[ic_idx] 
    theta = 0 
    state = np.array([[x],[y],[theta]])
    print(np.array_equal(state, np.array([[-0.5], [ 2.59154943], [0.0]]) ))
    if ONEOFF: 
        print(state)
        continue
        if not (ic_idx == oneoff_idx): 
            continue

    u_traj = []
    x_traj = []
    x_traj.append(state.tolist())

    for i in tqdm(range(int(T/dt))):
        x = float(state[0,0])
        y = -(float(state[1,0])) #+  y_offset
        theta = -float(state[2,0])

        if END2END: 
            position = airsim.Vector3r(x, y, 0)
            heading = airsim.utils.to_quaternion(pitch=0, roll=0, yaw=theta)

            pose = airsim.Pose(position, heading)
            client.simSetVehiclePose(pose, True)


            img, time_stamp1 = get_image()
            img = convertresponse(img,i)
            save_image(img,"./plots/imgs_traj/converted_img_current.png")

            img = Image.open("./plots/imgs_traj/converted_img_current.png")
            img_tensor = convert_tensor(img)

            img_tensor = img_tensor.unsqueeze(0)
            u = end2endController(img_tensor) 
            u = u.detach().numpy()
        else: 
            u,_ = raceTrackTROP(state)
            u = u.T

        state = eulerStep(state,u) 

        u_traj.append(u.tolist())
        x_traj.append(state.tolist())

    u_traj = np.array(u_traj).squeeze()
    x_traj = np.array(x_traj)

    if OVERWRITE_DATA: 
        np.save("plots/"+prepend+"x_traj_"+str(ic_idx)+".np", x_traj)
        np.save("plots/"+prepend+"u_traj_"+str(ic_idx)+".np", u_traj)
    if ONEOFF: 
        np.save("plots/"+prepend+"x_traj_oneoff.np", x_traj)
        np.save("plots/"+prepend+"u_traj_oneoff.np", u_traj)
    car_x = x_traj[:,0]
    car_y = x_traj[:,1]
    data_output_path = "."

    fig, axs = plt.subplots(2)
    axs[0].plot(u_traj[:,0])
    axs[1].plot(u_traj[:,1])
    plt.show()
    if OVERWRITE_DATA:
        saveFigure(fig, prepend+'inputs_'+str(i)+'.png', '.' )


    fig, axs = plt.subplots(3)
    axs[0].plot(x_traj[:,0])
    axs[1].plot(x_traj[:,1])
    axs[2].plot(x_traj[:,2])
    plt.show()
    if OVERWRITE_DATA: 
        saveFigure(fig, prepend+'states_'+str(ic_idx)+'.png', '.' )


    if OVERWRITE_DATA: 
        plotTrack(car_x, car_y+y_offset, l,w, data_output_path , True, prepend + str(ic_idx) , True)
    if ONEOFF: 
        plotTrack(car_x, car_y+y_offset, l,w, data_output_path , True, "oneoff", True)



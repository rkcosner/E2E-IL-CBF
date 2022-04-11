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

from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
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

def convertresponse(response): 
    # get numpy array
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
    # reshape array to 4 channel image array H X W X 4
    img_rgb = img1d.reshape(response.height, response.width, 3)
    if TORCH_IMAGE: 
        img_rgb = Image.fromarray(np.uint8(img_rgb)).convert("RGB")
        img_rgb = preprocess(img_rgb)
    return img_rgb 

def get_image(): 
    responses = client.simGetImages([airsim.ImageRequest("front_center_custom", airsim.ImageType.Scene, False, False)])
    return responses[0], responses[0].time_stamp


l = 20 
w = 2 
# def getOuterLoopDataset():

#     o_theta1 = np.arctan2(l/pi+ w, l / 2)
#     o_theta2 = np.arctan2(l/pi + w, -l / 2)
#     o_theta3 = np.arctan2(-l/pi - w, -l / 2) + 2*pi
#     o_theta4 = np.arctan2(-l/pi - w, l / 2)
#     def outer_loop(thetas):
#         r = l/pi + w
#         rs = []
#         for theta in thetas:
#             if (o_theta4 < theta and theta < o_theta1):
#                 rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
#             elif (o_theta4+2*pi < theta and theta < o_theta1+2*pi):
#                 rs.append(l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
#             elif o_theta1 < theta and theta < o_theta2:
#                 rs.append(r/np.sin(theta))
#             elif o_theta2 < theta and theta < o_theta3:
#                 rs.append(-l/2*np.cos(theta) + np.sqrt(r**2 - (l/2)**2*np.sin(theta)**2))
#             elif o_theta3 < theta and theta < o_theta4+2*pi:
#                 rs.append(-r/np.sin(theta))
#             else:
#                 rs.append(0)
#         return rs

#     theta = np.linspace()

r = 1
x_offset = 0
y_offset = -3
def getOuterLoopDataset(): 
    xs = []
    ys = []

    N = int(l/4/r)

    # Bottom Line 
    x = -l/8 + x_offset
    for i in range(N+1):
        xs.append(x)
        ys.append(-l/4/pi-w + y_offset)
        x += r
    
    # Top Line 
    x = -l/8 + x_offset
    for i in range(N+1):
        xs.append(x)
        ys.append(l/4/pi+w + y_offset)
        x += r


    # Right Side 
    N_arcs = int((l/4 + w*pi)/r)
    x = l/8 + x_offset
    y = -l/4/pi - w +y_offset
    for i in range(N_arcs+1):
        xs.append(x)
        ys.append(y) 
        x = l/8 + (l/4/pi + w)  * np.sin(float(i)*pi/N_arcs)+x_offset
        y = 0   - (l/4/pi + w)  * np.cos(float(i)*pi/N_arcs)+y_offset

    # Left Side
    x = -l/8  + x_offset
    y = -l/4/pi - w + y_offset
    for i in range(N_arcs+1):
        xs.append(x)
        ys.append(y) 
        x = -l/8 - (l/4/pi + w )* np.sin(float(i)*pi/N_arcs)+x_offset
        y =  0   - (l/4/pi + w )* np.cos(float(i)*pi/N_arcs)+y_offset

    return xs, ys 

##
# print("Getting data points with radius = ", r)
# xs, ys = getOuterLoopDataset()
# fig, ax = plt.subplots()
# ax.set_aspect("equal")
# plt.plot(xs, ys, '.')
# plt.show()

# thetas = np.linspace(0, 2*pi, int(2*pi/r ))

samples = np.load("racecarSamplePoints_r0.1.npy")
xs = samples[:,0]
ys = samples[:,1]
thetas = samples[:,2]
print("xs length = ", len(xs))
print("ys length = ", len(ys))
print("thetas length = ", len(thetas))
print(ys)


### 
print("Creating Directories")
path = os.getcwd()
now = datetime.now()
timestamp = now.strftime("%Y_%m_%d_%h_%m_%s")
directory_path = path + "/datasets/dataset" + str(timestamp)
os.system("mkdir " + path + "/datasets")
os.system("mkdir " + directory_path)
os.system("mkdir " + directory_path + "/imgs")

print("Getting Data")
with open(directory_path + "/data.csv", "w", newline="\n") as csvfile: 
    fieldnames = ["i",   "x", "y", "theta"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(len(thetas)): 
            # print("Iteration Step, i = ", i,"j = ", j)
            print(i, "\t", xs[i], " \t", ys[i])
            position = airsim.Vector3r(xs[i], ys[i], 0)
            heading = airsim.utils.to_quaternion(pitch=0, roll=0, yaw=thetas[i])

            pose = airsim.Pose(position, heading)
            client.simSetVehiclePose(pose, True)
            img, time_stamp1 = get_image()


            img = convertresponse(img)

            if TORCH_IMAGE: 
                save_image(img, directory_path + "/imgs/img_" + str(i) + ".png")
            else: 
                cv2.imwrite(directory_path + "/imgs/img_" + str(i) +".png", img)
            writer.writerow({"i":i, "x":xs[i], "y":ys[i], "theta":thetas[i]})


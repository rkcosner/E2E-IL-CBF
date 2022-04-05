from datetime import datetime
import os
from torchvision import transforms
from PIL import Image
import numpy as np
pi = np.pi

# Image Preprocessing Function for Learning Network
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def getTimeStamp():
    date_time_obj = datetime.now()
    timestamp_str = date_time_obj.strftime("%Y-%b-%d-%H-%M-%S")
    return timestamp_str


def saveFigure(fig, name, path):
    os.system("mkdir "+ path + "/plots")
    fig.savefig(path+"/plots/"+name)

def createDataOutputFolder(prepath, timestamp):
    path = "./data/dataOut/" + prepath + "/"+timestamp
    os.system("mkdir " + path)
    return path




def getEllipse():
    # Get ellipse
    c = np.sqrt(3) * (45 * pi / 180) ** 2  # 45 degrees
    x_min = -45 / 180 * pi - 0.1765
    x_max = -x_min
    x_now = x_min
    x = []
    y = []

    r = 0.01

    x_last = x_now
    y_last = (-2*x_now+np.sqrt(4*np.sqrt(3)*(c - np.sqrt(3)*x_now**2) + 4*x_now**2))/2/np.sqrt(3)#1 / np.sqrt(3) * np.sqrt(np.sqrt(3) * c - 2 * x_now ** 2) - x_now
    x.append(x_last)
    y.append(y_last)
    rs = []
    i = 0
    while x_now < x_max:
        y_now =  (-2*x_now+np.sqrt(4*np.sqrt(3)*(c - np.sqrt(3)*x_now**2) + 4*x_now**2))/2/np.sqrt(3)
        if ((x_now - x_last) ** 2 + (y_now - y_last) ** 2) > r ** 2:
            x.append(x_now)
            y.append(y_now)
            rs.append(np.sqrt((x_now - x_last) ** 2 + (y_now - y_last) ** 2))
            x_last = x_now
            y_last = y_now
        x_now += 0.00005
        i += 1

    x = np.array(x)
    ex = np.hstack([x, -x])
    y_bottom = -np.array(y)
    ey = np.hstack([y, y_bottom])
    return ex, ey

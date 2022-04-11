from controller_util import * 
import numpy as np 
import csv 
import os 
from tqdm import tqdm 

path = "/home/ryan/Documents/TripleCamera/AirSim/PythonClient/cdc22/set_r1/"
csv_filename = "data.csv"
csv_filename_new = "controller_data.csv"
socp_failure = 0

y_offset = -3


os.system("touch "+ path + csv_filename_new)
with open (path + csv_filename_new, "w", newline='\n') as newcsvfile: 
    fieldnames = ["i", "j",  "x", "y", "theta", "u1", "u2"]
    writer = csv.DictWriter(newcsvfile, fieldnames=fieldnames)
    writer.writeheader()

    with open(path + csv_filename, newline='\n') as csvfile: 
        reader = csv.reader(csvfile, delimiter=',')
        for row_idx, row in tqdm(enumerate(reader)): 
            if row_idx > 0: 
                state = np.array([[float(row[2])], [float(row[3])-y_offset], [float(row[4])]])
                u,flag = raceTrackTROP(state)
                if not flag: 
                    socp_failure += 1 
                writer.writerow({"i":row[0], "j":row[1], "x":row[2], "y":float(row[3])-y_offset, "theta":row[4], "u1":u[0,0], "u2":u[1,0]})

print("SOCP failures : ", socp_failure)
print("Failures are occuring when parallel to boundary")
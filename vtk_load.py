import espressomd
import object_in_fluid as oif

from espressomd import lb
from espressomd import lbboundaries
from espressomd import shapes
from espressomd import interactions

import numpy as np
import os, glob, sys, shutil, collections, pathlib
import random
import time, datetime
import argparse

# IJ Sept, 2021:
# capillary channel with one cell with inner particles
# checking the cell velocity wrt to cell size and stiffness
# using rhomboid vel

def get_middle_of_cell_first_method(data, n_of_meshpoints):

    axis_counter = 0

    sum_axis_X = 0
    sum_axis_Y = 0
    sum_axis_Z = 0

    for k in range(int(n_of_meshpoints)):

        axis_counter += 1

        sum_axis_X += float(data[k][0])
        sum_axis_Y += float(data[k][1])
        sum_axis_Z += float(data[k][2])

    centroid_X = sum_axis_X / axis_counter
    centroid_Y = sum_axis_Y / axis_counter
    centroid_Z = sum_axis_Z / axis_counter

    return centroid_X, centroid_Y, centroid_Z

# system settings --------

case = "C"
folder = "sim20007"
type_of_cell = "platelet"
number_of_cell = 12

# --------------------------


directory = "output/sim20004/vtk"

vtkFilePath = directory

# print("/mnt/c/Users/Ghost/Espresso/Espresso_januar_2022/output/output_deformed_clusters_with_particles/sim_"+ str(sim_no) + ".0/vtk")

# if file exists
file_exists = os.path.exists(vtkFilePath)
print("Subor existuje  "+ directory + " " + str(file_exists))

# checking number of all vtks and vtks for 1 cell
vtk_counter_of_all_cells = len(glob.glob1(vtkFilePath,type_of_cell+ "*.vtk"))
vtk_counter_of_one_cell = len(glob.glob1(vtkFilePath,type_of_cell + str(number_of_cell) + "_*.vtk")) # 2. parameter needed for loop

print("Pocet vtk : " + str(vtk_counter_of_all_cells))
print("Pocet vtk jednej bunky " + str(vtk_counter_of_one_cell))

# checking how many cells there are
n_of_cells = vtk_counter_of_all_cells/vtk_counter_of_one_cell # 1. parameter needed for loop

print("Pocet vsetkych buniek " + str(n_of_cells))

data = []

n_of_meshpoints = 0 # 3. parameter needed for loop
n_of_axis = 3 # 4. parameter needed for loop

for i in range(int(n_of_cells)):
    times_as_vtks = []
    print("Otvaram subor pre - " + str(i) + " bunku")
    for j in range(int(vtk_counter_of_one_cell)):
        meshpoints = []
        meshCounter = 0
        # opening file
        with open(vtkFilePath + "/" + type_of_cell + str(i) + "_" + str(j) + ".vtk",'r') as f:
            while True:
                meshpoint = f.readline()
                if len(meshpoint) > 40:
                    meshpoints.append(meshpoint.split())
                    meshCounter += 1
                if not meshpoint:
                    break
        n_of_meshpoints = meshCounter
        f.close()
        times_as_vtks.append(meshpoints)
    data.append(times_as_vtks)

number_of_non_moving_cells_at_the_end = 0
for j in range(int(n_of_cells)):
    number_of_steps = 0
    warning_level = 1
    prev = -100
    isNotMoving = False
    for i in range(len(data[j])):
        result = get_middle_of_cell_first_method(data[j][i], n_of_meshpoints)
        x = int(result[0])
        if (x == prev):
            warning_level = warning_level + 1

        if (x != prev):
            warning_level = 1
            if (isNotMoving == True):
                number_of_non_moving_cells_at_the_end = number_of_non_moving_cells_at_the_end - 1
                isNotMoving = False

        if (warning_level % 3 == 0):
            if (isNotMoving == False and (x % 90) >= 15 and (x % 90) <= 45):
                number_of_non_moving_cells_at_the_end = number_of_non_moving_cells_at_the_end + 1
                isNotMoving = True
        prev = x
print("Pocet buniek na konci sim : " + str(number_of_non_moving_cells_at_the_end))

exit()
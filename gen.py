import espressomd
import csv
import random
import logging
import object_in_fluid as oif
import json

from espressomd import lb
from espressomd import lbboundaries
from espressomd import shapes
from espressomd import interactions

import math

import numpy as np
import os, glob, sys, shutil

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

# region CheckPoint
class CheckPoint:
    def __init__(self, p_system, p_lbf, p_cp_freq=-1, p_last_cp_freq=-1):
        self.system = p_system
        self.lbf = p_lbf
        self.cp_freq = p_cp_freq
        self.last_cp_freq = p_last_cp_freq
        self.numCP = 1

    # metoda na ulozenie skriptu a argumentov
    def save_file(self, p_path):
        os.makedirs(p_path + "/CODE", exist_ok=True)
        shutil.copy(__file__, p_path + "/CODE/")
        f = open(p_path + "/CODE/args.txt", "w+")
        f.write("cp_freq:{}, last_cp_freq:{}".format(self.cp_freq, self.last_cp_freq))
        f.close()

    # metoda na automaticke ukladanie checkpointov
    # args - p_path sluzi iba ako cesta ku priecinku, kde si metoda uz vytvori vlastne subory
    # vysledkom su dva subory CP{poradie_iteracie}/particles.txt a CP{poradie_iteracie}/fluid.txt
    # tieto subory su dolezite na opatovne nacitanie systemu do podmienok v danej iteracie.
    # metoda automaticky uklada checkpoint podla zadaneho parametra triedy cp_freg, ktora udava ako casto
    # sa uklada checkpoint.
    # na druhu stranu, metoda taktiez uklada aj posledny checkpoint, a to tak casto ako je uvedene v parametri
    # triedy last_cp_freq, ktory ak je nezadany nebude ukladat posledne iteracie, a pri hodnote 1 bude ukladat
    # kazdy checkpoint
    # takto ulozene checkpointy budu mat predponu "LAST"
    def checkpoint_auto(self, p_path):
        if self.numCP == 1 and self.cp_freq == -1 and self.last_cp_freq == -1:
            print("Arguments for automatic checkpoint were not given. Therefore method checkpoint_auto() is redundant.")

        if self.cp_freq != -1 and self.numCP % self.cp_freq == 0:
            os.makedirs(p_path + "/CP" + str(self.numCP), exist_ok=True)
            f = open(p_path + "/CP" + str(self.numCP) + "/particles.txt", "w+")
            for p in self.system.part:
                origin = p.pos
                f.write("{} {} {}".format(origin[0], origin[1], origin[2]))
                velocity = p.v
                f.write(" {} {} {}".format(velocity[0], velocity[1], velocity[2]))
                force = p.f
                f.write(" {} {} {}".format(force[0], force[1], force[2]))
                f.write("\n")
            f.close()

            f = open(p_path + "/CP" + str(self.numCP) + "/fluid.txt", "w+")
            self.lbf.save_checkpoint(p_path + "/CP" + str(self.numCP) + "fluid.txt", 0)
            f.close()

            self.save_file(p_path + "/CP" + str(self.numCP))

            print("AUTO saved CP: {}".format(self.numCP))

        elif self.last_cp_freq != -1 and self.numCP % self.last_cp_freq == 0:
            os.makedirs(p_path + "/LAST", exist_ok=True)
            f = open(p_path + "/LAST/particles.txt", "w+")
            for p in self.system.part:
                origin = p.pos
                f.write("{} {} {}".format(origin[0], origin[1], origin[2]))
                velocity = p.v
                f.write(" {} {} {}".format(velocity[0], velocity[1], velocity[2]))
                force = p.f
                f.write(" {} {} {}".format(force[0], force[1], force[2]))
                f.write("\n")
            f.close()

            f = open(p_path + "/LAST/fluid.txt", "w+")
            self.lbf.save_checkpoint(p_path + "/LAST/fluid.txt", 0)
            f.close()

            self.save_file(p_path + "/LAST")

            print("LAST saved CP: {}".format(self.numCP))

        self.numCP = self.numCP + 1

    def load_checkpoint(self, p_path, check_point_num):
        f = open(p_path + "/CP" + str(check_point_num) + "/particles.txt", "r")
        for p in self.system.part:
            data = f.readline().split()
            origin = [float(ii) for ii in data[:3]]
            p.pos = origin
            velocity = [float(ii) for ii in data[3:6]]
            p.v = velocity
            force = [float(ii) for ii in data[6:9]]
            p.f = force
        f.close()

        self.lbf.load_checkpoint(p_path + "/CP" + str(check_point_num) + "fluid.txt", 0)

    def load_checkpoint_folder(self, p_path):
        f = open(p_path + "/particles.txt", "r")
        for p in self.system.part:
            data = f.readline().split()
            origin = [float(ii) for ii in data[:3]]
            p.pos = origin
            velocity = [float(ii) for ii in data[3:6]]
            p.v = velocity
            force = [float(ii) for ii in data[6:9]]
            p.f = force
        f.close()

        # self.lbf.load_checkpoint(p_path + "/fluid.txt", 0)

    def checkpoint_manual(self, p_path):
        os.makedirs(p_path, exist_ok=True)
        f = open(p_path + "/particles.txt", "w+")
        for p in self.system.part:
            origin = p.pos
            f.write("{} {} {}".format(origin[0], origin[1], origin[2]))
            velocity = p.v
            f.write(" {} {} {}".format(velocity[0], velocity[1], velocity[2]))
            force = p.f
            f.write(" {} {} {}".format(force[0], force[1], force[2]))
            f.write("\n")
        f.close()

        f = open(p_path + "/fluid.txt", "w+")
        self.lbf.save_checkpoint(p_path + "/fluid.txt", 0)
        f.close()

        self.save_file(p_path)

        print("MANUAL saved CP: {}".format(self.numCP))
# endregion

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def get_positions(modulo,  sim_id, is_platelets, start_tunel, end_tunel, x_position, y_position, z_position, r_tunnel, r_platelet, count, r_cell, cellPositions):
    helperPositions = cellPositions

    result = math.sqrt(2*math.pow(r_tunnel, 2))
    print(result - r_tunnel)
    positions = [] 
    for i in range(count):
        print(str(i))
        r = np.random.RandomState(int(sim_id) + 1)
        print("----")
        while (True):
            x = r.uniform(x_position - modulo/2 + r_cell, x_position + modulo/2 - r_cell)
            y = r.uniform(y_position - r_tunnel, y_position + r_tunnel)
            z = r.uniform(z_position - r_tunnel, z_position + r_tunnel)

            if x % modulo >= (start_tunel - r_platelet) and x % modulo <= (end_tunel + r_platelet):
                continue

            newPosition = [x % modulo, y % modulo, z % modulo]

            calculatedDistanceForChannel = distance(newPosition, [newPosition[0], y_position, z_position])

            distanceOne = r_cell
            distanceTwo = 2*r_cell
            if (is_platelets):
                distanceOne = r_platelet
                distanceTwo = r_platelet + r_cell

            if (calculatedDistanceForChannel + distanceOne < r_tunnel):
                canBeAdded = True
                for iterator in range(len(helperPositions)):
                    newHelperModuloPosition = convertToModulo(helperPositions[iterator], modulo)
                    calculatedDistance = distance(newPosition, newHelperModuloPosition)

                    if (calculatedDistance < distanceTwo):
                        canBeAdded = False
                        break
                if (canBeAdded):
                    helperPositions.append(newPosition)
                    positions.append(newPosition)
                    break

    return positions

def convertToModulo(arr, modulo):
    newArr = []
    for iterator in range(len(arr)):
        newValue = arr[iterator] % modulo
        newArr.append(newValue)
    return newArr

def set_inner_pore_particles(system, position, n_per_circle, n_parts, length, r_object, part_type, particles_positions):
    part_pos_x = position[0]
    part_pos_y = position[1]
    part_pos_z = position[2]

    part_angular_offset = 2 * math.pi / n_per_circle
    part_x_offset = length / n_parts

    x = part_pos_x
    for i in range(n_parts):
        x = part_pos_x + part_x_offset * i
        for j in range(n_per_circle):
            y = part_pos_y + r_object * math.cos(part_angular_offset * j + i % 2 * part_angular_offset / 2)
            z = part_pos_z + r_object * math.sin(part_angular_offset * j + i % 2 * part_angular_offset / 2)
            system.part.add(pos=[x, y, z], type=part_type, mass=1.0)
            particles_positions.append([x, y, z])
    return [part_pos_x, x]

def set_cone_parcticles_left(system, position, d, n_parts, length, r_object, part_type, particles_positions):
    part_pos_x = position[0]
    part_pos_y = position[1]
    part_pos_z = position[2]

    position_max_x = part_pos_x + length;
    isConstant = False
    constant = 0;
    for i in range(n_parts):
        x = part_pos_x + d * i
        r = ((length - (position_max_x - x))/length)*outside_radius_hollow
        r = abs(r)
        if (r >= r_object and isConstant == False):
          isConstant = True
          constant = d * i;

        if (isConstant):
          o = r * 2 * math.pi
          n_per_circle = int(o/d)
          part_angular_offset = 2 * math.pi / n_per_circle 
          for j in range(n_per_circle):
              y = part_pos_y + r * math.cos(part_angular_offset * j + i % 2 * part_angular_offset / 2)
              z = part_pos_z + r * math.sin(part_angular_offset * j + i % 2 * part_angular_offset / 2)
              system.part.add(pos=[x - constant, y, z] , type=part_type, mass=1.0);
              particles_positions.append([x - constant, y, z])

def get_left_cone_parameters(system, position, d, length, r_object):
    part_pos_x = position[0]
    part_pos_y = position[1]
    part_pos_z = position[2]
    
    position_max_x = part_pos_x + length;
    i = 1
    while(True):
        x = part_pos_x + d * i
        i = i + 1
        r = ((length - (position_max_x - x))/length)*outside_radius_hollow
        r = abs(r)
        if (r >= r_object):
          return r;
          

def set_cone_parcticles_right(system, position, d, n_parts, length, r_object, part_type, particles_positions):
    part_pos_x = position[0]
    part_pos_y = position[1]
    part_pos_z = position[2]

    position_max_x = part_pos_x + length;
    isConstant = False
    constant = 0;
    for i in range(n_parts):
        x = part_pos_x - d * i
        r = ((length - (position_max_x - x))/length)*outside_radius_hollow
        r = abs(r)
        if (r >= r_object and isConstant == False):
          isConstant = True
          constant = d * i;

        if (isConstant):
          o = r * 2 * math.pi
          n_per_circle = int(o/d)
          part_angular_offset = 2 * math.pi / n_per_circle
          for j in range(n_per_circle):
            y = part_pos_y + r * math.cos(part_angular_offset * j + i % 2 * part_angular_offset / 2)
            z = part_pos_z + r * math.sin(part_angular_offset * j + i % 2 * part_angular_offset / 2)
            system.part.add(pos=[x + constant, y, z], type=part_type, mass=1.0);
            particles_positions.append([x + constant, y, z])

def get_right_cone_parameters(system, position, d, length, r_object):
    part_pos_x = position[0]
    part_pos_y = position[1]
    part_pos_z = position[2]

    position_max_x = part_pos_x + length;
    i = 1
    while(True):
        x = part_pos_x - d * i
        i = i + 1
        r = ((length - (position_max_x - x))/length)*outside_radius_hollow
        r = abs(r)
        if (r >= r_object):
            return r

    
def set_particles(system, d, length_cone_wall, part_type, cone_length, pore_length, r_object , positions, opening_angle, particles_positions): 
    part_type = part_type

    o = math.pi * 2 * r_object

    parts_per_circle = int(o / d);

    parts_per_circle_pore = math.ceil(pore_length/d)

    init_poisition_x = positions[0] - (pore_length/2)
    init_position = [init_poisition_x, positions[1], positions[2]]

# initial particle positions
    x = set_inner_pore_particles(system, init_position, parts_per_circle, parts_per_circle_pore, pore_length, r_object, part_type, particles_positions)
# initial particle positions
    init_cone_position_left = [positions[0] + pore_length/2, positions[1], positions[2]]

    left_r = get_left_cone_parameters(system, init_cone_position_left, d, cone_length, r_object)
    l_length = left_r / math.tan(opening_angle)
    print(str(l_length+cone_length))

    helper_part_per_length = math.ceil((l_length+cone_length) / d)
    set_cone_parcticles_left(system, init_cone_position_left,  d, helper_part_per_length, cone_length + l_length, r_object, part_type, particles_positions)

    init_cone_position_right = [init_poisition_x,  y_position_inner_pore, z_position_inner_pore]

    right_r = get_right_cone_parameters(system, init_cone_position_right, d, cone_length, r_object)
    r_length = right_r / math.tan(opening_angle)
    print(str(r_length+cone_length))

    helper_part_per_length = math.ceil((r_length+cone_length) / d)
    set_cone_parcticles_right(system, init_cone_position_right, d, helper_part_per_length, cone_length + r_length, r_object, part_type, particles_positions)

# fix all particles (so they do not move)
    for part in system.part:
        part.fix = [1, 1, 1]

def create_pore(center_x, center_y, center_z, length, radius):
    tmp_shape = shapes.SimplePore(axis=[1, 0, 0],
                                  length=length,
                                  center=[center_x, center_y, center_z],
                                  radius=radius,
                                  smoothing_radius=0.5)
    boundaries.append(tmp_shape)
    oif.output_vtk_cylinder_open(cyl_shape=shapes.Cylinder(center=[center_x, center_y, center_z],
                                                      axis=[1.0, 0.0, 0.0],
                                                      length=length,
                                                      radius=radius,
                                                      direction=1),
                            out_file=vtk_directory + "/channel.vtk", n=20)

def create_inner_pore(center_x, center_y, center_z, length, radius):
    tmp_shape = shapes.SimplePore(axis=[1, 0, 0],
                                  length=length,
                                  center=[center_x, center_y, center_z],
                                  radius=radius,
                                  smoothing_radius=0.5)
    boundaries.append(tmp_shape)
    oif.output_vtk_cylinder_open(cyl_shape=shapes.Cylinder(center=[center_x, center_y, center_z],
                                                      axis=[1.0, 0.0, 0.0],
                                                      length=length,
                                                      radius=radius,
                                                      direction=1),
                            out_file=vtk_directory + "/inner.vtk", n=20)

def createBoundries(system, boundries, types):
    for boundary in boundries:
        system.lbboundaries.add(lbboundaries.LBBoundary(shape=boundary))
    for count, boundary_shape in enumerate(boundries):
    # hollowCone
        if count == 0:
            system.constraints.add(shape=boundary_shape, particle_type=types[0], penetrable=False)
        elif count == 1:
            system.constraints.add(shape=boundary_shape, particle_type=types[1], penetrable=False)
    # inner pore
        elif count == 2:
            system.constraints.add(shape=boundary_shape, particle_type=types[2], penetrable=False)
        else:
            system.constraints.add(shape=boundary_shape, particle_type=types[3], penetrable=False)

def setCellBoundriesIteractions(cell_types, system, types):
    for c_type in cell_types:
        system.non_bonded_inter[c_type, types[3]].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.0)
        system.non_bonded_inter[c_type, types[2]].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.0)
    for c_type in cell_types:
        system.non_bonded_inter[c_type, types[0]].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.3)
        system.non_bonded_inter[c_type, types[1]].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.3)

import time

if len(sys.argv) != 3:
    print ("1 argument are expected:")
    print ("sim_id: id of the simulation")
    print (" ")

 # read arguments
i = 0
for i, arg in enumerate(sys.argv):
    if i%2 == 1:
        print (str(arg) + " \t" + sys.argv[i + 1])
    if arg == "sim_id":
        sim_id = sys.argv[i + 1]

# check that we have everything
if sim_id == "ND":
    print("something wrong when reading arguments, quitting.")

# create folder structure
directory = "output/sim"+str(sim_id)
os.makedirs(directory)


vtk_directory = directory + "/vtk"
os.makedirs(vtk_directory)
data_directory = directory + "/data"
os.makedirs(data_directory)

# setting parameters from json
f = open('loading.json')
data = json.load(f)
f.close()

d = data['d']
outer_pore_r = data['outer_pore_r']
inner_pore_r = data['inner_pore_r']
length_pore = data['length_pore']
length_cone = data['length_cone']

# channel constants
part_type = 100

inner_length = length_pore + (length_cone * 2)
outer_length = inner_length * 3
inner_radius_hollow = inner_pore_r
outside_radius_hollow = outer_pore_r

opening_angle=math.atan((outside_radius_hollow - inner_radius_hollow) /length_cone)
length_cone_wall = math.ceil(math.sqrt(pow((outside_radius_hollow - inner_radius_hollow), 2) + pow(length_cone, 2)))


x_position_inner_pore = 0.0 + (inner_length)
y_position_inner_pore = 0.0 + outside_radius_hollow
z_position_inner_pore = 0.0 + outside_radius_hollow

x_position_outer_pore = 0.0 + (outer_length/2)
y_position_outer_pore = y_position_inner_pore 
z_position_outer_pore = z_position_inner_pore
# system constants
system = espressomd.System(box_l=[outer_length , 2*outside_radius_hollow, 2*outside_radius_hollow])
system.cell_system.skin = 0.2
system.time_step = 0.1

volumeOfCone = 1/3 * np.pi * length_cone * (pow(outside_radius_hollow, 2) + outside_radius_hollow*inner_radius_hollow + pow(inner_radius_hollow, 2))
volumeInner = length_pore * np.pi * pow(inner_radius_hollow, 2)
volumeOfRestOfChannel = (outer_length - inner_length) * np.pi * pow(outside_radius_hollow, 2)
volumeInflow = 2*volumeOfCone + volumeInner + volumeOfRestOfChannel
logging.info("VOLUME OF INFLOW : " + str(volumeInflow))


# --------- particles wall ------------
logging.info("particles wall start")
particles_positions = []
set_particles(system, d, length_cone_wall, part_type, length_cone, length_pore, inner_pore_r, [x_position_inner_pore, y_position_inner_pore, z_position_inner_pore], opening_angle,particles_positions)
system.part.writevtk(vtk_directory + "/part_all.vtk", types=part_type)
logging.info("particles wall end")
# --------- end particles wall ------------

logging.info("RBC, PLT started")
r_red_cell = data["rbc_r"]
r_platelet_cell = data["plt_r"]

# creating the template for Red blood cells (RBCs)
rbcType = oif.OifCellType(nodes_file="input/rbc374nodes.dat", triangles_file="input/rbc374triangles.dat",
                        check_orientation=False, system=system, ks=data["rbc"]["ks"], kb=0.007, kal=data["rbc"]["kal"],
                        kag=0.9, kv=0.5, resize=[r_red_cell, r_red_cell, r_red_cell], normal=True)

# creating the template for platelets
plateletType = oif.OifCellType(nodes_file="input/ellipsoid_130nodes.dat", triangles_file="input/ellipsoid_130triangles.dat",
                        check_orientation=False, system=system, ks=0.5, kb=0.07, kal=0.5,
                        kag=0.5, kv=0.9, resize=[r_platelet_cell, r_platelet_cell, r_platelet_cell], normal=True)

red_cells = []
red_cell_types = []
platelets = []
platelet_types = []

red_cells_count = data["rbc_count"]
platelets_count = data["plt_count"]

# creating the RBCs

start_of_stenosis = x_position_inner_pore - (length_pore/2) - length_cone
end_of_stenosis = x_position_inner_pore + (length_pore/2) + length_cone

# positions = get_positions(outer_length, sim_id, False,start_of_stenosis, end_of_stenosis, x_position_outer_pore, y_position_outer_pore ,z_position_outer_pore, outer_pore_r, r_platelet_cell, red_cells_count, r_red_cell, [])
positions = [0.0, 0.0, 0.0]
for i in range(red_cells_count):
    red_cell = oif.OifCell(cell_type=rbcType, particle_type=i,
                           origin=positions,
                           rotate=[0.0, np.pi / 2, 0.0], exclusion_neighbours=True, particle_mass=0.5)
    red_cells.append(red_cell)
    red_cell_types.append(i)

RBC_volume = 0
for i in range(red_cells_count):
    RBC_volume = RBC_volume + red_cells[i].volume();

logging.info("VOLUME OF RBC : " + str(RBC_volume))
logging.info("RBC hematocrit : " + str((RBC_volume / volumeInflow) * 100))

# platelet_positions = get_positions(outer_length,sim_id, True,start_of_stenosis, end_of_stenosis, x_position_outer_pore, y_position_outer_pore ,z_position_outer_pore, outer_pore_r, r_platelet_cell, platelets_count, r_red_cell, positions)
platelet_positions = [0.0, 0.0, 0.0] 
for i in range(platelets_count):
    platelet_part_type = i + red_cells_count
    platelet = oif.OifCell(cell_type=plateletType, particle_type=platelet_part_type,
                           origin=platelet_positions,
                           rotate=[0.0, random.uniform(0.0, 1 / 6 * np.pi / 2), random.uniform(0.0, 1 / 6 * np.pi / 2)])
    platelets.append(platelet)
    platelet_types.append(platelet_part_type)

PLT_volume = 0
for i in range(platelets_count):
    PLT_volume = PLT_volume + platelets[i].volume()

logging.info("VOLUME OF PLATELETS : " + str(PLT_volume))
logging.info("Ratio : " + str((PLT_volume / volumeInflow) * 100))

cell_types = red_cell_types + platelet_types

logging.info("RBC, PLT craeted")

logging.info("Fluid started")

lbf = espressomd.lb.LBFluid(agrid=1,
                            dens=1,
                            visc=data["fluid"]["viscosity"],
                            tau=system.time_step,
                            ext_force_density=[data["fluid"]["ext_force_density_x"], 0.0, 0.0])

system.actors.add(lbf)

logging.info("Fluid created")

gamma_friction_c = rbcType.suggest_LBgamma(visc=1.0, dens=1.0)
gamma_friction_p = plateletType.suggest_LBgamma(visc=1.0, dens=1.0)
logging.info("gamma_friction_ RBC " + str(gamma_friction_c))
logging.info("gamma_friction_ PLT " + str(gamma_friction_p))
system.thermostat.set_lb(LB_fluid=lbf,
                         seed=123,
                         gamma=(gamma_friction_c + gamma_friction_p) / 2)



boundaries = []
# --------------CONE-----------------
adv_rad = 5
tmp_hollow_cone = shapes.HollowCone(
    center=[(x_position_inner_pore - (length_pore/2)) - length_cone/2, y_position_inner_pore, z_position_inner_pore], 
    axis=[1.0, 0.0, 0.0], 
    width=5, 
    direction=1, 
    opening_angle=opening_angle, 
    inner_radius=inner_radius_hollow + adv_rad, 
    outer_radius=outside_radius_hollow + adv_rad 
    )
boundaries.append(tmp_hollow_cone)
oif.output_vtk_hollow_cone_open(hollow_shape=tmp_hollow_cone, out_file=vtk_directory+"/hollowCone.vtk", n=20)

tmp_hollow_cone = shapes.HollowCone(
    center=[(x_position_inner_pore + (length_pore/2)) + length_cone/2, y_position_inner_pore, z_position_inner_pore],
     axis=[-1.0, 0.0, 0.0], 
     width=5, 
     direction=1, 
     opening_angle=opening_angle, 
     inner_radius=inner_radius_hollow + adv_rad, 
     outer_radius=outside_radius_hollow + adv_rad)
boundaries.append(tmp_hollow_cone)
oif.output_vtk_hollow_cone_open(hollow_shape=tmp_hollow_cone, out_file=vtk_directory+"/hollowConeSecond.vtk", n=20)
# --------------END CONE --------------

#---------------PORE--------------------
create_pore(x_position_outer_pore, y_position_outer_pore , z_position_outer_pore, outer_length, outer_pore_r)
create_inner_pore(x_position_inner_pore, y_position_inner_pore, z_position_inner_pore, length_pore, inner_pore_r)
#---------------END PORE--------------------

typesInnerPore = 100
typesChannel = 130
typesHollowCone = 160
typesHollowConeSecond = 190

allTypes = [typesHollowConeSecond, typesHollowCone, typesChannel, typesInnerPore]
createBoundries(system, boundaries, allTypes)
logging.info("Boundaries created")


# region Interactions

# soft sphere interaction - because membrane overlap
# morse interaction for red-cell and platelet
for j in range(len(red_cell_types)):
    for k in range(len(platelet_types)):
        system.non_bonded_inter[red_cell_types[j], platelet_types[k]].morse.set_params(eps=data["rbc_plt_morse"]["eps"], alpha=data["rbc_plt_morse"]["alpha"], rmin=data["rbc_plt_morse"]["rmin"],
                                                                                       cutoff=data["rbc_plt_morse"]["cutoff"])
# morse interaction for platelet platelet
for j in range(len(platelet_types)):
    for k in range(j + 1, len(platelet_types)):
        system.non_bonded_inter[platelet_types[j], platelet_types[k]].morse.set_params(eps=data["plt_plt_morse"]["eps"], alpha=data["plt_plt_morse"]["alpha"], rmin=data["plt_plt_morse"]["rmin"],
                                                                                       cutoff=data["plt_plt_morse"]["cutoff"])
# morse interaction for platelets and particles
for p_type in platelet_types:
    system.non_bonded_inter[p_type, part_type].morse.set_params(eps=data["plt_particles_morse"]["eps"], alpha=data["plt_particles_morse"]["alpha"], rmin=data["plt_particles_morse"]["rmin"],
                                                                                       cutoff=data["plt_particles_morse"]["cutoff"])

# morse interaction RBC x RBC
for j in range(len(red_cell_types)):
    for k in range(j + 1, len(red_cell_types)):
        system.non_bonded_inter[red_cell_types[j], red_cell_types[k]].morse.set_params(eps=data["rbc_rbc_morse"]["eps"], alpha=data["rbc_rbc_morse"]["alpha"], rmin=data["rbc_rbc_morse"]["rmin"],
                                                                                       cutoff=data["rbc_rbc_morse"]["cutoff"])

# sphere r
for r_type in red_cell_types:
    system.non_bonded_inter[r_type, r_type].soft_sphere.set_params(a=0.002, n=1.5, cutoff=0.8, offset=0.0)

logging.info("Cell interactions with cells created")
setCellBoundriesIteractions(cell_types, system, allTypes)
logging.info("Iteractions with boundaries created")

previous_directory = data["previous_checkpoint"]

# Definition of CP class without frequency parameters
CP = CheckPoint(p_system=system, p_lbf=lbf)
# Loading of a checkpoint
CP.load_checkpoint_folder(previous_directory)

# store initial positions before running any step
for part_id, rbc in enumerate(red_cells):
    rbc.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(part_id) + "_" + str(0) + ".vtk")

for part_id, platelet in enumerate(platelets):
    platelet.output_vtk_pos_folded(file_name=vtk_directory + "/platelet" + str(part_id) + "_" + str(0) + ".vtk")

lbf.print_vtk_velocity(vtk_directory+"/fluid_" + str(0)+ "_.vtk")

# csv_file_plt = []
# export_writer_plt = []
# plt_number = 12
# for i in range(len(red_cells)):
#     csv_file_plt.append(open(data_directory + "/plt_" + str(plt_number) + "_rbc" + str(i) + ".csv", mode='w'))
#     export_writer_plt.append(csv.writer(csv_file_plt[len(csv_file_plt) -1], delimiter=';'))
#     export_writer_plt[len(export_writer_plt) - 1].writerow(['time', 'rbc-x', 'rbc-y', 'rbc-z', 'plt-x', 'plt-y', 'plt-z'])

# csv_file_plt_force = []
# export_writer_plt_force = []
# number_of_files = len(platelets)
# for i in range(number_of_files):
#     csv_file_plt_force.append(open(data_directory + "/plt_" + str(i) + ".csv", mode='w'))
#     export_writer_plt_force.append(csv.writer(csv_file_plt_force[len(csv_file_plt_force) -1], delimiter=';'))
#     export_writer_plt_force[len(export_writer_plt_force) - 1].writerow(['time', 'plt-x', 'plt-y', 'plt-z', 'particles-x', 'particles-y', 'particles-z'])



csv_fluid = open(data_directory + "/fluid_average.csv", mode='w')
export_writer_fluid = csv.writer(csv_fluid, delimiter=';')
export_writer_fluid.writerow(['time', 'avg'])


maxCycle = 100000
steps = 500
time = 0


number_of_values = 15;
last_plts_positions = []
for plt_iterator in range(len(platelets)):
    helper = []
    for iterator in range(number_of_values):
        helper.append(0)
    last_plts_positions.append(helper)

for plt_iterator in range(len(platelets)):
    dis = distance(platelets[plt_iterator].get_origin(), [platelets[plt_iterator].get_origin()[0], y_position_outer_pore, z_position_outer_pore])
    if (dis + r_platelet_cell >= outer_pore_r):
        print(str(plt_iterator) + " : " + str(dis + r_platelet_cell))
number_of_added_cells = 0
added_cell_types = []

for i in range(1, maxCycle):
    for plt_iterator in range(len(platelets)):
        origin = platelets[plt_iterator].get_origin()
        x_origin = origin[0]
        last_plts_positions[plt_iterator][i%number_of_values] = int(x_origin)


    system.integrator.run(steps=steps)

    velocity = lbf[int(x_position_inner_pore + (length_pore) + 3*length_cone), int(y_position_inner_pore), int(z_position_inner_pore)].velocity;
    
    logging.info("Maximum velocity of fluid : ")
    
    logging.info(velocity)
    
    logging.info("Average velcity of fluid : " + str(velocity[0]/2))
    export_writer_fluid.writerow([time, str(velocity[0]/2)])

    # for rbc_iterator in range(len(red_cells)):
    #     full_force_rbc_to_plt = [0.0, 0.0, 0.0]
    #     full_force_plt_to_rbc = [0.0, 0.0, 0.0]
    #     for point_rbc in red_cells[rbc_iterator].mesh.points:
    #         pos_rbc = point_rbc.get_pos()
    #         force_in_plt_rbc_to_plt = [0.0, 0.0, 0.0]
    #         force_in_plt_plt_to_rbc = [0.0, 0.0, 0.0]
    #         for point_plt in platelets[plt_number].mesh.points:
    #             distance_space = distance(pos_rbc, point_plt.get_pos())
    #             if distance_space < data["rbc_plt_morse"]["cutoff"]:
    #                 # vzdialenost rmin
    #                 x = distance_space - data["rbc_plt_morse"]["rmin"]
    #                 # vypocet morse
    #                 morse_potencial = ( data["rbc_plt_morse"]["eps"] * ( math.exp(-2*data["rbc_plt_morse"]["alpha"] * x) - 2*math.exp(-data["rbc_plt_morse"]["alpha"]* x) ) )
    #                 # ziskanie vectore
    #                 vector_rbc_to_plt = pos_rbc - point_plt.get_pos()
    #                 vector_plt_to_rbc = point_plt.get_pos() - pos_rbc
    #                 # ziskanie jednotkoveho vectora .. rbc posobi na plt
    #                 final_vector_rbc_to_plt = vector_rbc_to_plt / abs(pos_rbc - point_plt.get_pos())
    #                 final_vector_plt_to_rbc = vector_plt_to_rbc / abs(pos_rbc - point_plt.get_pos())
    #                 # vynasobenie jednotkoveho vectora silou
    #                 force_vector_rbc_to_plt = final_vector_rbc_to_plt * morse_potencial
    #                 force_vector_plt_to_rbc = final_vector_plt_to_rbc * morse_potencial

    #                 # zapocitanie sily jednej castice rbc ktora posobi na jednu casticu do celkovej sily ktora posobi na plt
    #                 force_in_plt_rbc_to_plt = [x + y for x,y in zip(force_in_plt_rbc_to_plt, force_vector_rbc_to_plt)]
    #                 force_in_plt_plt_to_rbc = [x + y for x,y in zip(force_in_plt_plt_to_rbc, force_vector_plt_to_rbc)]
    #         # zapocitanie sily ktorou posobi jedna rbc na jednu plt do celkovej sily ktora posobi na plt
    #         full_force_rbc_to_plt = [x + y for x,y in zip(full_force_rbc_to_plt, force_in_plt_rbc_to_plt)]
    #         full_force_plt_to_rbc = [x + y for x,y in zip(full_force_plt_to_rbc, force_in_plt_plt_to_rbc)]
    #     export_writer_plt[rbc_iterator].writerow([time, full_force_rbc_to_plt[0], full_force_rbc_to_plt[1], full_force_rbc_to_plt[2], full_force_plt_to_rbc[0], full_force_plt_to_rbc[1], full_force_plt_to_rbc[2]])

    # for plt_iterator in range(number_of_files):
    #     full_force_particles_to_plt = [0.0, 0.0, 0.0]
    #     full_force_plt_to_particles = [0.0, 0.0, 0.0]
    #     for point_plt in platelets[plt_iterator].mesh.points:
    #         pos_plt = point_plt.get_pos()
    #         force_particles_to_plt = [0.0, 0.0, 0.0]
    #         force_plt_to_particles = [0.0, 0.0, 0.0]
    #         for particle_pos in particles_positions:
    #             distance_space = distance(particle_pos, pos_plt)
    #             if distance_space < data["plt_particles_morse"]["cutoff"]:
    #                 # vzdialenost rmin
    #                 x = distance_space - data["plt_particles_morse"]["rmin"]
    #                 # vypocet morse
    #                 morse_potencial = ( data["plt_particles_morse"]["eps"] * ( math.exp(-2*data["plt_particles_morse"]["alpha"] * x) - 2*math.exp(-data["plt_particles_morse"]["alpha"]* x) ) )
    #                 # ziskanie vectore
    #                 vector_particles_to_plt = particle_pos - pos_plt
    #                 vector_plt_to_particles = pos_plt - particle_pos
    #                 # ziskanie jednotkoveho vectora .. rbc posobi na plt
    #                 final_vector_particles_to_plt = vector_particles_to_plt / abs(pos_plt - particle_pos)
    #                 final_vector_plt_to_particles = vector_plt_to_particles / abs(pos_plt - particle_pos)
    #                 # vynasobenie jednotkoveho vectora silou
    #                 force_vector_particles_to_plt = final_vector_particles_to_plt * morse_potencial
    #                 force_vector_plt_to_particles = final_vector_plt_to_particles * morse_potencial

    #                 # zapocitanie sily jednej castice rbc ktora posobi na jednu casticu do celkovej sily ktora posobi na plt
    #                 force_particles_to_plt = [x + y for x,y in zip(force_particles_to_plt, force_vector_particles_to_plt)]
    #                 force_plt_to_particles = [x + y for x,y in zip(force_plt_to_particles, force_vector_plt_to_particles)]
    #         # zapocitanie sily ktorou posobi jedna rbc na jednu plt do celkovej sily ktora posobi na plt
    #         full_force_particles_to_plt = [x + y for x,y in zip(full_force_particles_to_plt, force_particles_to_plt)]
    #         full_force_plt_to_particles = [x + y for x,y in zip(full_force_plt_to_particles, full_force_plt_to_particles)]
    #     export_writer_plt_force[plt_iterator].writerow([time, full_force_particles_to_plt[0], full_force_particles_to_plt[1], full_force_particles_to_plt[2], full_force_plt_to_particles[0], full_force_plt_to_particles[1], full_force_plt_to_particles[2]])
 
    count_to_add = 0
    if (len(last_plts_positions) > number_of_values):
        for plt_iterator in range(len(platelets)):
            last_positions = last_plts_positions[plt_iterator]
            result = all(element == last_positions[0] for element in last_positions)
            if result and (last_positions[i % number_of_values] % outer_length >= start_of_stenosis and last_positions[i % number_of_values] % outer_length <= end_of_stenosis):
                if plt_iterator not in added_cell_types:
                    added_cell_types.append(plt_iterator)
                    count_to_add = count_to_add + 1

    if (False):
        help_positions = []
        for iterator in range(len(red_cells)):
            help_positions.append(red_cells[iterator].get_origin())

        for iterator in range(len(platelets)):
            help_positions.append(platelets[iterator].get_origin())


        for iterator in range(count_to_add):
            # zvysenie count v aplikacii
            number_of_added_cells = number_of_added_cells + 1
            platelets_count = platelets_count + 1
            # ziskanie dalsieho type
            logging.info("POCET PLATELETS :" + str(platelet_types))
            platelet_part_type = platelet_types[len(platelet_types) - 1] + 1
            # vytvorenie pozicie
            new_positions = get_positions(outer_length,sim_id, True, start_of_stenosis, end_of_stenosis, x_position_outer_pore, y_position_outer_pore ,z_position_outer_pore, outer_pore_r, r_platelet_cell, 1,r_red_cell, help_positions)
            new_position = new_positions[0]
            # inicializacia oifcell
            platelet = oif.OifCell(cell_type=plateletType, particle_type=platelet_part_type,
                           origin=new_position,
                           rotate=[0.0, random.uniform(0.0, 1 / 6 * np.pi / 2), random.uniform(0.0, 1 / 6 * np.pi / 2)])

            # interakcie pre steny
            system.non_bonded_inter[platelet_part_type, typesInnerPore].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.0)
            system.non_bonded_inter[platelet_part_type, typesChannel].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.0)
            system.non_bonded_inter[platelet_part_type, typesHollowConeSecond].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.3)
            system.non_bonded_inter[platelet_part_type, typesHollowCone].soft_sphere.set_params(a=0.00022, n=2.0, cutoff=0.6, offset=0.3)
            # interakcie medzi bunkami
            # RBCxPLT
            for j in range(len(red_cell_types)):
                system.non_bonded_inter[red_cell_types[j], platelet_part_type].morse.set_params(eps=data["rbc_plt_morse"]["eps"], alpha=data["rbc_plt_morse"]["alpha"], rmin=data["rbc_plt_morse"]["rmin"],
                                                                                       cutoff=data["rbc_plt_morse"]["cutoff"])
            # PLTxPLT
            for j in range(len(platelet_types)):
                system.non_bonded_inter[platelet_types[j], platelet_part_type].morse.set_params(eps=data["plt_plt_morse"]["eps"], alpha=data["plt_plt_morse"]["alpha"], rmin=data["plt_plt_morse"]["rmin"],
                                                                                       cutoff=data["plt_plt_morse"]["cutoff"])
            # PLTxPRT
            system.non_bonded_inter[p_type, platelet_part_type].morse.set_params(eps=data["plt_particles_morse"]["eps"], alpha=data["plt_particles_morse"]["alpha"], rmin=data["plt_particles_morse"]["rmin"],
                                                                                       cutoff=data["plt_particles_morse"]["cutoff"])
            # pridanie do poli pre aplikaciu
            platelets.append(platelet)
            platelet_types.append(platelet_part_type)
            logging.info("POCET PLATELETS :" + str(platelet_types))

            for inside_iterator in range(i):
                f = open(vtk_directory + "/platelet" + str(len(platelets) - 1) + "_" + str(inside_iterator)+ ".vtk", "w+") 
                f.write('# vtk DataFile Version 3.0\n')
                f.write('Data\n')
                f.write('ASCII\n')
                f.write('DATASET POLYDATA\n')
                f.write('POINTS 0 float\n')
                f.close()

            last_plts_positions = []
            for plt_iterator in range(len(platelets)):
                helper = []
                for iterator in range(number_of_values):
                    helper.append(0)
                last_plts_positions.append(helper)

    lbf.print_vtk_velocity(vtk_directory+"/fluid_" + str(i)+ "_.vtk")
    for part_id, rbc in enumerate(red_cells):
        rbc.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(part_id) + "_" + str(i)+ ".vtk")

    for part_id, platelet in enumerate(platelets):
        platelet.output_vtk_pos_folded(file_name=vtk_directory + "/platelet" + str(part_id) + "_" + str(i) + ".vtk")

    # store each 10th cycle inside checkpoint (for later re-loading)
    if i % 20 == 0:
        CP.checkpoint_manual(directory + "/CheckPoint" + str(time))

    print("Added cells : " + str(number_of_added_cells))
    print("i : " + str(i) + " time: " + str(time))   
    time = time + steps
# for i in range(red_cells_count):
#     csv_file_plt[i].close();

# for i in range(number_of_files):
#     csv_file_plt_force[i].close();

csv_fluid.close();
exit()
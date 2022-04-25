# Copyright (C) 2010-2019 The ESPResSo project
#
# This file is part of ESPResSo.
#
# ESPResSo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ESPResSo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import numpy as np
import random
import math
import espressomd
from espressomd.interactions import OifLocalForces, OifGlobalForces, OifOutDirection
from .oif_utils import (
    large_number, small_epsilon, discard_epsilon, custom_str, norm,
    vec_distance, get_triangle_normal, area_triangle, angle_btw_triangles, angle_btw_vectors,
    oif_calc_stretching_force, oif_calc_bending_force,
    oif_calc_local_area_force, oif_calc_global_area_force, oif_calc_volume_force, output_vtk_lines
)

import object_in_fluid as oif


class FixedPoint:
    """
    Represents mesh points, not connected to any ESPResSo particle.

    """

    def __init__(self, pos, id):
        if not isinstance(id, int):
            raise TypeError("Id must be integer.")
        if not ((len(pos) == 3) and isinstance(pos[0], float) and isinstance(
                pos[1], float) and isinstance(pos[2], float)):
            raise TypeError("Pos must be a list of three floats.")

        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.id = id
        self.neighbour_ids = []

    def get_pos(self):
        return [self.x, self.y, self.z]

    def get_id(self):
        return self.id


class PartPoint:
    """
    Represents mesh points, connected to ESPResSo particle.

    """

    # part is physical ESPResSo particle corresponding to that particular point

    def __init__(self, part, id, part_id):
        if not (isinstance(part, espressomd.particle_data.ParticleHandle)
                and isinstance(id, int) and isinstance(part_id, int)):
            raise TypeError("Arguments to PartPoint are incorrect.")
        self.part = part
        self.part_id = part_id  # because in adding bonds to the particles in OifCell
        # one needs to know the global id of the particle.
        self.id = id
        self.neighbour_ids = []

    def get_pos(self):
        return self.part.pos

    def get_part_id(self):
        return self.part_id

    def get_vel(self):
        return self.part.v

    def get_mass(self):
        return self.part.mass

    def get_type(self):
        return self.part.type

    def get_force(self):
        return self.part.f

    def set_pos(self, pos):
        self.part.pos = pos

    def set_vel(self, vel):
        self.part.v = vel

    def set_force(self, force):
        self.part.ext_force = force

    def kill_motion(self):
        self.part.fix = [1, 1, 1]

    def unkill_motion(self):
        self.part.unfix()


class Edge:
    """
    Represents edges in a mesh.

    """

    def __init__(self, A, B):
        if not all(isinstance(x, (PartPoint, FixedPoint)) for x in [A, B]):
            TypeError("Arguments to Edge must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B

    def length(self):
        return vec_distance(self.A.get_pos(), self.B.get_pos())


class Triangle:
    """
    Represents triangles in a mesh.

    """

    def __init__(self, A, B, C):
        if not all(isinstance(x, (PartPoint, FixedPoint)) for x in [A, B, C]):
            TypeError("Arguments to Triangle must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B
        self.C = C

    def area(self):
        area = area_triangle(
            self.A.get_pos(), self.B.get_pos(), self.C.get_pos())
        return area


class Angle:
    """
    Represents angles in a mesh.

    """

    def __init__(self, A, B, C, D):
        if not all(isinstance(x, (PartPoint, FixedPoint))
                   for x in [A, B, C, D]):
            TypeError("Arguments to Angle must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def size(self):
        angle_size = angle_btw_triangles(
            self.A.get_pos(), self.B.get_pos(), self.C.get_pos(), self.D.get_pos())
        return angle_size


class ThreeNeighbors:
    """
    Represents three best spatially distributed neighbors of a point in a mesh.

    """

    def __init__(self, A, B, C):
        if not all(isinstance(x, (PartPoint, FixedPoint)) for x in [A, B, C]):
            TypeError(
                "Arguments to ThreeNeighbors must be FixedPoint or PartPoint.")
        self.A = A
        self.B = B
        self.C = C

    def outer_normal(self):
        outer_normal = get_triangle_normal(
            self.A.get_pos(), self.B.get_pos(), self.C.get_pos())
        return outer_normal


class Mesh:
    """
    Represents a triangular mesh.

    """

    def __init__(
            self, nodes_file=None, triangles_file=None, system=None, resize=(1.0, 1.0, 1.0),
            particle_type=-1, particle_mass=1.0, normal=False, check_orientation=True):
        if (system is None) or (not isinstance(system, espressomd.System)):
            raise Exception(
                "Mesh: No system provided or wrong type given. Quitting.")

        self.system = system
        self.normal = normal
        self.nodes_file = nodes_file
        self.triangles_file = triangles_file

        self.points = []
        self.edges = []
        self.triangles = []
        self.angles = []
        self.neighbors = []
        self.ids_extremal_points = [0, 0, 0, 0, 0, 0, 0]

        if not ((nodes_file is None) or (triangles_file is None)):
            if not (isinstance(nodes_file, str)
                    and isinstance(triangles_file, str)):
                raise TypeError("Mesh: Filenames must be strings.")
            if not ((len(resize) == 3) and isinstance(resize[0], float) and isinstance(
                    resize[1], float) and isinstance(resize[2], float)):
                raise TypeError("Mesh: Pos must be a list of three floats.")
            if not isinstance(particle_type, int):
                raise TypeError("Mesh: particle_type must be integer.")
            if not isinstance(particle_mass, float):
                raise TypeError("Mesh: particle_mass must be float.")
            if not isinstance(normal, bool):
                raise TypeError("Mesh: normal must be bool.")
            if not isinstance(check_orientation, bool):
                raise TypeError("Mesh: check_orientation must be bool.")
            # reading the mesh point positions from file
            in_file = open(nodes_file, "r")
            nodes_coord = in_file.read().split("\n")
            in_file.close()
            # removes a blank line at the end of the file if there is any:
            nodes_coord = filter(None, nodes_coord)
            # here we have list of lines with triplets of strings
            for line in nodes_coord:  # extracts coordinates from the string line
                line = np.array([float(x) for x in line.split()])
                coords = np.array(resize) * line
                tmp_fixed_point = FixedPoint(coords, len(self.points))
                self.points.append(tmp_fixed_point)

            # searching for extremal points IDs
            x_min = large_number
            x_max = -large_number
            y_min = large_number
            y_max = -large_number
            z_min = large_number
            z_max = -large_number
            for tmp_fixed_point in self.points:
                coords = tmp_fixed_point.get_pos()
                if coords[0] < x_min:
                    x_min = coords[0]
                    self.ids_extremal_points[0] = tmp_fixed_point.get_id()
                if coords[0] > x_max:
                    x_max = coords[0]
                    self.ids_extremal_points[1] = tmp_fixed_point.get_id()
                if coords[1] < y_min:
                    y_min = coords[1]
                    self.ids_extremal_points[2] = tmp_fixed_point.get_id()
                if coords[1] > y_max:
                    y_max = coords[1]
                    self.ids_extremal_points[3] = tmp_fixed_point.get_id()
                if coords[2] < z_min:
                    z_min = coords[2]
                    self.ids_extremal_points[4] = tmp_fixed_point.get_id()
                if coords[2] > z_max:
                    z_max = coords[2]
                    self.ids_extremal_points[5] = tmp_fixed_point.get_id()

            # reading the triangle incidences from file
            in_file = open(triangles_file, "r")
            triangles_incid = in_file.read().split("\n")
            in_file.close()
            # removes a blank line at the end of the file if there is any:
            triangles_incid = filter(None, triangles_incid)
            for line in triangles_incid:  # extracts incidences from the string line
                incid = np.array([int(x) for x in line.split()])
                tmp_triangle = Triangle(
                    self.points[incid[0]], self.points[incid[1]], self.points[incid[2]])
                self.triangles.append(tmp_triangle)

            if check_orientation is True:
                # check whether all triangles in file had the same orientation;
                # if not, correct the orientation
                self.check_orientation()

            # creating list of edge incidences from triangle incidences
            # using temporary list of edge incidences
            tmp_edge_incidences = []
            for triangle in self.triangles:
                pa = triangle.A.id
                pb = triangle.B.id
                pc = triangle.C.id
                if ([pa, pb] not in tmp_edge_incidences) and (
                        [pb, pa] not in tmp_edge_incidences):
                    tmp_edge_incidences.append([pa, pb])
                if ([pb, pc] not in tmp_edge_incidences) and (
                        [pc, pb] not in tmp_edge_incidences):
                    tmp_edge_incidences.append([pb, pc])
                if ([pa, pc] not in tmp_edge_incidences) and (
                        [pc, pa] not in tmp_edge_incidences):
                    tmp_edge_incidences.append([pa, pc])
            for tmp_incid in tmp_edge_incidences:
                tmp_edge = Edge(
                    self.points[tmp_incid[0]], self.points[tmp_incid[1]])
                self.edges.append(tmp_edge)

            # creating list angles (former bending incidences) from triangle
            # incidences
            for edge in self.edges:
                pa = edge.A.id
                pb = edge.B.id
                pc = -1
                pd = -1
                detected = 0
                # detected = number of detected triangles with current edge common
                # Algorithm is as follows: we run over all triangles and check
                # whether two vertices are those from current edge. If we find such triangle,
                # we put the ID of the third vertex to pc and we check if the orientation pa, pb, pc is the same as
                # was in the triangle list (meaning, that we found one of the following three triples
                # in the triangle list: pa, pb, pc or pb, pc, pa or pc, pa, pb).
                # If we have the same orientation, we set orient = 1, otherwise orient = -1.
                # Then we go further looking for the second triangle.
                # The second triangle should have the opposite orientation.
                # The normal of the first triangle will be P1P2 x P1P3, of the
                # second triangle will be P2P4 x P2P3
                orient = 0
                for triangle in self.triangles:
                    # Run over all triangles and determine the two triangles
                    # with the common current edge
                    if (pa == triangle.A.id) and (pb == triangle.B.id):
                        if detected == 0:
                            # if no triangle with such edge was detected before
                            pc = triangle.C.id
                            detected = 1
                            orient = 1
                        else:
                            # if this is the second triangle with this edge,
                            # then also quit the for-loop over triangles
                            pd = triangle.C.id
                            break
                    if (pa == triangle.B.id) and (pb == triangle.C.id):
                        if detected == 0:
                            pc = triangle.A.id
                            detected = 1
                            orient = 1
                        else:
                            pd = triangle.A.id
                            break
                    if (pa == triangle.C.id) and (pb == triangle.A.id):
                        if detected == 0:
                            pc = triangle.B.id
                            detected = 1
                            orient = 1
                        else:
                            pd = triangle.B.id
                            break
                    if (pa == triangle.B.id) and (pb == triangle.A.id):
                        if detected == 0:
                            pc = triangle.C.id
                            detected = 1
                            orient = -1
                        else:
                            pd = triangle.C.id
                            break
                    if (pa == triangle.C.id) and (pb == triangle.B.id):
                        if detected == 0:
                            pc = triangle.A.id
                            detected = 1
                            orient = -1
                        else:
                            pd = triangle.A.id
                            break
                    if (pa == triangle.A.id) and (pb == triangle.C.id):
                        if detected == 0:
                            pc = triangle.B.id
                            detected = 1
                            orient = -1
                        else:
                            pd = triangle.B.id
                            break
                if orient == 1:
                    tmp = pd
                    pd = pc
                    pc = tmp
                tmp_angle = Angle(
                    self.points[pc], self.points[pa], self.points[pb], self.points[pd])
                self.angles.append(tmp_angle)

            # fills the list of neighbours for each mesh point
            for point in self.points:
                for edge in self.edges:
                    if edge.A.id == point.id:
                        point.neighbour_ids.append(edge.B.id)
                    if edge.B.id == point.id:
                        point.neighbour_ids.append(edge.A.id)

            # creating list of three neighbors for membrane collision
            if normal is True:
                for point in self.points:
                    tmp_neighbors = []
                    # cycle through edges and select those that contain point
                    for edge in self.edges:
                        # take an edge and copy the nodes of the edge to pa, pb
                        if edge.A.id == point.id:
                            tmp_neighbors.append(edge.B)
                        if edge.B.id == point.id:
                            tmp_neighbors.append(edge.A)
                    # create vectors to all neighbors and normalize them
                    tmp_vectors_to_neighbors = []
                    p_coords = np.array(point.get_pos())
                    for neighbor in tmp_neighbors:
                        tmp_vector = neighbor.get_pos() - p_coords
                        tmp_length = norm(tmp_vector)
                        if tmp_length < small_epsilon:
                            raise Exception("Mesh: Degenerate edge. Quitting.")
                        tmp_vector /= tmp_length
                        tmp_vectors_to_neighbors.append(tmp_vector)
                    # check all triplets of neighbors and select the one that is best spatially distributed
                    # by adding the corresponding three normalized vectors
                    # and selecting the one with smallest resultant vector
                    n_neighbors = len(tmp_neighbors)
                    min_length = large_number
                    best_neighbors = [
                        tmp_neighbors[0], tmp_neighbors[1], tmp_neighbors[2]]
                    for i in range(0, n_neighbors):
                        for j in range(i + 1, n_neighbors):
                            for k in range(j + 1, n_neighbors):
                                tmp_result_vector = tmp_vectors_to_neighbors[i] + tmp_vectors_to_neighbors[j] + \
                                                    tmp_vectors_to_neighbors[k]
                                tmp_result_vector_length = norm(
                                    tmp_result_vector)
                                if tmp_result_vector_length < min_length:
                                    min_length = tmp_result_vector_length
                                    best_neighbors = [
                                        tmp_neighbors[i], tmp_neighbors[j], tmp_neighbors[k]]
                    # find one triangle that contains this point and compute
                    # its normal vector
                    for triangle in self.triangles:
                        if triangle.A.id == point.id or triangle.B.id == point.id or triangle.C.id == point.id:
                            tmp_normal_triangle = get_triangle_normal(
                                triangle.A.get_pos(), triangle.B.get_pos(),
                                triangle.C.get_pos())
                            break
                    # properly orient selected neighbors and save them to the
                    # list of neighbors
                    tmp_normal_neighbors = get_triangle_normal(
                        best_neighbors[
                            0].get_pos(), best_neighbors[1].get_pos(),
                        best_neighbors[2].get_pos())
                    tmp_length_normal_triangle = norm(tmp_normal_triangle)
                    tmp_length_normal_neighbors = norm(tmp_normal_neighbors)
                    tmp_product = np.dot(tmp_normal_triangle, tmp_normal_neighbors) / \
                                  (tmp_length_normal_triangle *
                                   tmp_length_normal_neighbors)
                    tmp_angle = np.arccos(tmp_product)
                    if tmp_angle > np.pi / 2.0:
                        selected_neighbors = ThreeNeighbors(
                            best_neighbors[0], best_neighbors[1], best_neighbors[2])
                    else:
                        selected_neighbors = ThreeNeighbors(
                            best_neighbors[0], best_neighbors[2], best_neighbors[1])
                    self.neighbors.append(selected_neighbors)
            else:
                for point in self.points:
                    selected_neighbors = ThreeNeighbors(point, point, point)
                    self.neighbors.append(selected_neighbors)

    def copy(self, origin=None, particle_type=-1, particle_mass=1.0,
             rotate=None):
        mesh = Mesh(system=self.system)
        mesh.ids_extremal_points = self.ids_extremal_points
        rotation = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])

        if rotate is not None:
            # variables for rotation
            ca = np.cos(rotate[0])
            sa = np.sin(rotate[0])
            cb = np.cos(rotate[1])
            sb = np.sin(rotate[1])
            cc = np.cos(rotate[2])
            sc = np.sin(rotate[2])
            rotation = np.array(
                [[cb * cc, sa * sb * cc - ca * sc, sc * sa + cc * sb * ca],
                 [cb * sc, ca * cc + sa * sb *
                  sc, sc * sb * ca - cc * sa],
                 [-sb, cb * sa, ca * cb]])
        for point in self.points:
            # PartPoints are created
            tmp_pos = point.get_pos()
            tmp_rotate_pos = np.array(point.get_pos())
            # rotation of nodes
            if rotate is not None:
                tmp_pos = rotation.dot(tmp_rotate_pos)
                tmp_pos = [discard_epsilon(tmp_pos[0]), discard_epsilon(
                    tmp_pos[1]), discard_epsilon(tmp_pos[2])]
            if origin is not None:
                tmp_pos += np.array(origin)
            # to remember the global id of the ESPResSo particle
            new_part_id = len(self.system.part)
            self.system.part.add(
                pos=tmp_pos, type=particle_type, mass=particle_mass, mol_id=particle_type)
            new_part = self.system.part[new_part_id]
            new_part_point = PartPoint(new_part, len(mesh.points), new_part_id)
            new_part_point.neighbour_ids = point.neighbour_ids
            mesh.points.append(new_part_point)
        for edge in self.edges:
            new_edge = Edge(mesh.points[edge.A.id], mesh.points[edge.B.id])
            mesh.edges.append(new_edge)
        for triangle in self.triangles:
            new_triangle = Triangle(
                mesh.points[triangle.A.id], mesh.points[triangle.B.id], mesh.points[triangle.C.id])
            mesh.triangles.append(new_triangle)
        for angle in self.angles:
            new_angle = Angle(
                mesh.points[angle.A.id], mesh.points[
                    angle.B.id], mesh.points[angle.C.id],
                mesh.points[angle.D.id])
            mesh.angles.append(new_angle)
        for neighbors in self.neighbors:
            new_neighbors = ThreeNeighbors(
                mesh.points[neighbors.A.id], mesh.points[neighbors.B.id],
                mesh.points[neighbors.C.id])
            mesh.neighbors.append(new_neighbors)
        return mesh

    def check_orientation(self):
        tmp_triangle_list = []
        tmp_triangle_list_ok = []
        t_ok = None
        corrected_triangle = None
        for triangle in self.triangles:
            tmp_triangle_list.append(triangle)

        # move the first triangle to the checked and corrected list
        tmp_triangle_list_ok.append(tmp_triangle_list[0])
        tmp_triangle_list.pop(0)

        while tmp_triangle_list:
            i = 0
            while i < len(tmp_triangle_list):
                tmp_triangle = tmp_triangle_list[i]
                for correct_triangle in tmp_triangle_list_ok:
                    # check if triangles have a common edge, if so, check
                    # orientation
                    are_neighbors = True
                    if tmp_triangle.A.id == correct_triangle.A.id:
                        if tmp_triangle.B.id == correct_triangle.B.id:
                            t_ok = False  # this is situation 123 and 124
                            corrected_triangle = Triangle(
                                tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                        else:
                            if tmp_triangle.B.id == correct_triangle.C.id:
                                t_ok = True  # this is situation 123 and 142
                            else:
                                if tmp_triangle.C.id == correct_triangle.B.id:
                                    t_ok = True  # this is situation 123 and 134
                                else:
                                    if tmp_triangle.C.id == correct_triangle.C.id:
                                        t_ok = False  # this is situation 123 and 143
                                        corrected_triangle = Triangle(
                                            tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                                    else:
                                        are_neighbors = False
                    else:
                        if tmp_triangle.A.id == correct_triangle.B.id:
                            if tmp_triangle.B.id == correct_triangle.C.id:
                                t_ok = False  # this is situation 123 and 412
                                corrected_triangle = Triangle(
                                    tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                            else:
                                if tmp_triangle.B.id == correct_triangle.A.id:
                                    t_ok = True  # this is situation 123 and 214
                                else:
                                    if tmp_triangle.C.id == correct_triangle.C.id:
                                        t_ok = True  # this is situation 123 and 413
                                    else:
                                        if tmp_triangle.C.id == correct_triangle.A.id:
                                            t_ok = False  # this is situation 123 and 314
                                            corrected_triangle = Triangle(
                                                tmp_triangle.A, tmp_triangle.C,
                                                tmp_triangle.B)
                                        else:
                                            are_neighbors = False
                        else:
                            if tmp_triangle.A.id == correct_triangle.C.id:
                                if tmp_triangle.B.id == correct_triangle.A.id:
                                    t_ok = False  # this is situation 123 and 241
                                    corrected_triangle = Triangle(
                                        tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                                else:
                                    if tmp_triangle.B.id == correct_triangle.B.id:
                                        t_ok = True  # this is situation 123 and 421
                                    else:
                                        if tmp_triangle.C.id == correct_triangle.A.id:
                                            t_ok = True  # this is situation 123 and 341
                                        else:
                                            if tmp_triangle.C.id == correct_triangle.B.id:
                                                t_ok = False  # this is situation 123 and 431
                                                corrected_triangle = Triangle(
                                                    tmp_triangle.A, tmp_triangle.C,
                                                    tmp_triangle.B)
                                            else:
                                                are_neighbors = False
                            else:
                                if tmp_triangle.B.id == correct_triangle.A.id:
                                    if tmp_triangle.C.id == correct_triangle.B.id:
                                        t_ok = False  # this is situation 123 and 234
                                        corrected_triangle = Triangle(
                                            tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                                    else:
                                        if tmp_triangle.C.id == correct_triangle.C.id:
                                            t_ok = True  # this is situation 123 and 243
                                        else:
                                            are_neighbors = False
                                else:
                                    if tmp_triangle.B.id == correct_triangle.B.id:
                                        if tmp_triangle.C.id == correct_triangle.C.id:
                                            t_ok = False  # this is situation 123 and 423
                                            corrected_triangle = Triangle(
                                                tmp_triangle.A, tmp_triangle.C,
                                                tmp_triangle.B)
                                        else:
                                            if tmp_triangle.C.id == correct_triangle.A.id:
                                                t_ok = True  # this is situation 123 and 324
                                            else:
                                                are_neighbors = False
                                    else:
                                        if tmp_triangle.B.id == correct_triangle.C.id:
                                            if tmp_triangle.C.id == correct_triangle.A.id:
                                                t_ok = False  # this is situation 123 and 342
                                                corrected_triangle = Triangle(
                                                    tmp_triangle.A, tmp_triangle.C,
                                                    tmp_triangle.B)
                                            else:
                                                if tmp_triangle.C.id == correct_triangle.B.id:
                                                    t_ok = True  # this is situation 123 and 432
                                                else:
                                                    are_neighbors = False
                                        else:
                                            are_neighbors = False
                    if are_neighbors:
                        # move the tmp_triangle to the checked and corrected
                        # list
                        if t_ok:
                            tmp_triangle_list_ok.append(tmp_triangle)
                        else:
                            tmp_triangle_list_ok.append(corrected_triangle)
                        tmp_triangle_list.pop(i)
                        break
                i += 1
        # replace triangles with checked triangles
        i = 0
        for tmp_triangle in tmp_triangle_list_ok:
            self.triangles[i] = Triangle(
                tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
            i += 1
        # all triangles now have the same orientation, check if it is correct
        tmp_volume = self.volume()
        if tmp_volume < 0:
            # opposite orientation, flip all triangles
            i = 0
            for tmp_triangle in self.triangles:
                self.triangles[i] = Triangle(
                    tmp_triangle.A, tmp_triangle.C, tmp_triangle.B)
                i += 1
        return 0

    def min_edge_length(self):
        min_length = large_number
        i = 0
        for edge in self.edges:
            i = i + 1
            l = edge.length()
            if (l < min_length):
                min_length = l
        if (i == 0):
            raise Exception("Mesh, min_edge_length: No edges. Quitting.")
        return min_length

    def total_fluid_force(self, lbfluid, friction):
        total_force = np.array([0.0, 0.0, 0.0])
        for p in self.points:
            vel_point = p.get_vel()
            pos_point = p.get_pos()
            vel_fluid = lbfluid.get_interpolated_velocity(pos_point)
            total_force += - friction * (vel_point - vel_fluid)
        return total_force

    def max_edge_length(self):
        max_length = - large_number
        i = 0
        for edge in self.edges:
            i = i + 1
            l = edge.length()
            if (l > max_length):
                max_length = l
        if (i == 0):
            raise Exception("Mesh, max_edge_length: No edges. Quitting.")
        return max_length

    def gen_new_mesh(self, filename):
        output_file = open(filename, "w")
        # orig = self.get_origin()
        c0 = 0.207161
        c1 = 2.002558
        c2 = -1.122762
        r0 = 3.91
        for p in self.points:
            pos = p.get_pos()
            w = (pos[0] * pos[0] + pos[2] * pos[2]) / (1.0 * r0 * r0)
            if (w > 1):
                w = 1.0
            new_z = 0.5 * r0 * np.sqrt(1 - w) * (c0 + c1 * w + c2 * w * w)
            if (pos[1] < 0):
                new_z = - 1.0 * new_z
            output_file.write(str(pos[0]) + " " + str(new_z) + " " + str(pos[2]) + "\n")
        output_file.close()
        return 1

    def aver_edge_length(self):
        edge_aver = 0.0;
        i = 0
        for edge in self.edges:
            i = i + 1
            edge_aver = edge_aver + edge.length()
        if (i == 0):
            raise Exception("Mesh, aver_edge_length: No edges. Quitting.")
        edge_aver = edge_aver / (1.0 * i)
        return edge_aver

    def stdev_edge_length(self):
        edge_aver = self.aver_edge_length();
        stdev = 0
        i = 0
        for edge in self.edges:
            stdev = stdev + (edge_aver - edge.length()) * (edge_aver - edge.length())
            i = i + 1
        if (i == 0):
            raise Exception("Mesh, stdev_edge_length: No edges. Quitting.")
        stdev = stdev / (1.0 * (i - 1))
        stdev = np.sqrt(stdev)
        return stdev

    def print_analysis(self):
        print("\t n_nodes: " + str(self.get_n_nodes()))
        print("\t n_triangles: " + str(self.get_n_triangles()))
        print("\t n_edges: " + str(self.get_n_edges()))
        print("\t aver_edge_length: " + str(self.aver_edge_length()))
        print("\t min_edge_length: " + str(self.min_edge_length()))
        print("\t max_edge_length: " + str(self.max_edge_length()))
        print("\t stdev_edge_length: " + str(self.stdev_edge_length()))

    def surface(self):
        surface = 0.0
        for triangle in self.triangles:
            surface += triangle.area()
        return surface

    def volume(self):
        volume = 0.0
        for triangle in self.triangles:
            tmp_normal = get_triangle_normal(
                triangle.A.get_pos(), triangle.B.get_pos(), triangle.C.get_pos())
            tmp_normal_length = norm(tmp_normal)
            tmp_sum_z_coords = 1.0 / 3.0 * \
                               (triangle.A.get_pos()[2] + triangle.B.get_pos()[2] +
                                triangle.C.get_pos()[2])
            volume -= (triangle.area() * tmp_normal[2] / tmp_normal_length *
                       tmp_sum_z_coords)
        return volume

    def get_n_nodes(self):
        return len(self.points)

    def get_n_triangles(self):
        return len(self.triangles)

    def get_n_edges(self):
        return len(self.edges)

    def output_mesh_triangles(self, triangles_file=None):
        # this is useful after the mesh correction
        # output of mesh nodes can be done from OifCell (this is because their
        # position may change)
        if triangles_file is None:
            raise Exception(
                "OifMesh: No file_name provided for triangles. Quitting.")
        output_file = open(triangles_file, "w")
        for t in self.triangles:
            output_file.write(
                str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.close()
        return 0

    def mirror(self, mirror_x=0, mirror_y=0, mirror_z=0, out_file_name=""):
        if out_file_name == "":
            raise Exception(
                "Cell.Mirror: output meshnodes file for new mesh is missing. Quitting.")
        if mirror_x not in (0, 1) or mirror_y not in (
                0, 1) or mirror_z not in (0, 1):
            raise Exception(
                "Mesh.Mirror: for mirroring only values 0 or 1 are accepted. 1 indicates that the corresponding coordinate will be flipped.  Exiting.")
        if mirror_x + mirror_y + mirror_z > 1:
            raise Exception(
                "Mesh.Mirror: flipping allowed only for one axis. Exiting.")
        if mirror_x + mirror_y + mirror_z == 1:
            out_file = open(out_file_name, "w")
            for p in self.points:
                coor = p.get_pos()
                if mirror_x == 1:
                    coor[0] *= -1.0
                if mirror_y == 1:
                    coor[1] *= -1.0
                if mirror_z == 1:
                    coor[2] *= -1.0
                out_file.write(custom_str(coor[0]) + " " + custom_str(
                    coor[1]) + " " + custom_str(coor[2]) + "\n")
            out_file.close()
        return 0


class OifCellType:  # analogous to oif_template

    """
    Represents a template for creating elastic objects.

    """

    def __init__(
            self, nodes_file="", triangles_file="", system=None, resize=(1.0, 1.0, 1.0), ks=0.0, kslin=0.0,
            kb=0.0, kal=0.0, kag=0.0, kv=0.0, kvisc=0.0, normal=False, check_orientation=True):
        if (system is None) or (not isinstance(system, espressomd.System)):
            raise Exception(
                "OifCellType: No system provided or wrong type. Quitting.")
        if (nodes_file == "") or (triangles_file == ""):
            raise Exception(
                "OifCellType: One of nodesfile or trianglesfile is missing. Quitting.")
        if not (isinstance(nodes_file, str)
                and isinstance(triangles_file, str)):
            raise TypeError("OifCellType: Filenames must be strings.")
        if not ((len(resize) == 3) and isinstance(resize[0], float) and isinstance(
                resize[1], float) and isinstance(resize[2], float)):
            raise TypeError(
                "OifCellType: Resize must be a list of three floats.")
        if not (isinstance(ks, float) and isinstance(kslin, float) and isinstance(kb, float) and isinstance(
                kal, float) and isinstance(kag, float) and isinstance(kv, float) and isinstance(kvisc, float)):
            raise TypeError("OifCellType: Elastic parameters must be floats.")
        if not isinstance(normal, bool):
            raise TypeError("OifCellType: normal must be bool.")
        if not isinstance(check_orientation, bool):
            raise TypeError("OifCellType: check_orientation must be bool.")
        if (ks != 0.0) and (kslin != 0.0):
            raise Exception(
                "OifCellType: Cannot use linear and nonlinear stretching at the same time. Quitting.")
        self.system = system
        self.mesh = Mesh(
            nodes_file=nodes_file, triangles_file=triangles_file, system=system, resize=resize,
            normal=normal, check_orientation=check_orientation)
        self.local_force_interactions = []
        self.resize = resize
        self.ks = ks
        self.kslin = kslin
        self.kb = kb
        self.kal = kal
        self.kag = kag
        self.kv = kv
        self.kvisc = kvisc
        self.normal = normal
        r_cut_global = 0.0
        if (ks != 0.0) or (kslin != 0.0) or (kb != 0.0) or (kal != 0.0) or (kvisc != 0.0):
            for angle in self.mesh.angles:
                r0 = vec_distance(angle.B.get_pos(), angle.C.get_pos())
                if r0 > r_cut_global:
                    r_cut_global = r0
                phi = angle_btw_triangles(
                    angle.A.get_pos(), angle.B.get_pos(), angle.C.get_pos(), angle.D.get_pos())
                area1 = area_triangle(
                    angle.A.get_pos(), angle.B.get_pos(), angle.C.get_pos())
                area2 = area_triangle(
                    angle.D.get_pos(), angle.B.get_pos(), angle.C.get_pos())
                tmp_local_force_inter = OifLocalForces(
                    r0=r0, ks=ks, kslin=kslin, phi0=phi, kb=kb, A01=area1, A02=area2,
                    kal=kal, kvisc=kvisc)
                self.local_force_interactions.append(
                    [tmp_local_force_inter, [angle.A, angle.B, angle.C, angle.D]])
                self.system.bonded_inter.add(tmp_local_force_inter)
        r_cut_global *= 5.0
        if (kag != 0.0) or (kv != 0.0):
            surface = self.mesh.surface()
            volume = self.mesh.volume()
            self.global_force_interaction = OifGlobalForces(
                A0_g=surface, ka_g=kag, V0=volume, kv=kv, r_cut=r_cut_global)
            self.system.bonded_inter.add(self.global_force_interaction)

    def print_info(self):
        print("\nThe following OifCellType was created: ")
        print("\t nodes_file: " + self.mesh.nodes_file)
        print("\t triangles_file: " + self.mesh.triangles_file)
        print("\t n_nodes: " + str(self.mesh.get_n_nodes()))
        print("\t n_triangles: " + str(self.mesh.get_n_triangles()))
        print("\t n_edges: " + str(self.mesh.get_n_edges()))
        print("\t ks: " + custom_str(self.ks))
        print("\t kslin: " + custom_str(self.kslin))
        print("\t kb: " + custom_str(self.kb))
        print("\t kal: " + custom_str(self.kal))
        print("\t kag: " + custom_str(self.kag))
        print("\t kv: " + custom_str(self.kv))
        print("\t kvisc: " + custom_str(self.kvisc))
        print("\t normal: " + str(self.normal))
        print("\t resize: " + str(self.resize))
        print(" ")

    def suggest_LBgamma(self, visc=None, dens=None):
        if not (isinstance(visc, float) and isinstance(dens, float)):
            raise Exception(
                "OifCellType: viscosity or density must be real numbers in suggest_LBgamma. Quitting.")
        noNodes = self.mesh.get_n_nodes()
        surface = self.mesh.surface()

        LBgamma = (393.0 / (1.0 * noNodes)) * np.sqrt(surface / 201.0619) * (
                (5.6 - 1.82) / (5.853658537 - 1.5) * (visc - 1.5) + (10 - 1.82) / (6 - 1.025) * (
                dens - 1.025) + 1.82)
        return LBgamma


class OifCell:
    """
    Represents a concrete elastic object.

    """

    def __init__(self, cell_type=None, origin=None, particle_type=None,
                 particle_mass=1.0, rotate=None, exclusion_neighbours=True, rotation_ids=[]):
        if (cell_type is None) or (not isinstance(cell_type, OifCellType)):
            raise Exception(
                "OifCell: No cellType provided or wrong type. Quitting.")
        if (origin is None) or \
                (not ((len(origin) == 3) and isinstance(origin[0], float) and isinstance(origin[1],
                                                                                         float) and isinstance(
                    origin[2], float))):
            raise TypeError("Origin must be tuple.")
        if (particle_type is None) or (not isinstance(particle_type, int)):
            raise Exception(
                "OifCell: No particle_type specified or wrong type. Quitting.")
        if not isinstance(particle_mass, float):
            raise Exception("OifCell: particle mass must be float.")
        if (rotate is not None) and not ((len(rotate) == 3) and isinstance(
                rotate[0], float) and isinstance(rotate[1], float) and isinstance(rotate[2], float)):
            raise TypeError("Rotate must be list of three floats.")
        if not isinstance(exclusion_neighbours, bool):
            raise Exception("OifCell: exclusion_neighbours must be bool.")

        self.cell_type = cell_type
        self.cell_type.system.max_oif_objects = self.cell_type.system.max_oif_objects + 1
        self.mesh = cell_type.mesh.copy(
            origin=origin, particle_type=particle_type, particle_mass=particle_mass, rotate=rotate)
        self.particle_mass = particle_mass
        self.particle_type = particle_type
        self.origin = origin
        self.rotate = rotate
        self.rot_ids = []
        self.rot_init_pos = []
        rot_ids_ok = 1
        for it in rotation_ids:
            if isinstance(it, int) != True:
                rot_ids_ok = 0
            it = int(it)
            if it < 0 or it >= len(self.mesh.points):
                rot_ids_ok = 0
        if rot_ids_ok == 0:
            raise Exception(
                "OifCell: rotation_ids must be a tuple of integer valued ids between 0 and the number of mesh points.")
        self.set_rotation(rotation_ids)

        for inter in self.cell_type.local_force_interactions:
            esp_inter = inter[0]
            points = inter[1]
            n_points = len(points)
            if n_points == 2:
                p0 = self.mesh.points[
                    points[0].id]  # Getting PartPoints from id's of FixedPoints
                p1 = self.mesh.points[points[1].id]
                p0.part.add_bond((esp_inter, p1.part_id))
            if n_points == 3:
                p0 = self.mesh.points[points[0].id]
                p1 = self.mesh.points[points[1].id]
                p2 = self.mesh.points[points[2].id]
                p0.part.add_bond((esp_inter, p1.part_id, p2.part_id))
            if n_points == 4:
                p0 = self.mesh.points[points[0].id]
                p1 = self.mesh.points[points[1].id]
                p2 = self.mesh.points[points[2].id]
                p3 = self.mesh.points[points[3].id]
                p1.part.add_bond(
                    (esp_inter, p0.part_id, p2.part_id, p3.part_id))

        if (self.cell_type.kag != 0.0) or (self.cell_type.kv != 0.0):
            for triangle in self.mesh.triangles:
                triangle.A.part.add_bond(
                    (self.cell_type.global_force_interaction, triangle.B.part_id,
                     triangle.C.part_id))

        # setting the out_direction interaction for membrane collision
        if self.cell_type.mesh.normal is True:
            tmp_out_direction_interaction = OifOutDirection()
            # this interaction could be just one for all objects, but here it
            # is created multiple times
            self.cell_type.system.bonded_inter.add(
                tmp_out_direction_interaction)
            for p in self.mesh.points:
                p.part.add_bond(
                    (tmp_out_direction_interaction, self.mesh.neighbors[
                        p.id].A.part_id,
                     self.mesh.neighbors[p.id].B.part_id, self.mesh.neighbors[p.id].C.part_id))

        if exclusion_neighbours is True:
            #            for edge in self.mesh.edges:
            #                self.cell_type.system.part[edge.A.part_id].add_exclusion(edge.B.part_id)
            for point in self.mesh.points:
                excl = []
                for id in point.neighbour_ids:
                    if id > point.id:
                        if id not in excl:
                            excl.append(id)
                    for idd in self.mesh.points[id].neighbour_ids:
                        if idd > point.id:
                            if idd not in excl:
                                excl.append(idd)
                for id in excl:
                    self.cell_type.system.part[point.part_id].add_exclusion(self.mesh.points[id].part_id)

                excl = []
                for id in point.neighbour_ids:
                    if id != point.id:
                        if id not in excl:
                            excl.append(id)
                    for idd in self.mesh.points[id].neighbour_ids:
                        if idd != point.id:
                            if idd not in excl:
                                excl.append(idd)
                point.exclusions = excl

    def set_rotation(self, ids=[]):
        if ids == []:
            self.rot_ids = [-1, -1, -1, -1, -1, -1]
            # searching for extremal points IDs
            x_min = large_number
            x_max = -large_number
            y_min = large_number
            y_max = -large_number
            z_min = large_number
            z_max = -large_number
            for tmp_part_point in self.cell_type.mesh.points:
                coords = tmp_part_point.get_pos()
                if coords[0] < x_min:
                    x_min = coords[0]
                    self.rot_ids[0] = tmp_part_point.get_id()
                if coords[0] > x_max:
                    x_max = coords[0]
                    self.rot_ids[1] = tmp_part_point.get_id()
                if coords[1] < y_min:
                    y_min = coords[1]
                    self.rot_ids[2] = tmp_part_point.get_id()
                if coords[1] > y_max:
                    y_max = coords[1]
                    self.rot_ids[3] = tmp_part_point.get_id()
                if coords[2] < z_min:
                    z_min = coords[2]
                    self.rot_ids[4] = tmp_part_point.get_id()
                if coords[2] > z_max:
                    z_max = coords[2]
                    self.rot_ids[5] = tmp_part_point.get_id()
        else:
            ok = 1
            for it in ids:
                if isinstance(it, int) != True:
                    ok = 0
                it = int(it)
                if it < 0 or it >= len(self.mesh.points):
                    ok = 0
            if ok == 0:
                raise Exception(
                    "OifCell: set_rotation: rotation_ids must be a tuple of integer valued ids between 0 and the number of mesh points.")
            self.rot_ids = ids

        self.rot_init_pos = []
        orig = self.get_origin()
        for it in self.rot_ids:
            pos = list(self.mesh.points[it].get_pos())
            for ii in range(0, 3):
                pos[ii] = pos[ii] - orig[ii]
            self.rot_init_pos.append(pos)

    def get_rotation_angles(self):
        i = 0
        ang = []
        orig = self.get_origin()
        for it in self.rot_ids:
            vec_cur = list(self.mesh.points[it].get_pos())
            vec_init = self.rot_init_pos[i]
            for ii in range(0, 3):
                vec_cur[ii] = vec_cur[ii] - orig[ii]
            ang.append(angle_btw_vectors(vec_init, vec_cur))
            i = i + 1
        return ang

    def get_rotation_positions(self):
        pos = []
        orig = self.get_origin()
        for it in self.rot_ids:
            tmp = list(self.mesh.points[it].get_pos())
            for ii in range(0, 3):
                tmp[ii] = tmp[ii] - orig[ii]
            pos.append(tmp)
        return pos

    def get_origin(self):
        center = np.array([0.0, 0.0, 0.0])
        for p in self.mesh.points:
            center += p.get_pos()
        return center / len(self.mesh.points)

    def set_origin(self, new_origin=(0.0, 0.0, 0.0)):
        old_origin = self.get_origin()
        for p in self.mesh.points:
            new_position = p.get_pos() - old_origin + new_origin
            p.set_pos(new_position)

    def get_approx_origin(self):
        approx_center = np.array([0.0, 0.0, 0.0])
        for id in self.mesh.ids_extremal_points:
            approx_center += self.mesh.points[id].get_pos()
        return approx_center / len(self.mesh.ids_extremal_points)

    def get_origin_folded(self):
        origin = self.get_origin()
        return np.mod(origin, self.cell_type.system.box_l)

    def get_velocity(self):
        velocity = np.array([0.0, 0.0, 0.0])
        for p in self.mesh.points:
            velocity += p.get_vel()
        return velocity / len(self.mesh.points)

    def set_velocity(self, new_velocity=(0.0, 0.0, 0.0)):
        for p in self.mesh.points:
            p.set_vel(new_velocity)

    def pos_bounds(self):
        x_min = large_number
        x_max = -large_number
        y_min = large_number
        y_max = -large_number
        z_min = large_number
        z_max = -large_number
        for p in self.mesh.points:
            coords = p.get_pos()
            if coords[0] < x_min:
                x_min = coords[0]
            if coords[0] > x_max:
                x_max = coords[0]
            if coords[1] < y_min:
                y_min = coords[1]
            if coords[1] > y_max:
                y_max = coords[1]
            if coords[2] < z_min:
                z_min = coords[2]
            if coords[2] > z_max:
                z_max = coords[2]
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def point_bound(self):
        x_min = large_number
        x_max = -large_number
        y_min = large_number
        y_max = -large_number
        z_min = large_number
        z_max = -large_number
        for p in self.mesh.points:
            coords = p.get_pos()
            if coords[0] < x_min:
                x_min = coords[0]
                x_min_point = p
            if coords[0] > x_max:
                x_max = coords[0]
                x_max_point = p
            if coords[1] < y_min:
                y_min = coords[1]
                y_min_point = p
            if coords[1] > y_max:
                y_max = coords[1]
                y_max_point = p
            if coords[2] < z_min:
                z_min = coords[2]
                z_min_point = p
            if coords[2] > z_max:
                z_max = coords[2]
                z_max_point = p
        return [x_min_point, x_max_point, y_min_point, y_max_point, z_min_point, z_max_point]

    def surface(self):
        return self.mesh.surface()

    def print_mesh_analysis(self):
        return self.mesh.print_analysis()

    def min_edge_length(self):
        return self.mesh.min_edge_length()

    def total_fluid_force(self, lbfluid, friction):
        return self.mesh.total_fluid_force(lbfluid, friction)

    def max_edge_length(self):
        return self.mesh.max_edge_length()

    def aver_edge_length(self):
        return self.mesh.aver_edge_length()

    def stdev_edge_length(self):
        return self.mesh.stdev_edge_length()

    def volume(self):
        return self.mesh.volume()

    def suggest_LBgamma(self, visc=None, dens=None):
        return self.cell_type.suggest_LBgamma(visc=visc, dens=dens)

    def diameter(self):
        max_distance = 0.0
        n_points = len(self.mesh.points)
        for i in range(0, n_points):
            for j in range(i + 1, n_points):
                p1 = self.mesh.points[i].get_pos()
                p2 = self.mesh.points[j].get_pos()
                tmp_dist = vec_distance(p1, p2)
                if tmp_dist > max_distance:
                    max_distance = tmp_dist
        return max_distance

    def get_n_nodes(self):
        return self.mesh.get_n_nodes()

    def set_force(self, new_force=(0.0, 0.0, 0.0)):
        for p in self.mesh.points:
            p.set_force(new_force)

    # this is not implemented
    # def kill_motion(self):
    #    for p in self.mesh.points:
    #        p.kill_motion()

    # this is not implemented
    # def unkill_motion(self):
    #    for p in self.mesh.points:
    #        p.unkill_motion()

    def output_vtk_pos(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for vtk output. Quitting")
        n_points = len(self.mesh.points)
        n_triangles = len(self.mesh.triangles)
        output_file = open(file_name, "w")
        output_file.write("# vtk DataFile Version 3.0\n")
        output_file.write("Data\n")
        output_file.write("ASCII\n")
        output_file.write("DATASET POLYDATA\n")
        output_file.write("POINTS " + str(n_points) + " float\n")
        for p in self.mesh.points:
            coords = p.get_pos()
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.write("TRIANGLE_STRIPS " + str(
            n_triangles) + " " + str(4 * n_triangles) + "\n")
        for t in self.mesh.triangles:
            output_file.write(
                "3 " + str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.close()

    def output_vtk_point_data(self, file_name=None, point_id=None, data_type=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for vtk output. Quitting")
        if point_id is None:
            raise Exception(
                "OifCell: No point_id provided for vtk output. Quitting")
        if point_id < 0 or point_id >= len(self.mesh.points):
            raise Exception(
                "OifCell: point_id is negative or larger than number of mesh points. Quitting")
        if (data_type != "neighbours" and data_type != "exclusions"):
            raise Exception(
                "OifCell: Wrong data_type provided, exclusions or neighbours allowed. Quitting")

        n_points = len(self.mesh.points)
        n_triangles = len(self.mesh.triangles)
        output_file = open(file_name, "w")
        output_file.write("# vtk DataFile Version 3.0\n")
        output_file.write("Data\n")
        output_file.write("ASCII\n")
        output_file.write("DATASET POLYDATA\n")
        output_file.write("POINTS " + str(n_points) + " float\n")
        for p in self.mesh.points:
            coords = p.get_pos()
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.write("TRIANGLE_STRIPS " + str(
            n_triangles) + " " + str(4 * n_triangles) + "\n")
        for t in self.mesh.triangles:
            output_file.write(
                "3 " + str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.write("POINT_DATA " + str(n_points) + "\n")
        output_file.write("SCALARS neighbours float 1\n")
        output_file.write("LOOKUP_TABLE default\n")
        tmp_point = self.mesh.points[point_id]
        for p in self.mesh.points:
            if p.id == point_id:
                output_file.write("1.5\n")
            else:
                if p.id in tmp_point.neighbour_ids:
                    output_file.write("1.0\n")
                else:
                    if (p.id in tmp_point.exclusions) and (data_type == "exclusions"):
                        output_file.write("0.5\n")
                    else:
                        output_file.write("0.0\n")
        output_file.close()

    def output_vtk_pos_folded(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for vtk output. Quitting.")
        n_points = len(self.mesh.points)
        n_triangles = len(self.mesh.triangles)

        # get coordinates of the origin
        center = np.array([0.0, 0.0, 0.0])
        for p in self.mesh.points:
            center += p.get_pos()
        center /= len(self.mesh.points)
        center_folded = np.floor(center / self.cell_type.system.box_l)
        # this gives how many times the origin is folded in all three
        # directions

        output_file = open(file_name, "w")
        output_file.write("# vtk DataFile Version 3.0\n")
        output_file.write("Data\n")
        output_file.write("ASCII\n")
        output_file.write("DATASET POLYDATA\n")
        output_file.write("POINTS " + str(n_points) + " float\n")
        for p in self.mesh.points:
            coords = p.get_pos() - center_folded * self.cell_type.system.box_l
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.write("TRIANGLE_STRIPS " + str(
            n_triangles) + " " + str(4 * n_triangles) + "\n")
        for t in self.mesh.triangles:
            output_file.write(
                "3 " + str(t.A.id) + " " + str(t.B.id) + " " + str(t.C.id) + "\n")
        output_file.close()

    def append_point_data_to_vtk(self, file_name=None, data_name=None,
                                 data=None, first_append=None):
        if file_name is None:
            raise Exception(
                "OifCell: append_point_data_to_vtk: No file_name provided. Quitting.")
        if data is None:
            raise Exception(
                "OifCell: append_point_data_to_vtk: No data provided. Quitting.")
        if data_name is None:
            raise Exception(
                "OifCell: append_point_data_to_vtk: No data_name provided. Quitting.")
        if first_append is None:
            raise Exception("OifCell: append_point_data_to_vtk: Need to know whether this is the first data list to be "
                            "appended for this file. Quitting.")
        n_points = self.get_n_nodes()
        if len(data) != n_points:
            raise Exception(
                "OifCell: append_point_data_to_vtk: Number of data points does not match number of mesh points. Quitting.")
        output_file = open(file_name, "a")
        if first_append is True:
            output_file.write("POINT_DATA " + str(n_points) + "\n")
        output_file.write("SCALARS " + data_name + " float 1\n")
        output_file.write("LOOKUP_TABLE default\n")
        for p in self.mesh.points:
            output_file.write(str(data[p.id]) + "\n")
        output_file.close()

    def output_raw_data(self, file_name=None, data=None):
        if file_name is None:
            raise Exception(
                "OifCell: output_raw_data: No file_name provided. Quitting.")
        if data is None:
            raise Exception(
                "OifCell: output_raw_data: No data provided. Quitting.")
        n_points = self.get_n_nodes()
        if len(data) != n_points:
            raise Exception(
                "OifCell: output_raw_data: Number of data points does not match number of mesh points. Quitting.")
        output_file = open(file_name, "w")
        for p in self.mesh.points:
            output_file.write(" ".join(map(str, data[p.id])) + "\n")
        output_file.close()

    def output_mesh_points(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for mesh nodes output. Quitting.")
        output_file = open(file_name, "w")
        center = self.get_origin()
        for p in self.mesh.points:
            coords = p.get_pos() - center
            output_file.write(custom_str(coords[0]) + " " + custom_str(
                coords[1]) + " " + custom_str(coords[2]) + "\n")
        output_file.close()

    def set_mesh_points(self, file_name=None):
        if file_name is None:
            raise Exception(
                "OifCell: No file_name provided for set_mesh_points. Quitting.")
        center = self.get_origin()
        n_points = self.get_n_nodes()

        in_file = open(file_name, "r")
        nodes_coord = in_file.read().split("\n")
        in_file.close()
        # removes a blank line at the end of the file if there is any:
        nodes_coord = list(filter(None, nodes_coord))
        # here we have list of lines with triplets of strings
        if len(nodes_coord) != n_points:
            raise Exception("OifCell: Mesh nodes not set to new positions: "
                            "number of lines in the file does not equal number of Cell nodes. Quitting.")
        else:
            i = 0
            for line in nodes_coord:  # extracts coordinates from the string line
                line = line.split()
                new_position = np.array(line).astype(np.float) + center
                self.mesh.points[i].set_pos(new_position)
                i += 1

    def print_info(self):
        print("\nThe following OifCell was created: ")
        print("\t particle_mass: " + custom_str(self.particle_mass))
        print("\t particle_type: " + str(self.particle_type))
        print("\t rotate: " + str(self.rotate))
        print("\t origin: " + str(self.origin[0]) + " " + str(
            self.origin[1]) + " " + str(self.origin[2]))

    def elastic_forces(
            self, el_forces=(0, 0, 0, 0, 0, 0), f_metric=(0, 0, 0, 0, 0, 0), vtk_file=None,
            raw_data_file=None):
        # the order of parameters in elastic_forces and in f_metric is as follows (ks, kb, kal, kag, kv, total)
        # vtk_file means that a vtk file for visualisation of elastic forces will be written
        # raw_data_file means that just the elastic forces will be written into
        # the output file

        stretching_forces_list = []
        bending_forces_list = []
        local_area_forces_list = []
        global_area_forces_list = []
        volume_forces_list = []
        elastic_forces_list = []
        stretching_forces_norms_list = []
        bending_forces_norms_list = []
        local_area_forces_norms_list = []
        global_area_forces_norms_list = []
        volume_forces_norms_list = []
        elastic_forces_norms_list = []
        ks_f_metric = 0.0
        kb_f_metric = 0.0
        kal_f_metric = 0.0
        kag_f_metric = 0.0
        kv_f_metric = 0.0
        total_f_metric = 0.0

        for i in range(0, 6):
            if (el_forces[i] != 0) and (el_forces[i] != 1):
                raise Exception(
                    "OifCell: elastic_forces: Incorrect argument. el_forces has to be a sixtuple of 0s and 1s, "
                    "specifying which elastic forces will be calculated. The order in the sixtuple is (ks, kb, "
                    "kal, kag, kv, total).")
        for i in range(0, 6):
            if (f_metric[i] != 0) and (f_metric[i] != 1):
                raise Exception(
                    "OifCell: elastic_forces: Incorrect argument. f_metric has to be a sixtuple of 0s and 1s, "
                    "specifying which f_metric will be calculated. The order in the sixtuple is (ks, kb, kal, "
                    "kag, kv, total)")
        # calculation of stretching forces and f_metric
        if (el_forces[0] == 1) or (el_forces[5] == 1) or (
                f_metric[0] == 1) or (f_metric[5] == 1):
            # initialize list
            stretching_forces_list = []
            for p in self.mesh.points:
                stretching_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses edges, but results are stored for nodes
            for e in self.mesh.edges:
                a_current_pos = e.A.get_pos()
                b_current_pos = e.B.get_pos()
                a_orig_pos = self.cell_type.mesh.points[e.A.id].get_pos()
                b_orig_pos = self.cell_type.mesh.points[e.B.id].get_pos()
                current_dist = e.length()
                orig_dist = vec_distance(a_orig_pos, b_orig_pos)
                tmp_stretching_force = oif_calc_stretching_force(
                    self.cell_type.ks, a_current_pos, b_current_pos,
                    orig_dist, current_dist)
                stretching_forces_list[e.A.id] += tmp_stretching_force
                stretching_forces_list[e.B.id] -= tmp_stretching_force
            # calculation of stretching f_metric, if needed
            if f_metric[0] == 1:
                ks_f_metric = 0.0
                for p in self.mesh.points:
                    ks_f_metric += norm(stretching_forces_list[p.id])

        # calculation of bending forces and f_metric
        if (el_forces[1] == 1) or (el_forces[5] == 1) or (
                f_metric[1] == 1) or (f_metric[5] == 1):
            # initialize list
            bending_forces_list = []
            for p in self.mesh.points:
                bending_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses bending incidences, but results are stored for
            # nodes
            for angle in self.mesh.angles:
                a_current_pos = angle.A.get_pos()
                b_current_pos = angle.B.get_pos()
                c_current_pos = angle.C.get_pos()
                d_current_pos = angle.D.get_pos()
                a_orig_pos = self.cell_type.mesh.points[angle.A.id].get_pos()
                b_orig_pos = self.cell_type.mesh.points[angle.B.id].get_pos()
                c_orig_pos = self.cell_type.mesh.points[angle.C.id].get_pos()
                d_orig_pos = self.cell_type.mesh.points[angle.D.id].get_pos()
                current_angle = angle.size()
                orig_angle = angle_btw_triangles(
                    a_orig_pos, b_orig_pos, c_orig_pos, d_orig_pos)
                tmp_bending_forces = oif_calc_bending_force(
                    self.cell_type.kb, a_current_pos, b_current_pos, c_current_pos,
                    d_current_pos, orig_angle, current_angle)
                tmp_bending_force1 = np.array(
                    [tmp_bending_forces[0], tmp_bending_forces[1], tmp_bending_forces[2]])
                tmp_bending_force2 = np.array(
                    [tmp_bending_forces[3], tmp_bending_forces[4], tmp_bending_forces[5]])
                bending_forces_list[angle.A.id] += tmp_bending_force1
                bending_forces_list[angle.B.id] -= 0.5 * \
                                                   tmp_bending_force1 + 0.5 * tmp_bending_force2
                bending_forces_list[angle.C.id] -= 0.5 * \
                                                   tmp_bending_force1 + 0.5 * tmp_bending_force2
                bending_forces_list[angle.D.id] += tmp_bending_force2
            # calculation of bending f_metric, if needed
            if f_metric[1] == 1:
                kb_f_metric = 0.0
                for p in self.mesh.points:
                    kb_f_metric += norm(bending_forces_list[p.id])

        # calculation of local area forces and f_metric
        if (el_forces[2] == 1) or (el_forces[5] == 1) or (
                f_metric[2] == 1) or (f_metric[5] == 1):
            # initialize list
            local_area_forces_list = []
            for p in self.mesh.points:
                local_area_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses triangles, but results are stored for nodes
            for t in self.mesh.triangles:
                a_current_pos = t.A.get_pos()
                b_current_pos = t.B.get_pos()
                c_current_pos = t.C.get_pos()
                a_orig_pos = self.cell_type.mesh.points[t.A.id].get_pos()
                b_orig_pos = self.cell_type.mesh.points[t.B.id].get_pos()
                c_orig_pos = self.cell_type.mesh.points[t.C.id].get_pos()
                current_area = t.area()
                orig_area = area_triangle(a_orig_pos, b_orig_pos, c_orig_pos)
                tmp_local_area_forces = oif_calc_local_area_force(
                    self.cell_type.kal, a_current_pos, b_current_pos,
                    c_current_pos, orig_area, current_area)
                local_area_forces_list[t.A.id] += np.array(
                    [tmp_local_area_forces[0], tmp_local_area_forces[1],
                     tmp_local_area_forces[2]])
                local_area_forces_list[t.B.id] += np.array(
                    [tmp_local_area_forces[3], tmp_local_area_forces[4],
                     tmp_local_area_forces[5]])
                local_area_forces_list[t.C.id] += np.array(
                    [tmp_local_area_forces[6], tmp_local_area_forces[7],
                     tmp_local_area_forces[8]])

            # calculation of local area f_metric, if needed
            if f_metric[2] == 1:
                kal_f_metric = 0.0
                for p in self.mesh.points:
                    kal_f_metric += norm(local_area_forces_list[p.id])

        # calculation of global area forces and f_metric
        if (el_forces[3] == 1) or (el_forces[5] == 1) or (
                f_metric[3] == 1) or (f_metric[5] == 1):
            # initialize list
            global_area_forces_list = []
            for p in self.mesh.points:
                global_area_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses triangles, but results are stored for nodes
            for t in self.mesh.triangles:
                a_current_pos = t.A.get_pos()
                b_current_pos = t.B.get_pos()
                c_current_pos = t.C.get_pos()
                current_surface = self.mesh.surface()
                orig_surface = self.cell_type.mesh.surface()
                tmp_global_area_forces = oif_calc_global_area_force(
                    self.cell_type.kag, a_current_pos, b_current_pos,
                    c_current_pos, orig_surface, current_surface)
                global_area_forces_list[t.A.id] += np.array(
                    [tmp_global_area_forces[0], tmp_global_area_forces[1],
                     tmp_global_area_forces[2]])
                global_area_forces_list[t.B.id] += np.array(
                    [tmp_global_area_forces[3], tmp_global_area_forces[4],
                     tmp_global_area_forces[5]])
                global_area_forces_list[t.C.id] += np.array(
                    [tmp_global_area_forces[6], tmp_global_area_forces[7],
                     tmp_global_area_forces[8]])
            # calculation of global area f_metric, if needed
            if f_metric[3] == 1:
                kag_f_metric = 0.0
                for p in self.mesh.points:
                    kag_f_metric += norm(global_area_forces_list[p.id])

        # calculation of volume forces and f_metric
        if (el_forces[4] == 1) or (el_forces[5] == 1) or (
                f_metric[4] == 1) or (f_metric[5] == 1):
            # initialize list
            volume_forces_list = []
            for p in self.mesh.points:
                volume_forces_list.append([0.0, 0.0, 0.0])
            # calculation uses triangles, but results are stored for nodes
            for t in self.mesh.triangles:
                a_current_pos = t.A.get_pos()
                b_current_pos = t.B.get_pos()
                c_current_pos = t.C.get_pos()
                current_volume = self.mesh.volume()
                orig_volume = self.cell_type.mesh.volume()
                tmp_volume_force = oif_calc_volume_force(
                    self.cell_type.kv, a_current_pos, b_current_pos, c_current_pos,
                    orig_volume, current_volume)
                volume_forces_list[t.A.id] += tmp_volume_force
                volume_forces_list[t.B.id] += tmp_volume_force
                volume_forces_list[t.C.id] += tmp_volume_force
            # calculation of volume f_metric, if needed
            if f_metric[4] == 1:
                kv_f_metric = 0.0
                for p in self.mesh.points:
                    kv_f_metric += norm(volume_forces_list[p.id])

        # calculation of total elastic forces and f_metric
        if (el_forces[5] == 1) or (f_metric[5] == 1):
            elastic_forces_list = []
            for p in self.mesh.points:
                total_elastic_forces = stretching_forces_list[p.id] + bending_forces_list[p.id] + \
                                       local_area_forces_list[p.id] + global_area_forces_list[p.id] + \
                                       volume_forces_list[p.id]
                elastic_forces_list.append(total_elastic_forces)
            # calculation of total f_metric, if needed
            if f_metric[5] == 1:
                total_f_metric = 0.0
                for p in self.mesh.points:
                    total_f_metric += norm(elastic_forces_list[p.id])

        # calculate norms of resulting forces
        if sum(el_forces) != 0:
            if el_forces[0] == 1:
                stretching_forces_norms_list = []
                for p in self.mesh.points:
                    stretching_forces_norms_list.append(
                        norm(stretching_forces_list[p.id]))
            if el_forces[1] == 1:
                bending_forces_norms_list = []
                for p in self.mesh.points:
                    bending_forces_norms_list.append(
                        norm(bending_forces_list[p.id]))
            if el_forces[2] == 1:
                local_area_forces_norms_list = []
                for p in self.mesh.points:
                    local_area_forces_norms_list.append(
                        norm(local_area_forces_list[p.id]))
            if el_forces[3] == 1:
                global_area_forces_norms_list = []
                for p in self.mesh.points:
                    global_area_forces_norms_list.append(
                        norm(global_area_forces_list[p.id]))
            if el_forces[4] == 1:
                volume_forces_norms_list = []
                for p in self.mesh.points:
                    volume_forces_norms_list.append(
                        norm(volume_forces_list[p.id]))
            if el_forces[5] == 1:
                elastic_forces_norms_list = []
                for p in self.mesh.points:
                    elastic_forces_norms_list.append(
                        norm(elastic_forces_list[p.id]))

        # output vtk (folded)
        if vtk_file is not None:
            if el_forces == (0, 0, 0, 0, 0, 0):
                raise Exception("OifCell: elastic_forces: The option elastic_forces was not used. "
                                "Nothing to output to vtk file.")
            self.output_vtk_pos_folded(vtk_file)
            first = True
            if el_forces[0] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="ks_f_metric",
                    data=stretching_forces_norms_list, first_append=first)
                first = False
            if el_forces[1] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kb_f_metric",
                    data=bending_forces_norms_list, first_append=first)
                first = False
            if el_forces[2] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kal_f_metric",
                    data=local_area_forces_norms_list, first_append=first)
                first = False
            if el_forces[3] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kag_f_metric",
                    data=global_area_forces_norms_list, first_append=first)
                first = False
            if el_forces[4] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="kav_f_metric",
                    data=volume_forces_norms_list, first_append=first)
                first = False
            if el_forces[5] == 1:
                self.append_point_data_to_vtk(
                    file_name=vtk_file, data_name="total_f_metric",
                    data=elastic_forces_norms_list, first_append=first)
                first = False

        # output raw data
        if raw_data_file is not None:
            if sum(el_forces) != 1:
                raise Exception("OifCell: elastic_forces: Only one type of elastic forces can be written into one "
                                "raw_data_file. If you need several, please call OifCell.elastic_forces multiple times - "
                                "once per elastic force.")
            if el_forces[0] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=stretching_forces_list)
            if el_forces[1] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=bending_forces_list)
            if el_forces[2] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=local_area_forces_list)
            if el_forces[3] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=global_area_forces_list)
            if el_forces[4] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=volume_forces_list)
            if el_forces[5] == 1:
                self.output_raw_data(
                    file_name=raw_data_file, data=elastic_forces_list)

        # return f_metric
        if f_metric[0] + f_metric[1] + f_metric[2] + \
                f_metric[3] + f_metric[4] + f_metric[5] > 0:
            results = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if f_metric[0] == 1:
                results[0] = ks_f_metric
            if f_metric[1] == 1:
                results[1] = kb_f_metric
            if f_metric[2] == 1:
                results[2] = kal_f_metric
            if f_metric[3] == 1:
                results[3] = kag_f_metric
            if f_metric[4] == 1:
                results[4] = kv_f_metric
            if f_metric[5] == 1:
                results[5] = total_f_metric
            return results
        else:
            return 0


class OifCluster:
    def __init__(self, cells=[]):
        self.cells = cells
        
        # morse interactions
        self.m_eps = None
        self.m_alpha = None
        self.m_cutoff = None
        self.m_rmin = None

        # soft-sphere interactions
        self.soft_a = None
        self.soft_n = None
        self.soft_cutoff = None
        self.soft_offset = None
        
        # "self-cell" self soft-sphere interactions
        self.sc_a = None
        self.sc_n = None
        self.sc_cutoff = None
        self.sc_offset = None
        
        # membrane collision interactions
        self.mc_a = None
        self.mc_n = None
        self.mc_cutoff = None
        self.mc_offset = None

    def get_origin(self):
        center = np.array([0.0, 0.0, 0.0])
        for cell in self.cells:
            center += cell.get_origin()
        return center / len(self.cells)

    def set_origin(self, new_origin=(0.0, 0.0, 0.0)):
        old_origin = self.get_origin()
        for cell in self.cells:
            new_position = cell.get_origin() - old_origin + new_origin
            cell.set_origin(new_position)

    def get_cells(self):
        return self.cells

    def get_n_nodes_cluster(self):
        num_points = 0
        for i in range(len(self.cells)):
            num_points += self.cells[i].get_n_nodes()
        return num_points

    def get_vtk_cluster(self, dir, num):
        for id, cell in enumerate(self.cells):
            cell.output_vtk_pos_folded(file_name=dir + "/cell" + str(id) + "_" + str(num) + ".vtk")

    def get_vtk_proximity_interactions(self,out_file, max_distance=0.5):
        pairs = []
        positions1 = {}
        positions2 = {}
        count = 0

        for k in range(self.get_num_cells() - 1):
            positions1 = {}
            positions2 = {}
            for i in self.cells[k].mesh.points:
                positions1[i.part_id] = i.get_pos()
            for j in self.cells[k + 1].mesh.points:
                positions2[j.part_id] = j.get_pos()

            for i in self.cells[k].mesh.points:
                for j in self.cells[k + 1].mesh.points:
                    ipos = positions1[i.part_id]
                    jpos = positions2[j.part_id]
                    if self.distance_cells(ipos, jpos) <= max_distance:
                        line = [ipos[0], ipos[1], ipos[2], jpos[0], jpos[1], jpos[2]]
                        pairs.append(line)
        output_vtk_lines(lines=pairs, out_file=out_file)

    def add_cells(self, cells):
        self.cells.extend(cells);

    def add_cell(self, cell):
        self.cells.append(cell)

    def set_velocity(self, new_velocity=(0.0, 0.0, 0.0)):
        for c in self.cells:
            c.set_velocity(new_velocity)

    def get_velocity(self):
        velocities = []
        for c in self.cells:
            velocities.append((c.particle_type, c.get_velocity()))
        return velocities

    def get_avg_velocity(self):
        velocity = np.array([0.0, 0.0, 0.0])
        for c in self.cells:
            velocity += c.get_velocity()
        return velocity / len(self.cells)

    def set_rotation(self, rotate=(0.0, 0.0, 0.0)):
        origin = np.array(self.get_origin())
        self.set_origin()
        for cell in self.cells:
            ca = np.cos(rotate[0])
            sa = np.sin(rotate[0])
            cb = np.cos(rotate[1])
            sb = np.sin(rotate[1])
            cc = np.cos(rotate[2])
            sc = np.sin(rotate[2])
            rotation = np.array(
                [[cb * cc, sa * sb * cc - ca * sc, sc * sa + cc * sb * ca],
                 [cb * sc, ca * cc + sa * sb *
                  sc, sc * sb * ca - cc * sa],
                 [-sb, cb * sa, ca * cb]])

            for point in cell.mesh.points:
                tmp_rotate_pos = np.array(point.get_pos())
                tmp_pos = rotation.dot(tmp_rotate_pos)
                tmp_pos = [discard_epsilon(tmp_pos[0]), discard_epsilon(tmp_pos[1]), discard_epsilon(tmp_pos[2])]
                point.set_pos(tmp_pos)
        self.set_origin(origin)

    def pos_bounds(self):
        x_min = large_number
        x_max = -large_number
        y_min = large_number
        y_max = -large_number
        z_min = large_number
        z_max = -large_number
        for c in self.cells:
            cell_pos_bounds = c.pos_bounds()

            if cell_pos_bounds[0] < x_min:
                x_min = cell_pos_bounds[0]
            if cell_pos_bounds[1] > x_max:
                x_max = cell_pos_bounds[1]
            if cell_pos_bounds[2] < y_min:
                y_min = cell_pos_bounds[2]
            if cell_pos_bounds[3] < y_max:
                y_max = cell_pos_bounds[3]
            if cell_pos_bounds[4] < z_min:
                z_min = cell_pos_bounds[4]
            if cell_pos_bounds[5] < z_max:
                z_max = cell_pos_bounds[5]

        return [x_min, x_max, y_min, y_max, z_min, z_max]
        
    def set_cell_boundary_interactions(self, system, boundary_particle_type, soft_a=0.00022, soft_n=0.5, soft_cutoff=0.5, soft_offset=0):
        for cell in self.cells:
            system.non_bonded_inter[cell.particle_type,boundary_particle_type].soft_sphere.set_params(a=soft_a,
                                                                                                      n=soft_n,
                                                                                                      cutoff=soft_cutoff,
                                                                                                      offset=soft_offset)
        
    def set_soft_sphere_interactions(self, system, soft_a=0.00022, soft_n=0.5, soft_cutoff=0.5, soft_offset=0):
        self.soft_a = soft_a
        self.soft_n = soft_n
        self.soft_cutoff = soft_cutoff
        self.soft_offset = soft_offset
    
        for i in range(0,len(self.cells)):
            for j in range(i+1,len(self.cells)):            
                system.non_bonded_inter[i,j].soft_sphere.set_params(a=soft_a,
                                                                    n=soft_n,
                                                                    cutoff=soft_cutoff,
                                                                    offset=soft_offset)
                                                                    
    def set_morse_interactions(self, system, m_eps=0.0145, m_alpha=1.0, m_cutoff=0.7, m_rmin=0.5):
        self.m_eps = m_eps
        self.m_alpha = m_alpha
        self.m_cutoff = m_cutoff
        self.m_rmin = m_rmin
    
        for i in range(0,len(self.cells)):
            for j in range(i+1,len(self.cells)):
                system.non_bonded_inter[i,j].morse.set_params(eps=m_eps,
                                                              alpha=m_alpha,
                                                              cutoff=m_cutoff,
                                                              rmin=m_rmin)
                                                            
    def set_membrane_collision(self, system, mc_a=0.00022, mc_n=1.2, mc_cutoff=0.1, mc_offset=0.0):
        self.mc_a = mc_a
        self.mc_n = mc_n
        self.mc_cutoff = mc_cutoff
        self.mc_offset = mc_offset
    
        for i in range(0,len(self.cells)):
            for j in range(i+1,len(self.cells)):
                system.non_bonded_inter[i, j].membrane_collision.set_params(a=mc_a,
                                                                            n=mc_n,
                                                                            cutoff=mc_cutoff,
                                                                            offset=mc_offset)
  
    def set_self_cell_soft_sphere_interactions(self, system, sc_a=0.0005, sc_n=1.2, sc_cutoff=0.1,sc_offset=0.0):
        self.sc_a = sc_a
        self.sc_n = sc_n
        self.sc_cutoff = sc_cutoff
        self.sc_offset = sc_offset
        
        for i in range(len(self.cells)):
            system.non_bonded_inter[i, i].soft_sphere.set_params(a=sc_a,
                                                                n=sc_n,
                                                                cutoff=sc_cutoff,
                                                                offset=sc_offset)

    # distance between two points in space
    def __distance(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    
    # vector from a to b:  a -> b in three dimensions
    def __direction(self, a, b):
        return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
      
    # made for only 2 cells
    # Attracts cells into cluster center set by these cells  
    # Not tested on multiple cells
    def attract(self, system, vtk_directory, max_deformation_n, cells, min_distance_ratio = 0.8):
        force = 2.0
 
        centroid = np.array([0.0, 0.0, 0.0])
        for cell in cells:
            centroid += cell.get_origin()
        centroid = centroid / len(cells)
      
      
        for cell in cells:
            cell_origin = cell.get_origin()
                  
            distance = self.__distance(centroid, cell_origin)
              
            direction = [0.0, 0.0, 0.0]
                  
            for x in range(0, 3):
                direction[x] = (centroid[x] - cell_origin[x]) / distance
                  
                cell.set_force([force * direction[0] / cell.get_n_nodes(),
                                force * direction[1] / cell.get_n_nodes(),
                                force * direction[2] / cell.get_n_nodes()])
      
        # deformation
        pairs = []
        for i in range(0, max_deformation_n):
            for j, cell in enumerate(cells):
                cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(j) + "_" + str(i) + ".vtk")
              
            if i % 10 == 0:
                min = 20.0
                
                # pairs
                pairs = []
                positions = []
                for cell in cells:
                    mesh_points = []
                    for point in cell.mesh.points:
                        mesh_points.append(point.get_pos())
                    positions.append(mesh_points)
              
                for k,cell_k in enumerate(cells):
                    for kk, part_k in enumerate(cell_k.mesh.points):           
                        kpos = positions[k][kk]                
                        for m,cell_m in enumerate(cells[k+1:len(cells)]):                   
                            for mm, part_m in enumerate(cell_m.mesh.points):
                                mpos = positions[k+m+1][mm]
                                d = self.__distance(kpos, mpos)
                                
                                if d < min:
                                    min = d
                                
                                if d <= 0.7:
                                    line = [kpos[0], kpos[1], kpos[2], mpos[0], mpos[1], mpos[2]]
                                    pairs.append(line)
                                  
                print("min: {}".format(min))
              
                end = False;
                for k in range(len(cells)):
                    for m in range(k+1,len(cells)):
                        distance = self.__distance(cells[k].get_origin(), cells[m].get_origin())
                        if(distance < min_distance_ratio * self.radius * 2 + 0.5):
                            end = True 
                if end == True:
                    break;
                  
            oif.output_vtk_lines(lines = pairs, out_file = vtk_directory + "/lines_" + str(i) + ".vtk")   
            system.integrator.run(steps=100)
      
        print("deformation done")
        # cut force
        for cell in cells:
            cell.set_force([0.0, 0.0, 0.0])
     
    # pulls all cells towards weight center until they create cluster
    # vtk_directory parameter is purely for debugging and final version should not contain it, or any code asociated with exporting vtk
    def deform(self, system, vtk_directory, max_deformation_n, min_distance_ratio = 0.8):
        force = 5.0

        # set force
        centroid = self.get_origin()
        for cell in self.cells:
            cell_origin = cell.get_origin()
                  
            distance = self.__distance(centroid, cell_origin)
              
            direction = [0.0, 0.0, 0.0]
                  
            for x in range(0, 3):
                direction[x] = (centroid[x] - cell_origin[x]) / distance
                  
                cell.set_force([force * direction[0] / cell.get_n_nodes(),
                                force * direction[1] / cell.get_n_nodes(),
                                force * direction[2] / cell.get_n_nodes()])
        pairs = []
        # deformation
        # runs for max_deformation_n cycles * steps 
        # or until any 2 cells reach desired distance set by min_distance_ratio 
        
        steps = 100
        time = 0
        for i in range(0, max_deformation_n):
            for j, cell in enumerate(self.cells):
                cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(j) + "_" + str(i) + ".vtk")
            
            # outputs morse interaction pairt only every x iterations
            if i % 20 == 0 and i != 0:
                min = 20.0
                # pairs
                pairs = []
                positions = []
                for cell in self.cells:
                    mesh_points = []
                    for point in cell.mesh.points:
                        mesh_points.append(point.get_pos())
                    positions.append(mesh_points)
              
                for k,cell_k in enumerate(self.cells):
                    for kk, part_k in enumerate(cell_k.mesh.points):           
                        kpos = positions[k][kk]                
                        for m,cell_m in enumerate(self.cells[k+1:len(self.cells)]):                   
                            for mm, part_m in enumerate(cell_m.mesh.points):
                                mpos = positions[k+m+1][mm]
                                d = self.__distance(kpos, mpos)
                                
                                if d < min:
                                    min = d
                                
                                if d <= 0.7:
                                    line = [kpos[0], kpos[1], kpos[2], mpos[0], mpos[1], mpos[2]]
                                    pairs.append(line)
                                  
                print("min: {}".format(min))
              
                end = False;
                for k in range(len(self.cells)):
                    for m in range(k+1,len(self.cells)):
                        distance = self.__distance(self.cells[k].get_origin(), self.cells[m].get_origin())
                        if(distance < min_distance_ratio * self.radius * 2 + 0.3):
                            end = True 
                if end == True:
                    break;
            oif.output_vtk_lines(lines = pairs, out_file = vtk_directory + "/lines_" + str(i) + ".vtk")
            
            print("time: " + str(time))
            system.integrator.run(steps=steps)
            time = time + steps
      
        print("deformation done")
        # cut force
        for cell in self.cells:
            cell.set_force([0.0, 0.0, 0.0])
      
        for j, cell in enumerate(self.cells):
            cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(j) + "_" + str(max_deformation_n) + ".vtk")
      
        # free ride
        for i in range(1,20):

            print("time: " + str(time))       
            system.integrator.run(steps=100)
            time = time + steps
            for j, cell in enumerate(self.cells):
                cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(j) + "_" + str(max_deformation_n + i) + ".vtk")

    # creates two cell with premade shared interface(contact face)
    # so far only with 2 cells
    # the size of face and also distance between cell centers is set by parameter distance_from_center
    # which dictates, which particles of object will be affected 
    def deform_experiment(self, system, vtk_directory, distance_from_center = 0.9):
        cell_1 = self.cells[2]
        cell_2 = self.cells[0]
        
        cell_distance = self.__distance(cell_1.get_origin(), cell_2.get_origin())    

        cell_1_points = []
        
        for point in cell_1.mesh.points:
            position = point.get_pos()
            d = - cell_1.get_origin()[0] + position[0]
            if(d >= self.radius * 0.8):
                cell_1_points.append(point)
                point.set_pos([cell_1.get_origin()[0] + self.radius * distance_from_center ,position[1],position[2]])
    
        cell_2_points = []
    
        for point in cell_2.mesh.points:
            position = point.get_pos()
            d = cell_2.get_origin()[0] - position[0]
            if(d >= self.radius * 0.8):
                cell_2_points.append(point)
                point.set_pos([cell_2.get_origin()[0] - self.radius * distance_from_center ,position[1],position[2]])
                
        print(len(cell_1_points))
        print(len(cell_2_points))
        
        origin = cell_1.get_origin()
        
        # cells are seeded at the length of self.m_rmin parameter of morse interaction
        # only the second cell is moved, the first stays in its place
        cell_2.set_origin([origin[0] + 2 * distance_from_center * self.radius + self.m_rmin, origin[1], origin[2]])
        
        for i, cell in enumerate(self.cells):
            cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(i) + "_0.vtk")
        
        steps = 100
        for i in range(1, 100):
            print("time: " + str(i * steps))
            system.integrator.run(steps=steps)
            
            for j, cell in enumerate(self.cells):
                cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(j) + "_" + str(i) + ".vtk")

    # not finished
    # deform_experiment method created for multiple objects
    def deform_experiment_v2(self, system, vtk_directory, min_distance_ratio = 0.9):
        cluster = []
        
        cell_1 = self.cells[0]
        
        min_distance = 50
        min_i = 0
        for i in range(1, len(self.cells)):
            d = self.__distance(cell_1.get_origin(), self.cells[i].get_origin())
            if d < min_distance:
                min_distance = d
                min_i = i
        
        cell_2 = self.cells[min_i]

        deform_distance = 2 * self.radius * 0.8
        direction = self.__direction(cell_1.get_origin(), cell_2.get_origin())
        origin = cell_1.get_origin()
            
        cell_1.set_origin([origin[0] + direction[0] * (1-(deform_distance/min_distance))/2,
                           origin[1] + direction[1] * (1-(deform_distance/min_distance))/2,
                           origin[2] + direction[2] * (1-(deform_distance/min_distance))/2])
        
        origin = cell_2.get_origin()
        
        cell_2.set_origin([origin[0] - direction[0] * (1-(deform_distance/min_distance))/2,
                           origin[1] - direction[1] * (1-(deform_distance/min_distance))/2,
                           origin[2] - direction[2] * (1-(deform_distance/min_distance))/2])
        
        
        
        
        
        for i, cell in enumerate(self.cells):
            cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(i) + "_0.vtk")
        
        direction = self.__direction(cell_1.get_origin(), cell_2.get_origin())
        p = [cell_1.get_origin()[0] + direction[0]/2, cell_1.get_origin()[1] + direction[1]/2, cell_1.get_origin()[2] + direction[2]/2]
        
        direction = self.__direction(p, cell_2.get_origin())
        
        or1 = cell_1.get_origin()
        or2 = cell_2.get_origin()
        
        print("cell_1 " + str(or1[0]) + " " + str(or1[1]) + " " + str(or1[2]))
        print("cell_2 " + str(or2[0]) + " " + str(or2[1]) + " " + str(or2[2]))
        
        print("p " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]))
        
        for point in cell_1.mesh.points:
            q = point.get_pos()
            d = self.__distance(cell_2.get_origin(), q)
            if(d <= self.radius):
                #smer point -> origin
                v = [q[0] - p[0],q[1] - p[1],q[2] - p[2]]
                
                #normal vector
                n = [0, 0, 0]
                
                dist = v[0] * n[0] + v[1] * n[1] + v[2] * n[2]
                
                point.set_pos([q[0] - dist*n[0],q[1] - dist*n[1], q[2] - dist*n[2]])
                
        
        
        """
        for i, cell in enumerate(self.cells):
            cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(i) + "_0.vtk")
        
        for point in cell_1.mesh.points:
            q = point.get_pos()
            d = self.__distance(cell_2.get_origin(), q)
            if(d <= self.radius):
                
                pqn = [(q[0] - p[0]) * -direction[0], (q[1] - p[1]) * -direction[1], (q[2] - p[2]) * -direction[2]]
                point.set_pos([q[0] + pqn[0],q[1] + pqn[1],q[2] + pqn[2]])
        """
        for i, cell in enumerate(self.cells):
            cell.output_vtk_pos_folded(file_name=vtk_directory + "/cell" + str(i) + "_1.vtk")
    
    def save_cluster(self, directory, save_interactions=True):
        import os
        import json
        import shutil
        os.makedirs(directory)
        
        nodes_files = []
        triangles_files = []
        cell_types = []
        
        # centers the cluster into [0,0,0] position -> it has to be reverted at the end of the script.
        old_origin = self.get_origin()
        self.set_origin([0,0,0])

        data = {}
        data['cells'] = []
        
        for cell in self.cells:
            cellType = cell.cell_type
            
            # only compares the pointer, not object variable values
            if(cellType not in cell_types):
                cell_types.append(cellType)
                    
            mesh = cell.mesh
            
            if not cellType.mesh.nodes_file in nodes_files:
                nodes_files.append(cellType.mesh.nodes_file)
                
            if not cellType.mesh.triangles_file in triangles_files:
                triangles_files.append(cellType.mesh.triangles_file)
            
            # cell 
            json_cell = {}
            # index of array in cell_types json
            json_cell['cell_type'] = cell_types.index(cellType)
            json_cell['origin'] = cell.origin
            json_cell['particle_type'] = cell.particle_type
            json_cell['particle_mass'] = cell.particle_mass
                
            positions = []
            for point in mesh.points:
                pos = point.get_pos()
                positions.append([pos[0],pos[1],pos[2]])
            json_cell['positions'] = positions
            
            data['cells'].append(json_cell)
            
        # cell type    
        json_cell_types = {}
        i = 0;
        for cell_type in cell_types:
            json_cell_type = {}
            # mesh
            json_cell_type['nodes_file'] = os.path.basename(cell_type.mesh.nodes_file)
            json_cell_type['triangles_file'] = os.path.basename(cell_type.mesh.triangles_file)
            
            # cell type variables
            json_cell_type['resize'] = cell_type.resize
            json_cell_type['ks'] = cell_type.ks
            json_cell_type['kslin'] = cell_type.kslin
            json_cell_type['kb'] = cell_type.kb
            json_cell_type['kal'] = cell_type.kal
            json_cell_type['kag'] = cell_type.kag
            json_cell_type['kv'] = cell_type.kv
            json_cell_type['kvisc'] = cell_type.kvisc
            json_cell_type['normal'] = cell_type.normal
            
            json_cell_types[i] = json_cell_type
            i = i + 1
        
        
        data['cell_types'] = json_cell_types
            
        # interactions saving
        if(save_interactions):
            interactions = {}
            
            # morse
            if(self.m_eps is not None and self.m_alpha is not None and
               self.m_cutoff is not None and self.m_rmin is not None):
               morse = {}
               morse['m_eps'] = self.m_eps
               morse['m_alpha'] = self.m_alpha
               morse['m_cutoff'] = self.m_cutoff
               morse['m_rmin'] = self.m_rmin
               
               interactions['morse'] = morse
            
            # soft-sphere
            if(self.soft_a is not None and self.soft_n is not None and
               self.soft_cutoff is not None and self.soft_offset is not None):
               soft_sphere = {}
               soft_sphere['soft_a'] = self.soft_a
               soft_sphere['soft_n'] = self.soft_n
               soft_sphere['soft_cutoff'] = self.soft_cutoff
               soft_sphere['soft_offset'] = self.soft_offset
               
               interactions['soft_sphere'] = soft_sphere
            
            # self-cell soft-sphere
            if(self.sc_a is not None and self.sc_n is not None and
               self.sc_cutoff is not None and self.sc_offset is not None):
               self_cell = {}
               self_cell['sc_a'] = self.sc_a
               self_cell['sc_n'] = self.sc_n
               self_cell['sc_cutoff'] = self.sc_cutoff
               self_cell['sc_offset'] = self.sc_offset
               
               interactions['self_cell'] = self_cell
               
            # membrane collision
            if(self.mc_a is not None and self.mc_n is not None and
               self.mc_cutoff is not None and self.mc_offset is not None):
               membrane_collision = {}
               membrane_collision['mc_a'] = self.mc_a
               membrane_collision['mc_n'] = self.mc_n
               membrane_collision['mc_cutoff'] = self.mc_cutoff
               membrane_collision['mc_offset'] = self.mc_offset
               
               interactions['membrane_collision'] = membrane_collision
            
            data['interactions'] = interactions
        
        with open(directory + '/data.json', 'w') as outfile:
            json.dump(data, outfile)
        
        os.makedirs(directory + "/nodes_files")
        os.makedirs(directory + "/triangles_files")
        
        for file in nodes_files:
            shutil.copyfile(file, directory + "/nodes_files/" + os.path.basename(file))
        for file in triangles_files:
            shutil.copyfile(file, directory + "/triangles_files/" + os.path.basename(file))
       
        # reverts cluster origin into it's previous position
        self.set_origin(old_origin)
            
       
    def load_cluster(self, system, directory, origin=[0,0,0], load_interactions=True):
        import json
        f = open(directory + '/data.json',)  
        data = json.load(f)
        
        json_cell_types = data['cell_types']
        cell_types = [None] * len(json_cell_types)
        
        for type_key, cell_type in json_cell_types.items():
            typeCell = oif.OifCellType(nodes_file=directory + '/nodes_files/' + cell_type['nodes_file'],
                                       triangles_file=directory + '/triangles_files/' + cell_type['triangles_file'],
                                       check_orientation=False,
                                       system=system,
                                       ks=cell_type['ks'],
                                       kslin=cell_type['kslin'],
                                       kb=cell_type['kb'],
                                       kal=cell_type['kal'],
                                       kag=cell_type['kag'],
                                       kv=cell_type['kv'],
                                       kvisc=cell_type['kvisc'],
                                       normal=cell_type['normal'],
                                       resize=cell_type['resize'])
                                       
            cell_types[int(type_key)] = typeCell                   
              
        json_cells = data['cells']
        cells = []
        
        for index, cell in enumerate(json_cells):
            cell_type_key = cell['cell_type']
            cellType = cell_types[int(cell_type_key)]
            
            
            new_cell = oif.OifCell(cell_type=cellType,
                                   particle_type=cell['particle_type'],
                                   origin=cell['origin'],
                                   particle_mass=cell['particle_mass'],
                                   rotate=None,
                                   exclusion_neighbours=False)
            
            positions = cell['positions']
            
            for i, point in enumerate(new_cell.mesh.points):
                point.set_pos(positions[i])
                
            cells.append(new_cell)
            
        self.add_cells(cells)
        self.set_origin(origin)
        
        if load_interactions:        
            json_interactions = data['interactions']
            if 'morse' in json_interactions:
                morse = json_interactions['morse']
                
                self.set_morse_interactions(system=system, m_eps=morse['m_eps'], 
                                                           m_alpha=morse['m_alpha'], 
                                                           m_cutoff=morse['m_cutoff'],
                                                           m_rmin=morse['m_rmin'])
            if 'soft_sphere' in json_interactions:
                soft_sphere = json_interactions['soft_sphere']
                
                self.set_soft_sphere_interactions(system=system, soft_a=soft_sphere['soft_a'],
                                                                 soft_n=soft_sphere['soft_n'], 
                                                                 soft_cutoff=soft_sphere['soft_cutoff'], 
                                                                 soft_offset=soft_sphere['soft_offset'])
             
            if 'self_cell' in json_interactions:
                self_cell = json_interactions['self_cell']
                
                self.set_self_cell_soft_sphere_interactions(system=system, sc_a=self_cell['sc_a'], 
                                                                           sc_n=self_cell['sc_n'], 
                                                                           sc_cutoff=self_cell['sc_cutoff'],
                                                                           sc_offset=self_cell['sc_offset'])
                
            if 'membrane_collision' in json_interactions:
                membrane_collision = json_interactions['membrane_collision']
               
                self.set_membrane_collision(system=system, mc_a=membrane_collision['mc_a'], 
                                                           mc_n=membrane_collision['mc_n'], 
                                                           mc_cutoff=membrane_collision['mc_cutoff'], 
                                                           mc_offset=membrane_collision['mc_offset'])

class StarCluster(OifCluster):
    def __init__(self, cell_type=None, radius=4, cluster_centroid=[0.0, 0.0, 0.0]):
        self.cell_type = cell_type
        self.radius = radius
        self.cell_positions = []
        super(StarCluster, self).__init__([]);
        self.createPositions(radius, cluster_centroid)
        self.createCells()

    def createPositions(self, radius, cluster_centroid):
        self.cell_positions = [];
        space = 0.5;
        advance = 2 * radius + space
        [x, y, z] = cluster_centroid
        self.cell_positions.append([x, y, z])
        self.cell_positions.append([x, y, z + advance])
        self.cell_positions.append([x, y, z - advance])
        self.cell_positions.append([x - advance, y, z + radius + 0.5])
        self.cell_positions.append([x - advance, y, z - radius])
        self.cell_positions.append([x + advance, y, z + radius + 0.5])
        self.cell_positions.append([x + advance, y, z - radius])

    def createCells(self):
        for i in range(len(self.cell_positions)):
            self.add_cell(oif.OifCell(cell_type=self.cell_type,
                                      particle_type=i,
                                      origin=self.cell_positions[i],
                                      particle_mass=0.5,
                                      rotate=None,
                                      exclusion_neighbours=False))


class DiamondCluster(OifCluster):
    def __init__(self, cell_type=None, radius=4, cluster_centroid=[0.0, 0.0, 0.0]):
        self.cell_type = cell_type
        self.radius = radius
        self.cell_positions = []
        super(DiamondCluster, self).__init__([]);
        self.createPositions(radius, cluster_centroid)
        self.createCells()

    def createPositions(self, radius, cluster_centroid, space=1.0):
        self.cell_positions = []

        [x, y, z] = cluster_centroid

        taznica_length = ((radius * 2 + space) ** 2.0 - (radius + space / 2.0) ** 2.0) ** (1 / 2)
        tazisko_distance_from_base = taznica_length * (1.0 / 3.0)
        tazisko_distance_from_top = taznica_length * (2.0 / 3.0)
        taznica_length_from_center = ((radius * 2.0 + space) ** 2.0 - (tazisko_distance_from_top) ** 2.0) ** (1 / 2)

        self.cell_positions.append([x - tazisko_distance_from_base, y + (radius + space / 2.0), z])
        self.cell_positions.append([x - tazisko_distance_from_base, y - (radius + space / 2.0), z])
        self.cell_positions.append([x + tazisko_distance_from_top, y, z])
        self.cell_positions.append([x, y, z + taznica_length_from_center])
        self.cell_positions.append([x, y, z - taznica_length_from_center])

    def createCells(self):
        for i in range(len(self.cell_positions)):
            # ak pridane do oif_classes -> nieje potrebne oif.
            self.add_cell(OifCell(cell_type=self.cell_type,
                                  particle_type=i,
                                  origin=self.cell_positions[i],
                                  particle_mass=0.5,
                                  rotate=None,
                                  exclusion_neighbours=False))

class ChainCluster(OifCluster):
    def __init__(self, cell_type=None, radius=4, number_of_cells = 5, angle = 15,space = 0.5, cluster_start=[0.0, 0.0, 0.0], start_number_particle = 1, system = None):
        self.cell_type = cell_type
        self.radius = radius
        self.cell_positions = []
        self.numberOfCells = number_of_cells
        self.angle = angle
        self.space = space
        self.start_number_particle = start_number_particle
        self.system = system
        super(ChainCluster, self).__init__([]);
        self.createPositions(radius, cluster_start, angle, number_of_cells)
        self.createCells()
        self.createInteractions()
        self.pressCells()

    def createPositions(self, radius, cluster_start, angle, number_of_cells):
        anglesurface1 = 0
        anglesurface2 = 0
        self.cell_positions = [];
        space = 10;
        advance = 2 * radius + space
        [x, y, z] = cluster_start
        self.cell_positions.append([x, y, z])


        for i in range(number_of_cells - 1):
            alfa = random.randrange(-angle, angle);
            beta = random.randrange(-angle, angle);
            anglesurface1 = anglesurface1 + alfa
            anglesurface2 = anglesurface2 + beta

            x = x + math.cos(math.radians(anglesurface1)) * math.cos(math.radians(anglesurface2)) * advance
            y = y + math.sin(math.radians(anglesurface1)) * math.cos(math.radians(anglesurface2)) * advance
            z = z + math.sin(math.radians(anglesurface2)) * advance

            self.cell_positions.append([x, y, z])

    def createCells(self):
        for i in range(len(self.cell_positions)):
            self.add_cell(OifCell(cell_type=self.cell_type,
                                      particle_type=i,
                                      origin=self.cell_positions[i],
                                      particle_mass=0.5,
                                      rotate=None,
                                      exclusion_neighbours=False))

    def createInteractions(self):
        self.enable_sphere_on_cluster(start_cell_type = self.start_number_particle, system=self.system)
        self.enable_morse_on_cluster(start_cell_type=self.start_number_particle, system=self.system)

    def pressCells(self):
        counter = 0
        steps_in_one_cycle = 100  # integration steps
        cycle = 0
        press_f = 5.0

        anglesurface1 = 0;
        anglesurface2 = 0;
        advance = 2 * self.radius + self.space

        for m in range(len(self.cells) - 1):
            # move m+1 cell
            alfa = random.randrange(-self.angle, self.angle);
            beta = random.randrange(-self.angle, self.angle);
            anglesurface1 = anglesurface1 + alfa
            anglesurface2 = anglesurface2 + beta

            x = self.cells[m].get_origin()[0] + math.cos(math.radians(anglesurface1)) * math.cos(math.radians(anglesurface2)) * advance
            y = self.cells[m].get_origin()[1] + math.sin(math.radians(anglesurface1)) * math.cos(math.radians(anglesurface2)) * advance
            z = self.cells[m].get_origin()[2] + math.sin(math.radians(anglesurface2)) * advance

            self.cells[m + 1].set_origin([x, y, z])

            oA = self.cells[m].get_origin()
            oB = self.cells[m + 1].get_origin()
            dAB = np.sqrt(
                (oA[0] - oB[0]) * (oA[0] - oB[0]) + (oA[1] - oB[1]) * (oA[1] - oB[1]) + (oA[2] - oB[2]) * (
                            oA[2] - oB[2]))

            unitAB = [0.0, 0.0, 0.0]
            for i in range(0, 3):
                unitAB[i] = (oB[i] - oA[i]) / dAB

            self.cells[m].set_force(
                [press_f * unitAB[0] / self.cells[m].get_n_nodes(), press_f * unitAB[1] / self.cells[m].get_n_nodes(),
                 press_f * unitAB[2] / self.cells[m].get_n_nodes()])
            self.cells[m + 1].set_force(
                [-press_f * unitAB[0] / self.cells[m + 1].get_n_nodes(), -press_f * unitAB[1] / self.cells[m + 1].get_n_nodes(),
                 -press_f * unitAB[2] / self.cells[m + 1].get_n_nodes()])

            # vzd buniek stop podmienka
            distCells = dAB;


            while distCells > (0.9 * (2 * self.radius) + self.space):
                self.system.integrator.run(steps=steps_in_one_cycle)
                counter += 1
                cycle = counter * steps_in_one_cycle

                oA = self.cells[m].get_origin()
                oB = self.cells[m + 1].get_origin()
                distCells = np.sqrt(
                    (oA[0] - oB[0]) * (oA[0] - oB[0]) + (oA[1] - oB[1]) * (oA[1] - oB[1]) + (oA[2] - oB[2]) * (
                                oA[2] - oB[2]))

            self.cells[m].set_force([0.0, 0.0, 0.0])
            self.cells[m + 1].set_force([0.0, 0.0, 0.0])

            # 5 steps free running
            stopCycle = (counter + 5) * steps_in_one_cycle

            while cycle < stopCycle:

                self.system.integrator.run(steps=steps_in_one_cycle)
                counter += 1
                cycle = counter * steps_in_one_cycle

class DifferentSizesCluster(OifCluster):
    def __init__(self, cell_type, r1, r2, r3, r4, system, cluster_start = [0.0, 30.0, 20.0],  start_number_particle = 1, space = 0.5, cutoff=0.7):
        self.cell_type = cell_type
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.space = space
        self.cutoff = cutoff
        self.system = system
        self.start_number_particle = start_number_particle
        self.cell_positions = []
        self.cell_radius = [r1, r2, r3, r4]
        super(DifferentSizesCluster, self).__init__([]);
        self.createPositions(cluster_start)
        self.createCells()
        self.createInteractions()
        self.createCluster(cluster_start)

    def createPositions(self, cluster_start, space=1.0):
        self.cell_positions = []
        [x, y, z] = cluster_start

        self.cell_positions.append([x, y, z])
        self.cell_positions.append([25.0, y + 15.0, z + 10.0])
        self.cell_positions.append([45.0, y + 15.0, z + 10.0])
        self.cell_positions.append([65.0, y + 15.0, z + 10.0])

    def createCells(self):
        for i in range(len(self.cell_positions)):
          self.add_cell(OifCell(cell_type=self.cell_type[i],
                                  particle_type=i,
                                  origin=self.cell_positions[i],
                                  particle_mass=0.5,
                                  rotate=None,
                                  exclusion_neighbours=False))

        self.get_vtk_cluster(dir = "output/sim"+str(3) + "/vtk", num=0)

    def createInteractions(self):
        self.enable_sphere_on_cluster(start_cell_type = self.start_number_particle, system=self.system)
        self.enable_morse_on_cluster(start_cell_type=self.start_number_particle, system=self.system)

    def createCluster(self, cluster_start, space=0.8):
      counter = 0
      steps_in_one_cycle = 100  # integration steps
      cycle = 0
      press_f = 5.0
      number = 1

      for m in range(len(self.cells) - 1):
          # move m+1 cell
          advance = self.cell_radius[m] + self.cell_radius[m+1] + space
          x_new = self.cells[m].get_origin()[0] + advance
          y_new = self.cells[m].get_origin()[1]
          z_new = self.cells[m].get_origin()[2]
          self.cells[m + 1].set_origin([x_new, y_new, z_new])

          oA = self.cells[m].get_origin()
          oB = self.cells[m + 1].get_origin()
          dAB = np.sqrt(
              (oA[0] - oB[0]) * (oA[0] - oB[0]) + (oA[1] - oB[1]) * (oA[1] - oB[1]) + (oA[2] - oB[2]) * (
                          oA[2] - oB[2]))

          unitAB = [0.0, 0.0, 0.0]
          for i in range(0, 3):
              unitAB[i] = (oB[i] - oA[i]) / dAB

          self.cells[m].set_force(
              [press_f * unitAB[0] / self.cells[m].get_n_nodes(), press_f * unitAB[1] / self.cells[m].get_n_nodes(),
               press_f * unitAB[2] / self.cells[m].get_n_nodes()])
          self.cells[m + 1].set_force(
              [-press_f * unitAB[0] / self.cells[m + 1].get_n_nodes(), -press_f * unitAB[1] / self.cells[m + 1].get_n_nodes(),
               -press_f * unitAB[2] / self.cells[m + 1].get_n_nodes()])

          # vzd buniek stop podmienka
          distCells = dAB;

          while distCells > (0.7 * (2*self.cell_radius[m]) + self.cell_radius[m+1] + self.space):
              self.system.integrator.run(steps=steps_in_one_cycle)
              counter += 1
              cycle = counter * steps_in_one_cycle

              oA = self.cells[m].get_origin()
              oB = self.cells[m + 1].get_origin()
              distCells = np.sqrt(
                  (oA[0] - oB[0]) * (oA[0] - oB[0]) + (oA[1] - oB[1]) * (oA[1] - oB[1]) + (oA[2] - oB[2]) * (
                              oA[2] - oB[2]))

          self.cells[m].set_force([0.0, 0.0, 0.0])
          self.cells[m + 1].set_force([0.0, 0.0, 0.0])

          # 5 steps free running
          stopCycle = (counter + 5) * steps_in_one_cycle

          while cycle < stopCycle:

              self.system.integrator.run(steps=steps_in_one_cycle)
              counter += 1
              cycle = counter * steps_in_one_cycle

class BiCluster(OifCluster):
    def __init__(self, cell_type, radius = 4, cluster_centroid = [0.0, 0.0, 0.0], starting_id = 0):
        self.cell_type = cell_type
        self.radius = radius
        self.cell_positions = []
        super(BiCluster, self).__init__([]);
        self.createPositions(radius, cluster_centroid)
        self.createCells(starting_id)
        
    def createPositions(self, radius, cluster_centroid, space=1):
        self.cell_positions = []
        
        [x, y, z] = cluster_centroid
        
        self.cell_positions.append([x - (radius + (space / 2.0)), y, z])
        self.cell_positions.append([x + (radius + (space / 2.0)), y, z])
        
        
    def createCells(self, starting_id=0):
        for i in range(len(self.cell_positions)):
          self.add_cell(OifCell(cell_type=self.cell_type,
                                  particle_type=starting_id + i,
                                  origin=self.cell_positions[i],
                                  particle_mass=0.5,
                                  rotate=None,
                                  exclusion_neighbours=False))
                                  

class NCluster(OifCluster):
    def __init__(self, cell_type, cell_count, radius = 4, cluster_centroid = [0.0, 0.0, 0.0], starting_id = 0):
        self.cell_type = cell_type
        self.radius = radius
        self.cell_positions = []
        self.cell_count = cell_count
        super(NCluster, self).__init__([]);
        self.createPositions(radius, cluster_centroid)
        self.createCells()
             
    def createPositions(self, radius, cluster_centroid, space=1):
        self.cell_positions = []
        
        [x, y, z] = cluster_centroid
        
        if(self.cell_count == 2):
            self.cell_positions.append([x - (radius + (space / 2.0)), y, z])
            self.cell_positions.append([x + (radius + (space / 2.0)), y, z])
            
        elif(self.cell_count == 3):
            
            side_length = 2 * radius + space
            
            median_length = ((radius * 2 + space) ** 2.0 - (radius + space / 2.0) ** 2.0) ** (1 / 2)
            median_distance_from_base = median_length * (1.0 / 3.0)
            median_distance_from_top = median_length * (2.0 / 3.0)
            
            self.cell_positions.append([x - median_distance_from_base, y + (radius + space / 2.0), z])
            self.cell_positions.append([x - median_distance_from_base, y - (radius + space / 2.0), z])
            self.cell_positions.append([x + median_distance_from_top, y, z])
            
        elif(self.cell_count == 4):
            median_length = ((radius * 2 + space) ** 2.0 - (radius + space / 2.0) ** 2.0) ** (1 / 2)
            median_distance_from_base = median_length * (1.0 / 3.0)
            median_distance_from_top = median_length * (2.0 / 3.0)
            median_length_from_center = ((radius * 2.0 + space) ** 2.0 - (median_distance_from_top) ** 2.0) ** (1 / 2)

            self.cell_positions.append([x - median_distance_from_base, y + (radius + space / 2.0), z])
            self.cell_positions.append([x - median_distance_from_base, y - (radius + space / 2.0), z])
            self.cell_positions.append([x + median_distance_from_top, y, z])
            self.cell_positions.append([x, y, z + median_length_from_center])
            
        elif(self.cell_count == 5):
            median_length = ((radius * 2 + space) ** 2.0 - (radius + space / 2.0) ** 2.0) ** (1 / 2)
            median_distance_from_base = median_length * (1.0 / 3.0)
            median_distance_from_top = median_length * (2.0 / 3.0)
            median_length_from_center = ((radius * 2.0 + space) ** 2.0 - (median_distance_from_top) ** 2.0) ** (1 / 2)

            self.cell_positions.append([x - median_distance_from_base, y + (radius + space / 2.0), z])
            self.cell_positions.append([x - median_distance_from_base, y - (radius + space / 2.0), z])
            self.cell_positions.append([x + median_distance_from_top, y, z])
            self.cell_positions.append([x, y, z + median_length_from_center])
            self.cell_positions.append([x, y, z - median_length_from_center])
        
    def createCells(self):
        for i in range(len(self.cell_positions)):
            # ak pridane do oif_classes -> nieje potrebne oif.
            self.add_cell(OifCell(cell_type=self.cell_type,
                                  particle_type=i,
                                  origin=self.cell_positions[i],
                                  particle_mass=0.5,
                                  rotate=None,
                                  exclusion_neighbours=False))  
                                  

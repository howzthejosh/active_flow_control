"""
Mesh generation script using GMSH Python API
Author: Joshua Mathew Jacob

TODO: Turn this into a function that can be called in another program
"""

import gmsh
import sys 
import math
from math import *

gmsh.initialize()

# Mesh Parameters
# TODO Read from either configuration file or from Command line arguments using Parser

jet_width = 10*pi/180
radius = 0.5
jet_positions = [pi/2, 3*pi/2]
# jet_positions = []

length = 22
width = 4.1
front_distance = 2
bottom_distance = 2

cylinder_size = 0.1
box_size = 0.2



cylinder = []       # List to hold tags of all lines on cylinder - both jet and no slip surfaces
lower_bound = []    # List to hold tags of lower bound points of jets
upper_bound = []    # List to hold tags of upper bound points of jets


# Cylinder - Jets

# Define cylinder center
cylinder_center = gmsh.model.geo.add_point(0,0,0)
cylinder_tag = 4

n = len(jet_positions)

if n>0: 
    for i in range(n):

        angle = jet_positions[i]
    
        x = radius*cos(angle-jet_width/2)
        y = radius*sin(angle-jet_width/2)
        p = gmsh.model.geo.add_point(x, y, 0, cylinder_size)
        lower_bound.append(p)
    
        x0 = radius*cos(angle)
        y0 = radius*sin(angle)
        arch_centre = gmsh.model.geo.add_point(x0, y0, 0, cylinder_size)

        x = radius*cos(angle+jet_width/2)
        y = radius*sin(angle+jet_width/2)
        q = gmsh.model.geo.add_point(x, y, 0, cylinder_size)
        upper_bound.append(q)


        jet_tag = 5+i
        jet_arcs = []


        jet_arc1 = gmsh.model.geo.add_circle_arc(p, cylinder_center, arch_centre)
        jet_arcs.append(jet_arc1)
        cylinder.append(jet_arc1)
        jet_arc2 = gmsh.model.geo.add_circle_arc(arch_centre, cylinder_center, q)
        jet_arcs.append(jet_arc2)
        cylinder.append(jet_arc2)

        gmsh.model.add_physical_group(1,jet_arcs,jet_tag)




    # Cylinder - No Slip Surfaces 
    # TODO make it general for the case of one jet where one arc may exceed pi (From Rabault, Why?)
    
    
    cylinder_arcs = []
    lower_bound.append(lower_bound[0])

    # Synchronize needed before getValue
    gmsh.model.geo.synchronize()

    for i in range(n):

        p = upper_bound[i]
        q = lower_bound[i+1]

        cylinderarc = gmsh.model.geo.add_circle_arc(p, cylinder_center, q)
        cylinder_arcs.append(cylinderarc)
        cylinder.append(cylinderarc)

    gmsh.model.add_physical_group(1,cylinder_arcs,cylinder_tag)

else:
    c1 = gmsh.model.geo.add_point(-radius, 0, 0, cylinder_size)
    c2 = gmsh.model.geo.add_point(0, radius, 0, cylinder_size)
    c3 = gmsh.model.geo.add_point(radius, 0, 0, cylinder_size)
    c4 = gmsh.model.geo.add_point(0, -radius, 0, cylinder_size)

    cl1 = gmsh.model.geo.add_circle_arc(c1, cylinder_center, c2)
    cl2 = gmsh.model.geo.add_circle_arc(c2, cylinder_center, c3)
    cl3 = gmsh.model.geo.add_circle_arc(c3, cylinder_center, c4)
    cl4 = gmsh.model.geo.add_circle_arc(c4, cylinder_center, c1)

    cylinder = [cl1,cl2,cl3,cl4]
    gmsh.model.add_physical_group(1,cylinder, cylinder_tag)





# Channel 

p1 = gmsh.model.geo.add_point(-front_distance, -bottom_distance, 0, box_size)
p2 = gmsh.model.geo.add_point(-front_distance, -bottom_distance+width, 0, box_size)
p3 = gmsh.model.geo.add_point(-front_distance+length, -bottom_distance+width, 0, box_size)
p4 = gmsh.model.geo.add_point(-front_distance+length, -bottom_distance, 0, box_size)


# Inlet
inlet_tag = 1
l1 = gmsh.model.geo.add_line(p1,p2)
gmsh.model.geo.synchronize()
gmsh.model.add_physical_group(1, [l1], inlet_tag)

# Outlet
outlet_tag = 2
l3 = gmsh.model.geo.add_line(p3,p4)
gmsh.model.geo.synchronize()
gmsh.model.add_physical_group(1, [l3], outlet_tag)

# Walls
wall_tag = 3
l2 = gmsh.model.geo.add_line(p2,p3)
l4 = gmsh.model.geo.add_line(p4,p1)
gmsh.model.geo.synchronize()
gmsh.model.add_physical_group(1, [l2,l4], wall_tag)


# Curve Loops 
box = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4])
cylinder = gmsh.model.geo.add_curve_loop(cylinder)

gmsh.model.geo.add_plane_surface([box,cylinder],1)
gmsh.model.add_physical_group(2,[1],1, name="domain")



# Synchronize CAD kernel
gmsh.model.geo.synchronize()

# Generate mesh
gmsh.model.mesh.generate(2)

# Write mesh to file
# TODO Generalize with inputted file name...
gmsh.write("cylinder.msh")



# Displays mesh using gmsh gui unless -nopopup is passed
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()


gmsh.finalize()

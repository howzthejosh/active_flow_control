# Author:   Joshua Mathew Jacob
# Date:     28/06/2023
#
#
# Program to convert from the gmsh format '.msh' to the FEniCS 
# compatible format either '.xml' or '.xdmf' using the functionality
# of meshio package
# 
# Meshio allows us to access the data inside '.msh' file in the form
# of points, cells, cell_data, field_data, etc...that hold information
# about the mesh
#
# The main aim of this mesh conversion is to create a separate mesh for 
# the surfaces/triangles and the lines/edges. This is because when we read the 
# mesh into FEniCS we want to collect these values (usually markers/id) 
# using MeshValueCollection. We then use MeshFunction to read the markers from
# the MVC
#
#
# The following code takes inspiration from JSDokken on FEniCS Project discourse
# group as well as Abhinav Gupta from CMLabIITR 
 


import meshio
import numpy

# Read the mesh file
mesh_from_file = meshio.read("cylinder.msh")

def create_mesh(mesh, cell_type, name_to_read, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={name_to_read:[cell_data]})
    return out_mesh

# Create mesh file with line elements
line_mesh = create_mesh(mesh_from_file, "line", "line_markers", prune_z=True)
meshio.write("facet_mesh.xdmf", line_mesh)

# Create mesh file with triangle elements
triangle_mesh = create_mesh(mesh_from_file, "triangle", "surf_markers", prune_z=True)
meshio.write("mesh.xdmf", triangle_mesh)
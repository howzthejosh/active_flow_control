"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for flow around a cylinder using the Incremental Pressure Correction
Scheme (IPCS).

    u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function
from dolfin import *
import numpy as np

T = 1.0            # final time
num_steps = 100    # number of time steps
dt = T / num_steps # time step size
mu = 0.01        # dynamic viscosity
rho = 1            # density
nu = mu/rho

# Create mesh
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)

mvc = MeshValueCollection("size_t", mesh, 1) 

with XDMFFile("facet_mesh.xdmf") as infile:
   infile.read(mvc, "name_to_read")

boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)



# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define circle boundary for Drag, Lift measurement
ds_circle = Measure("ds",subdomain_data=boundaries, subdomain_id=4)


# Define inflow profile
inflow_profile = ('1.5*(1 - pow(x[1]/2.05, 2))','0')


inlet_tag = 1
outlet_tag = 2
wall_tag = 3
cylinder_noslip = 4
jet1_tag = 5
jet2_tag = 6


# Define boundary conditions
bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), boundaries, inlet_tag)
bcu_walls = DirichletBC(V, Constant((0, 0)), boundaries, wall_tag)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), boundaries, cylinder_noslip)
bcp_outflow = DirichletBC(Q, Constant(0), boundaries, outlet_tag)

radius = 0.5
alpha = (10*np.pi/180)/2
half_width =  radius*np.sin(alpha)
u_max = 0.5
my_jet_bc = Expression(('0', 'u_max*(1 - pow(x[0]/half_width, 2))'), u_max=u_max, half_width=half_width, degree=2)

bcu_jet1 = DirichletBC(V, my_jet_bc, boundaries, jet1_tag)
bcu_jet2 = DirichletBC(V, my_jet_bc, boundaries, jet2_tag)

# bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcu = [bcu_inflow, bcu_walls, bcu_cylinder, bcu_jet1, bcu_jet2]
bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('navier_stokes_cylinder/velocity.xdmf')
xdmffile_p = XDMFFile('navier_stokes_cylinder/pressure.xdmf')

# Create time series (for use in reaction_system.py)
#timeseries_u = TimeSeries('navier_stokes_cylinder/velocity_series')
#timeseries_p = TimeSeries('navier_stokes_cylinder/pressure_series')

# Save mesh to file (for use in reaction_system.py)
File('navier_stokes_cylinder/cylinder.xml.gz') << mesh

# Create progress bar
progress = Progress('Time-stepping',num_steps)
set_log_level(LogLevel.PROGRESS)


# Time-stepping
t = 0
for n in range(num_steps):
    # Update current time
    t += dt

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Plot solution
    #plot(u_, title='Velocity')
    #plot(p_, title='Pressure')

    # Save solution to file (XDMF/HDF5)
    # if n > 1000 and n%10 == 0:
    #     xdmffile_u.write(u_, t)
    #     xdmffile_p.write(p_, t)

    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)


    # Save nodal values to file
    #timeseries_u.store(u_.vector(), t)
    #timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

    # Compute Drag and Lift
    U_mean = 1.0
    L = 1.0
    n = FacetNormal(mesh)
    I = Identity(u_.geometric_dimension())
    force = dot(-p_*I + 2.0*nu*sym(grad(u_)), n)
    F_D = assemble(-force[0]*ds_circle)
    C_D = 2/(U_mean**2*L)*F_D
    print("Drag Force: %f",F_D)
    print("Cd: %f",C_D)


    # Update progress bar
    progress+=1
    print('u max:', np.array(u_.vector()).max())


# import matplotlib.pyplot as plt
# plot(u_)
# plt.show()
 
 

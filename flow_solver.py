"""
Incompressible Navier-Stokes equations
for flow around a cylinder using 
the Incremental Pressure Correction Scheme (IPCS).

    u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

import tqdm.autonotebook
import csv 
import os
import argparse
# import pandas as pd

from dolfin import *
import numpy as np
from mpi4py import MPI


def flow_solver(A, ki, Re, sim_time, num_steps, ave_start_iter):

    mesh_comm = MPI.COMM_WORLD
    rank = mesh_comm.rank
    model_rank = 0

    t = 0.0
    T = sim_time             # final time
    dt = T / num_steps       # time step size
    mu = 1.0/Re              # dynamic viscosity
    rho = 1.0                # density
    nu = mu/rho

    # Import mesh
    mesh = Mesh(mesh_comm)
    with XDMFFile(mesh_comm,"mesh/mesh.xdmf") as infile:
        infile.read(mesh)

    mvc = MeshValueCollection("size_t", mesh, 1) 

    with XDMFFile(mesh_comm,"mesh/facet_mesh.xdmf") as infile:
        infile.read(mvc, "line_markers")

    boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc)


    ##-----Variational Problem Setup-----##

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)


    # Define inflow profile
    inflow_profile = ('1.5*(1 - pow(x[1]/2.05, 2))','0')

    # Hardcode facet tags
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
    A = A
    ki = ki

    my_jet_bc = Expression(('0', '(1.0 - pow(x[0]/half_width, 2.0))*(A*cos(2*pi*ki*t))'), 
                            A=A,
                            ki=ki,
                            half_width=half_width, 
                            t=t,
                            degree=2)

    bcu_jet1 = DirichletBC(V, my_jet_bc, boundaries, jet1_tag)
    bcu_jet2 = DirichletBC(V, my_jet_bc, boundaries, jet2_tag)


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

    # Form definition for drag and lift
    # Define circle boundary for Drag, Lift measurement
    ds_circle = Measure("ds",subdomain_data=boundaries, subdomain_id=4)
    U_mean = 1.0
    L = 1.0
    ncap = FacetNormal(mesh)
    I = Identity(u_.geometric_dimension())
    force = dot(-p_*I + 2.0*nu*sym(grad(u_)), ncap)

    # Save mesh to file 
    f = HDF5File(mesh_comm,'output/mesh.h5','w')
    f.write(mesh,'mesh') 

    # Create XDMF files for visualization output
    xdmffile_u = XDMFFile('output/velocity.xdmf')
    xdmffile_p = XDMFFile('output/pressure.xdmf')

    # Rename variable name for output
    u_.rename('velocity','0')
    p_.rename('pressure','0')

    # Storing daag and lift values in csv file
       
    arr = [[] for i in range(num_steps)]

    if (rank == model_rank): 
        csvfilename = "output_controlled_forces.csv"
        if os.path.exists(csvfilename):
            if os.stat(csvfilename)!=0:
                open(csvfilename, 'w').close()

        headerList = ['t','CD','CL']
        with open(csvfilename,"w") as file:
            dw = csv.DictWriter(file,delimiter=',',fieldnames=headerList)
            dw.writeheader()


    # Progress bar
    if rank == model_rank:
        progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)

    # Time-stepping
    t = 0.0
    for n in range(num_steps):

        # Update progress bar
        if rank == model_rank:
            progress.update(1)

        # Update current time
        t += dt

        # Initializing jet boundary condition 
        my_jet_bc.t = t
        # if n>10000:
        #     my_jet_bc.t = t
        # else:
        #     my_jet_bc.t = 0.5

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
        
        
        # Save solution to file (XDMF/HDF5)
        if (n>15000 and n%100==0):
            xdmffile_u.write(u_, t)
            xdmffile_p.write(p_, t)

        
        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)

        # Compute Drag and Lift
        F_D = assemble(-force[0]*ds_circle)
        F_L = assemble(-force[1]*ds_circle)
        C_D = 2/(U_mean**2*L)*F_D
        C_L = 2/(U_mean**2*L)*F_L
        
        
        
        arr[n].append(t)
        arr[n].append(C_D)
        arr[n].append(C_L)

        if (rank==model_rank):
            if ((n+1)%1000==0):
                with open(csvfilename,"a") as file:
                    writer = csv.writer(file)
                    writer.writerows(arr[(n+1)-1000:(n+1)])


    
    # Return CD mean using nested list loop from arr
    
    CD_list = []
    for i1 in range(len(arr)):
        CD_list.append(arr[i1][1])

    # Start averaging only from ave_start_iter
    CD = np.array(CD_list[ave_start_iter:])
    CD_mean = np.mean(CD)

    if rank==model_rank:
        print(CD_mean,rank)
    return CD_mean
    
            

    #----------------------------------------------------




if __name__ =="__main__":
    
    
    # Parsing parameters from command line
    parser = argparse.ArgumentParser(description="Flow Across Cylinder with Actuation")
    parser.add_argument("--A", help="Signal amplitude value", type=float, default=1.25)
    parser.add_argument("--ki", help="Signal frequency value", type=float, default=1.0)
    parser.add_argument("--Re", help="Reynolds number", type=int, default=100)
    parser.add_argument("--sim_time", help="Simulation time", type=float, default=1.0)
    parser.add_argument("--num_steps", help="Number of time steps", type=int, default=500)
    parser.add_argument("--ave_start_iter", help="Number of iter to skip before starting averaging", type=int, default=15000)

    # Create args object
    args = parser.parse_args()

    A = args.A
    ki = args.ki
    Re = args.Re 
    sim_time = args.sim_time
    num_steps = args.num_steps
    ave_start_iter = args.ave_start_iter


    a = flow_solver(A=A,
                ki=ki,
                Re=Re,
                sim_time=sim_time,
                num_steps=num_steps,
                ave_start_iter=ave_start_iter)
  
    print(a)
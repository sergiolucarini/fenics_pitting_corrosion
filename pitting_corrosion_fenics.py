#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:56:03 2021
Pitting corrosion in FEniCs
@author: Sergio Lucarini
"""

import fenics as fn

# Length and discretization of mesh
L = [200e-6,140e-6]; n=[200,140]
# Create mesh rectangle L0,L1 with n0,n1 nodes
mesh = fn.RectangleMesh(fn.Point(0,0),fn.Point(L[0],L[1]),n[0],n[1],"crossed")

# Define Space: Lagrange 1-linear 2-quadratic elements
Fun = fn.FiniteElement('CG', mesh.ufl_cell(), 1)
pcFun = fn.FunctionSpace(mesh, Fun * Fun)

# solution variables, funtion and test functions
pc_sol = fn.Function(pcFun)
p_sol, c_sol = fn.split(pc_sol)
tpc = fn.TrialFunction(pcFun)
tc, tp = fn.split(tpc)
dpc = fn.TestFunction(pcFun)
dc, dp = fn.split(dpc)
# auxiliary variables for saving previous solutions
pc_t = fn.Function(pcFun)
p_t, c_t = fn.split(pc_t)

# Introduce manually the material parameters
alphap=4.8e-5
omegap=3.33e7
DD=8.5e-10
AA=5.35e7
Lp=2e0
cse=1.
cle=5100/1.43e5

# Boundary conditions
def ConcentrationBoundary(x):
    return fn.near(x[1],140.0e-6) and x[0] <= 100.0e-6+5e-6 and x[0] >= 100.0e-6-5e-6
bc_pc = [fn.DirichletBC(pcFun.sub(0), fn.Constant(0.0), ConcentrationBoundary),\
          fn.DirichletBC(pcFun.sub(1), fn.Constant(0.0), ConcentrationBoundary)]

# Initial conditions as expression
class InitialConditions(fn.UserExpression):
    def eval_cell(self, values, x, ufc_cell):
        if (x[0]-200.0e-6/2.)**2+(x[1]-140.0e-6)**2<5e-6**2:
            values[0] = 0.0
            values[1] = 0.0
        else:
            values[0] = 1.0
            values[1] = 1.0
    def value_shape(self):
        return (2,)
# Apply initial conditions        
pc_sol.interpolate(InitialConditions())
pc_t.interpolate(InitialConditions())
    
# Weak Form # Constituive functions
dx = fn.dx() # define the volume integral
DT = fn.Expression("dtime", dtime=0, degree=0) # difine time incrementation
E_pc = (c_sol-c_t)/DT*dc*dx+\
        -fn.inner(-DD*fn.grad(c_sol)+DD*(cse-cle)*fn.grad(-2*p_sol**3 + 3*p_sol**2),fn.grad(dc))*dx+\
        (p_sol-p_t)/DT/Lp*dp*dx+\
        -2*AA*(c_sol-(-2*p_sol**3 + 3*p_sol**2)*(cse-cle)-cle)*(cse-cle)*(-6*p_sol**2 + 6*p_sol)*dp*dx+\
        omegap*(4*p_sol**3-6*p_sol**2+2*p_sol)*dp*dx+\
        fn.inner(alphap*fn.grad(p_sol),fn.grad(dp))*dx

# Automatically calculate jacobian
Jpc = fn.derivative(E_pc, pc_sol, tpc)
# Define the non linear problem and solver parameters
p_pc = fn.NonlinearVariationalProblem(E_pc, pc_sol, bc_pc, J=Jpc)
solver_pc = fn.NonlinearVariationalSolver(p_pc)
solver_pc.parameters['newton_solver']['absolute_tolerance'] = 1E-8
solver_pc.parameters['newton_solver']['linear_solver'] = 'mumps'
solver_pc.parameters['newton_solver']["convergence_criterion"] = "incremental"
solver_pc.parameters['newton_solver']["relative_tolerance"] = 1e-6
solver_pc.parameters['newton_solver']["maximum_iterations"] = 10

# Initialization of the time incremental procedure and output requests
t=0;DeltaT=1e-3
DT.dtime=1.*DeltaT
conc_f = fn.File ("./results/fields.pvd")
conc_f << pc_sol

# Time incremental loop
while t<=400:
    # Update time
    t += DeltaT
    DT.dtime=DeltaT
    
    # Newton solver with updated time increment
    try:
        info=solver_pc.solve()
    # If not solved, reduce time increment and start again or stop
    except:
        t -= DeltaT;DeltaT/=2.;print('red dt',DeltaT)
        if DeltaT<1e-4: break
        continue
         
    #save ouput
    print ('Iterations:', info[0], ', Total time:', t)
    conc_f << pc_sol
    
    # save state in t+dt for next increment
    pc_t.vector()[:]=pc_sol.vector()
    
    # increase if good convergence
    if info[0]<6: DeltaT*=2.;print('inc dt',DeltaT);

print('Simu complete') 

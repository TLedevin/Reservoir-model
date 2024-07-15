import pandas as pd
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import math
import os

from colorama import Fore, Back, Style

from tqdm import tqdm
from mshr import*

from scipy.interpolate import griddata
import pickle as pkl
from Auxiliary_class import*

########################################################################################################################
# Extract lists of points, pressure values and gas thickness values for a given iteration
def extract_points_p_h(folder, iteration):
    reader_p = vtk.vtkXMLUnstructuredGridReader()
    reader_p.SetFileName(file_name_p(folder, iteration))
    reader_p.Update()  # Needed because of GetScalarRange

    points = vtk_to_numpy(reader_p.GetOutput().GetPoints().GetData()).tolist()
    pressure_int = (1e-5*vtk_to_numpy(reader_p.GetOutput().GetPointData().GetArray(0))).tolist()
    reader_h = vtk.vtkXMLUnstructuredGridReader()
    reader_h.SetFileName(file_name_h(folder, iteration))
    reader_h.Update()  # Needed because of GetScalarRange
    h = vtk_to_numpy(reader_h.GetOutput().GetPointData().GetArray(0)).tolist()
    return(points, pressure_int, h)

########################################################################################################################
# Search name of the file that values of pressures for a given iteration
def file_name_p(folder, iteration):
    if iteration == 0 : return(os.path.join(folder, "p/output_p000000.vtu"))
    digits = int(math.log10(iteration))+1
    car = ""
    for i in range(6-digits): car += "0"
    return(os.path.join(folder, "p/output_p" + car + str(iteration)+ ".vtu"))

########################################################################################################################
# Search name of the file that values of pressures for a given iteration
def file_name_h(folder, iteration):
    if iteration == 0 : return(os.path.join(folder, "h/output_h000000.vtu"))
    digits = int(math.log10(iteration))+1
    car = ""
    for i in range(6-digits): car += "0"
    return(os.path.join(folder, "h/output_h" + car + str(iteration)+ ".vtu"))

########################################################################################################################
# Convert time t in seconds to calendar format (min, hour, day, year)

def day_format(t):
    min = 60 ; hour = 60*min ; day = 24*hour ; year = 365*day
    if t<min : return(str(int(t)) + ' s')
    elif t<hour : return(str(int(t/min)) + ' min')
    elif t<day : return(str(int(10*t/hour)/10) + ' h')
    elif t<year : return(str(int(10*t/day)/10) + ' days')
    else : return(str(int(100*t/year)/100) + ' years')

########################################################################################################################
# Create the meshgrid of the simulation
# domain, Nm, Nhr, Nst, Sources_g, sigma, T_tot, phi, sg, rho_g0, H must be provided
def create_mesh(domain, Nm, Nhr, Nst, Sources_g, sigma, T_tot, phi, sg, rho_g0, H):
    mesh = generate_mesh(domain, Nm)
    for i in range(Nhr):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            cell_markers[c] = False
            for source in Sources_g :
                p = source[0]
                rate = np.nanmean(source[1])
                Hydraulic_radius = np.sqrt(np.abs(rate)*T_tot/(phi*sg*rho_g0*H))
                if c.midpoint().distance(p) < 2*Hydraulic_radius: cell_markers[c] = True
        mesh = refine(mesh, cell_markers)
    for i in range(Nst):
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            cell_markers[c] = False
            for source in Sources_g :
                p = source[0]
                if c.midpoint().distance(p) < 5*sigma: cell_markers[c] = True
        mesh = refine(mesh, cell_markers)
    return(mesh)

########################################################################################################################
# Increase the number of points of the mesh in an area composed by circles centered on some points and with a given radius
# Nst gives the number of times we reprecise the meshgrid
def precise_mesh(mesh, points_wells, N_wells, radius):
    N_wells_tamp = N_wells
    while N_wells_tamp > 0:
        cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
        for c in cells(mesh):
            cell_markers[c] = False
            if N_wells_tamp>0:
                for point in points_wells :
                    if c.midpoint().distance(point) < radius: cell_markers[c] = True
        if N_wells_tamp>0: N_wells_tamp -= 1
        mesh = refine(mesh, cell_markers)
    return(mesh)

########################################################################################################################
# Define the true physical value for the thickness of the gas (if h<0 : h=0, if h>H : h=H, else : keep value of h)
# H must be provided

def true_h(H, h): return((H+(h+abs(h))/2 - abs(H-(h+abs(h))/2))/2)

########################################################################################################################
# Calculate extrem values for h and p during the entire simulations
# Nt, H, g, sg, Nz, rho_g, rho_w, List_t must be provided

def min_max_pressure_h_total(folder, Nt, H, g, sg, Nz, rho_g, rho_w, List_t):
    p_min, p_max, h_min, h_max = 1e10,0,1e10,0
    for i in tqdm(range(Nt)):
        points_init, pressure, h = extract_points_p_h(folder, i)
        p_min_t, p_max_t, h_min_t, h_max_t = np.min(pressure), np.max(pressure), np.min(h), np.max(h)
        if p_min_t < p_min : p_min = p_min_t
        if p_max_t > p_max : p_max = p_max_t
        if h_min_t < h_min : h_min = h_min_t
        if h_max_t > h_max : h_max = h_max_t
    return(p_min, p_max, h_min, h_max)

########################################################################################################################
# Determine if we inject or not at time t for a given injection profile (which corresponds to a source)

def bin_inject(t, Inj_profile):
    if len(Inj_profile) == 0 : return 0
    elif t>=Inj_profile[0] and t<=Inj_profile[1] : return 1
    return 0

########################################################################################################################
# Returns the true value of gas thickness : 0 if h<0 ; h if 0<h<H ; H if H<h.
def true_h(H, h): return((H+(h+abs(h))/2 - abs(H-(h+abs(h))/2))/2)
def true_p(p): return((1000000+p+abs(1000000-p))/2)

########################################################################################################################
# Define source function at time t from the list with sources data (Sources_g)
# ME1, sigma must be provided
def bin_inject(t, Inj_profile):
    if len(Inj_profile) == 0 : return 0
    elif t>=Inj_profile[0] and t<=Inj_profile[1] : return 1
    return 0

def s(Sources_g, t, ME1, sigma):
    alpha = 1/sigma**2 ; K = 1/(np.pi*sigma**2)
    f_g = '0'
    s_g_ = Function(ME1)
    for source in Sources_g :
        x = source[0][0] ; y = source[0][1]
        for i in range(len(source[1])):
            rate = source[1][i] ; Inj_profile = source[2][i]
            bin = bin_inject(t, Inj_profile)
            if bin:
                f_g += '+' + str(rate) + '*K*exp(-alpha*((x[0]-'+str(x)+')*(x[0]-'+str(x)+')+(x[1]-'+str(y)+')*(x[1]-'+str(y)+')))'
    s_g = Expression(f_g, K=1/(np.pi*sigma**2), alpha=1/sigma**2, degree=2)
    return(s_g)

########################################################################################################################
# Define problem solved by the solver at a time t
# dt Sources_g, ME1, H, phi, sg, g, rho_g, rho_w, rho_g0, rho_w0, mug, muw, cr, cg, cw, u, p, h, p0, h0, du, bcs, sigma, v_p, v_h, k, krg,
# krw must be provided

def define_problem(dt, t, Sources_g, ME1, H, phi, sg, g, rho_g, rho_w, rho_g0, rho_w0, mug, muw, cr, cg, cw, u, p, h, p0, h0, du, sigma, v_p, v_h, k, krg, krw, bcs):
    s_g_ = Function(ME1)
    s_g_.interpolate(s(Sources_g, t, ME1, sigma))

    L_p = phi*(rho_g(p)*(1-sg)*(h-true_h(H, h0))+((1-sg)*h+sg*H)*(cr*rho_g(p)+cg*rho_g0)*(p-true_p(p0)))*v_p*dx \
    + dt*rho_g(p)*k*krg/mug(p)*h*inner(nabla_grad(p) - rho_g0*g*nabla_grad(h), nabla_grad(v_p))*dx \
    - dt*s_g_*v_p*dx

    L_h = phi*(-rho_w(p)*(h-true_h(H, h0))+(cr*rho_w(p)+cw*rho_w0)*(p-true_p(p0)))*v_h*dx \
    + dt*rho_w(p)*k*krw/muw(p)*(H-h)*inner(nabla_grad(p) - rho_w0*g*nabla_grad(h), nabla_grad(v_h))*dx \
    
    L = L_h + L_p
    
    J = derivative(L, u, du) # Compute directional derivative about u in the direction of du (Jacobian)
    
    return(VFE_MP_model(J, L, bcs))

def define_problem_topography(dt, t, Sources_g, ME1, H, phi, sg, g, rho_g, rho_w, rho_g0, rho_w0, mug, muw, cr, cg, cw, u, p, h, p0, h0, du, sigma, v_p, v_h, k, krg, krw, bcs, depth, angle_rad):
    s_g_ = Function(ME1)
    s_g_.interpolate(s(Sources_g, t, ME1, sigma))
    
    L_p = phi*((cr*rho_g(p)+cg*rho_g0)*sg*h*(p-p0) + rho_g(p)*sg*(h-true_h(H, h0)))*v_p*dx \
    + dt*rho_g(p)*k*krg/mug(p)*h*inner(nabla_grad(p)  - rho_g0*g*nabla_grad(cos(angle_rad)*h) -rho_g0*g*nabla_grad(depth), nabla_grad(v_p))*dx \
    - dt*s_g_*v_p*dx  

    L_h = phi*((cr*rho_w(p) + rho_w0*cw)*(H-sg*h)*(p-p0) - rho_w(p)*sg*(h-true_h(H, h0)))*v_h*dx \
    + dt*rho_w(p)*k*krw/muw(p)*(H-h)*inner(nabla_grad(p) + rho_w0*g*nabla_grad(cos(angle_rad)*(H-h))-rho_w0*g*nabla_grad(depth+H),nabla_grad(v_h))*dx \

    L = L_h + L_p
    J = derivative(L, u, du) # Compute directional derivative about u in the direction of du (Jacobian)
    
    return(VFE_MP_model(J, L, bcs))

########################################################################################################################
# Calculating theoritical injected (or extracted) mass. Positive for injected, negative for extracted

def injected_mass(t, Sources_g):
    amount = 0
    for source in Sources_g:
        New_inj_profile = []
        if len(source[2]) != 0:
            for i in range(len(source[2])):
                Inj_profile = source[2][i] ; rate = source[1][i]
                if Inj_profile[1]<= t: amount += rate*(Inj_profile[1]-Inj_profile[0])
                elif Inj_profile[0]<= t: amount += rate*(t-Inj_profile[0])
    return(amount)

########################################################################################################################
# Create functions for permeability, porosity, thickness and depth

def H_function(H_points, H_field, ME1):
    H = Function(ME1)
    H_data = griddata(H_points, H_field, ME1.tabulate_dof_coordinates(), method='nearest')
    H_tamp = np.nanmean(H_data)
    for l in range(ME1.dim()):
        if np.isnan(H_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            H.vector()[l] = H_tamp
            H_tamp = H.vector()[l]
        else:
            H.vector()[l] = H_data[l]
            H_tamp = H_data[l]
    return(H)

def k_function(k_points, k_field, ME1):
    k = Function(ME1)
    k_data = griddata(k_points, k_field, ME1.tabulate_dof_coordinates(), method='linear')
    k_tamp = np.nanmean(k_data)
    for l in range(ME1.dim()):
        if np.isnan(k_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            k.vector()[l] = k_tamp
        else:
            k.vector()[l] = k_data[l]
            k_tamp = k_data[l]
    return(k)

def phi_function(phi_points, phi_field, ME1):
    phi = Function(ME1)
    phi_data = griddata(phi_points, phi_field, ME1.tabulate_dof_coordinates(), method='linear')
    phi_tamp = np.nanmean(phi_data)
    for l in range(ME1.dim()):
        if np.isnan(phi_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            phi.vector()[l] = phi_tamp
        else:
            phi.vector()[l] = phi_data[l]
            phi_tamp = phi_data[l]
    return(phi)

def depth_function(depth_points, depth_field, ME1):
    depth = Function(ME1)
    depth_data = griddata(depth_points, depth_field, ME1.tabulate_dof_coordinates(), method='linear')
    depth_tamp = np.nanmean(depth_data)
    for l in range(ME1.dim()):
        if np.isnan(depth_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            depth.vector()[l] = depth_tamp
        else:
            depth.vector()[l] = depth_data[l]
            depth_tamp = depth_data[l]
    return(depth)

def angle_function(angle_points, angle_field, ME1):
    angle = Function(ME1)
    angle_data = griddata(angle_points, angle_field, ME1.tabulate_dof_coordinates(), method='linear')
    angle_tamp = np.nanmean(angle_data)
    for l in range(ME1.dim()):
        if np.isnan(angle_data[l]): # if there is a NaN value, we give the precedent value to the vector.
            angle.vector()[l] = angle_tamp
        else:
            angle.vector()[l] = angle_data[l]
            angle_tamp = angle_data[l]
    return(angle)

########################################################################################################################
# Search the minimum and the maximum values of pressure and gas thickness for a list of iterations N_plot
def min_h_p(folder, N_plot):
    data_plot = extract_points_p_h(folder, N_plot[0])
    p_min_int, p_max_int = np.min(data_plot[1]), np.max(data_plot[1])
    h_min_int, h_max_int = np.min(data_plot[2]), np.max(data_plot[2])

    for j in range(1, len(N_plot)):
        data_plot = extract_points_p_h(folder, N_plot[j])
        min_p_tamp, max_p_tamp = np.min(data_plot[1]), np.max(data_plot[1])
        if min_p_tamp < p_min_int : p_min_int = min_p_tamp
        if max_p_tamp > p_max_int : p_max_int = max_p_tamp
        min_h_tamp, max_h_tamp = np.min(data_plot[2]), np.max(data_plot[2])
        if min_h_tamp < h_min_int : h_min_int = min_h_tamp
        if max_h_tamp > h_max_int : h_max_int = max_h_tamp
    return(p_min_int, p_max_int, h_min_int, h_max_int)

########################################################################################################################
# Export list of pressure of an iteration
# Nx, Ny, Nz, List_t, AOI, H, H0, rho_g, rho_w, g, Outline must be provided

def export_p(iteration, Nx, Ny, Nz, List_t, AOI, Outline, phi0, k0):
    points_init, pressure_int, h = extract_points_p_h(iteration)
    points = [] ; P = []

    for i in tqdm(range(len(points_init))):
        points.append([points_init[i][0], points_init[i][1]])

    X = np.linspace(Outline[0][0], Outline[1][0], int(Nx)) ; Y = np.linspace(Outline[0][1], Outline[2][1], int(Ny))
    grid_x, grid_y = np.meshgrid(X, Y)

    grid_z = np.array(griddata(points, pressure_int, (grid_x, grid_y), method='linear')) #interpolate the data
    pkl.dump(X, open('../Paper Figures/Data/No_Topo_Porosity_permeability_variation_data_X'+str(phi0)+'-'+str(k0)+'.p', 'wb'))
    pkl.dump(Y, open( '../Paper Figures/Data/No_Topo_Porosity_permeability_variation_data_Y'+str(phi0)+'-'+str(k0)+'.p', 'wb'))
    pkl.dump(grid_z, open( '../Paper Figures/Data/No_Topo_Porosity_permeability_variation_data_P'+str(phi0)+'-'+str(k0)+'.p', 'wb'))

########################################################################################################################
# Export list of gas height of an iteration
# Nx, Ny, Nz, List_t, AOI, H, H0, rho_g, rho_w, g, Outline must be provided

def export_h(iteration, Nx, Ny, Nz, List_t, AOI, Outline, phi0, k0):
    points_init, pressure_int, h = extract_points_p_h(iteration)
    points = [] ; h_data = []

    for i in tqdm(range(len(points_init))):
        points.append([points_init[i][0], points_init[i][1]])
        h_data.append(h[i])
    X = np.linspace(Outline[0][0], Outline[1][0], int(Nx)) ; Y = np.linspace(Outline[0][1], Outline[2][1], int(Ny))
    grid_x, grid_y = np.meshgrid(X, Y)

    grid_z = np.array(griddata(points, h_data, (grid_x, grid_y), method='linear')) #interpolate the data
    pkl.dump(X, open('../Paper Figures/Data/h_Topo_Porosity_permeability_variation_data_X'+str(phi0)+'-'+str(k0)+'.p', 'wb'))
    pkl.dump(Y, open( '../Paper Figures/Data/h_Topo_Porosity_permeability_variation_data_Y'+str(phi0)+'-'+str(k0)+'.p', 'wb'))
    pkl.dump(grid_z, open( '../Paper Figures/Data/h_Topo_Porosity_permeability_variation_data_P'+str(phi0)+'-'+str(k0)+'.p', 'wb'))

########################################################################################################################
# Export list of pressures for mechanical model
# folder_input, folder_output, Nt must be provided
def extract_pressure_list(folder_output, Nt):
    pressure_list = []
    reader_p = vtk.vtkXMLUnstructuredGridReader()
    reader_p.SetFileName(file_name_p(folder_output, 0))
    reader_p.Update()
    points = vtk_to_numpy(reader_p.GetOutput().GetPoints().GetData()).tolist()
    for i in tqdm(range(Nt)):
        reader_p.SetFileName(file_name_p(folder_output, i))
        reader_p.Update()  # Needed because of GetScalarRange
        pressure_int = vtk_to_numpy(reader_p.GetOutput().GetPointData().GetArray(0)).tolist()
        pressure_list.append(pressure_int)

    pkl.dump(np.array(pressure_list), open(folder_output + 'pressure_list.npy', 'wb'))

########################################################################################################################
# Compute RMSE and MAE
# --------- RMSE ---------------------------------------------------------------------------------------
def well_pressure_difference_RMSE(List_index, Nt, List_t, Well, BottomHP: pd.DataFrame, PArray_):
        well_difference = 0
        n_measures = 0
        
        if Nt > len(BottomHP): IHM = len(BottomHP)
        else: IHM = Nt
        T0 = int(List_t[0]/2629800)
        for j in range(IHM):
            T = int(List_t[j]/2629800)
            if T != T0:
                T0 = T
                for i in range(Well.shape[0]):
                    if T < len(BottomHP[Well[i][0]]):
                        if np.isfinite(BottomHP[Well[i][0]][T]):
                            well_difference += abs((BottomHP[Well[i][0]][T] - PArray_[List_index[i],j]))**2
                            n_measures += 1
            
        return np.sqrt(well_difference), np.sqrt(n_measures), np.sqrt(well_difference)/np.sqrt(n_measures)/1e6
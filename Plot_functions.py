import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from colorama import Fore, Back, Style

import matplotlib as mpl

from Auxiliary_class import*
from Auxiliary_functions import*

########################################################################################################################
# plot the Area Of Interest
def plot_AOI(ax, AOI, color): ax.plot(np.array(AOI)[:,0], np.array(AOI)[:,1], color=color, linewidth=3)

########################################################################################################################
# plot the pressure at the interface for a given iteration
# AOI, p_min, p_max, List_t, ME1 must be provided

def plot_p(output_folder, figures_folder, iteration, AOI, p_min, p_max, List_t, ME1, xmin, xmax, ymin, ymax):
    points = extract_points_p_h(output_folder, iteration)[0]
    pressure = extract_points_p_h(output_folder, iteration)[1]
    points = [[points[i][0], points[i][1]] for i in range(len(points))]
    
    interp = RBFInterpolator(points, pressure)
    Data = interp(ME1.tabulate_dof_coordinates())
    
    P = Function(ME1)
    P_moy = np.nanmean(Data) # Average on the 2D domain
    for l in range(ME1.dim()):
        if np.isnan(Data[l]): # if there is a NaN value, we give the precedent value to the vector.
            P.vector()[l] = P_moy
            print('Warning : NaN found after interpolation, replacing by mean value')
        else: P.vector()[l] = Data[l]

    print('Maximum pressure at the interface at t = ' + day_format(List_t[iteration]) + ' : ' + str(np.amax(pressure)) + ' bar')
    print('After interpolation : ' + str(np.amax(Data)) + ' bar')
    fig = plt.figure(figsize=(6,6))
    plt.colorbar(plot(P, mode='color', cmap='Spectral_r', vmin=p_min, vmax=p_max), orientation='vertical', label = 'pressure (bar)')
    plt.title('Pressure at t = ' + day_format(List_t[iteration]))
    ax = plt.gca() ; plot_AOI(ax, np.array(AOI), 'red')
    ax.set_xlabel('Distance (m)') ; ax.set_ylabel('Distance (m)')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.savefig(os.path.join(figures_folder, 'QUEST_p_int.png'), dpi=300, bbox_inches='tight')

########################################################################################################################
# plot the h for a given iteration
# AOI, h_min, h_max, List_t, ME1 must be provided

def plot_h(output_folder, figures_folder, iteration, AOI, h_min, h_max, List_t, ME1, xmin, xmax, ymin, ymax):
    points = extract_points_p_h(output_folder, iteration)[0]
    h = extract_points_p_h(output_folder, iteration)[2]
    points = [[points[i][0], points[i][1]] for i in range(len(points))]
    
    interp = RBFInterpolator(points, h)
    Data = interp(ME1.tabulate_dof_coordinates())
    
    h_function = Function(ME1)
    h_moy = np.nanmean(Data) # Average on the 2D domain
    for l in range(ME1.dim()):
        if np.isnan(Data[l]): # if there is a NaN value, we give the precedent value to the vector.
            h_function.vector()[l] = h_moy
            print('Warning : NaN found after interpolation, replacing by mean value')
        else: h_function.vector()[l] = Data[l]

    print('Maximum pressure at the interface at t = ' + day_format(List_t[iteration]) + ' : ' + str(np.amax(h)) + ' bar')
    print('After interpolation : ' + str(np.amax(Data)) + ' bar')
    fig = plt.figure(figsize=(6, 6))
    plt.colorbar(plot(h_function, mode='color', cmap='YlGnBu', vmin=h_min, vmax=h_max), orientation='vertical')
    plt.title('Thickness of gas (m) at t = ' + day_format(List_t[iteration]))
    ax = plt.gca() ; plot_AOI(ax, np.array(AOI), 'red')
    ax.set_xlabel('Distance (m)') ; ax.set_ylabel('Distance (m)')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.savefig(os.path.join(figures_folder, 'QUEST_h.png'), dpi=300, bbox_inches='tight')

########################################################################################################################
# plot temporal profile of a source
# T_tot must be provided

def plot_source(source, T_tot,ax):
    if len(source[2]) == 0:
        X = [0, T_tot/(365*24*3600)] ; Y = [0, 0]
    else:
        X = [0] ; Y = [0]
        for i in range(len(source[2])):
            Inj_profile = source[2][i] ; rate = source[1][i]
            X.append(Inj_profile[0]/(365*24*3600)) ; X.append(Inj_profile[0]/(365*24*3600)) ; X.append(Inj_profile[1]/(365*24*3600)) ; X.append(Inj_profile[1]/(365*24*3600))
            Y.append(0) ; Y.append(rate) ; Y.append(rate) ; Y.append(0)
        X.append(T_tot/(365*24*3600)) ; Y.append(0)
    ax.plot(X, Y)
    ax.set_xlabel('time (year)') ; ax.set_ylabel('Injection rate (kg.s-1)')


########################################################################################################################
# plot wells positions
# Sources_g must be provided

def plot_wells(Sources_g, color):
    for source in Sources_g:
        plt.plot(source[0][0], source[0][1], marker="o", markersize=2, markeredgecolor=color, markerfacecolor=color)

########################################################################################################################
# Plot pressure evolution at different points, and store the figure into a folder
# mesh, Nt, List_t must be provided
    
def plot_pressure_point(folder, Points, mesh, Nt, List_t):
    plt.figure(figsize=(12,6))
    
    for point in Points:
        distance = 1e10
        x, y = point[0], point[1]
        index = 0
        # Search the index of the point of the mesh which is the closest to (x,y)
        for i in range(len(mesh.coordinates())):
            c = mesh.coordinates()[i]
            if np.sqrt((c[0]-x)**2 + (c[1]-y)**2) < distance:
                distance = np.sqrt((c[0]-x)**2 + (c[1]-y)**2)
                index = i
        T = [] ; P = []
        for i in tqdm(range(Nt)):
            T.append(List_t[i]/(365*24*3600))
            P.append(extract_points_p_h(i)[1][index])
        plt.plot(T[int(Nt/2):], P[int(Nt/2):], label='(' + str(x) + ', '+ str(y) + ')')
    
    plt.title('Pressure at the interface through time at coordinates : (' + str(x) + ', '+ str(y) + ')')
    plt.xlabel('Time (years)')
    plt.ylabel('Pressure (bar)')
    plt.legend()
    plt.savefig(folder + '/Pressure evolution_at_each_well.png', dpi=300, bbox_inches='tight')

########################################################################################################################
# Plot total extracted mass over the extraction/injection period
# Folder, List_t, List_found_extracted_mass, List_expected_extracted_mass, List_err1, T_tot must be provided
def plot_mass(Folder, Nt, List_t, timef, time1, time0, Mass_found0, List_expected_extracted_mass, List_found_extracted_mass, Out_rsrvr, T_tot, List_err1):
    plt.rcParams.update({'font.size': 11})
    fig = plt.subplots(figsize=(14, 10))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = ax1.twinx()

    ax1.set_ylabel('Mass (Mt)')
    ax1.plot(np.array(List_t)/(24*3600*365), 1e-9*Mass_found0-abs(np.array(List_found_extracted_mass)), color='blue', label='Extracted mass found')
    ax1.plot(np.array(List_t)/(24*3600*365), 1e-9*Mass_found0-abs(np.array(List_expected_extracted_mass)), color='black', label='Expected extracted mass', linestyle='dotted')
    ax1.plot([], [], color='red', label='Relative error', linestyle='--')
    ax1.legend(loc = 'upper right')
    ax1.set_xlim([10, T_tot/(365*24*3600)])
    ax1.set_ylim([-0.05*1.3*np.max(abs(np.array(List_expected_extracted_mass))), 1.3*np.max(abs(np.array(List_expected_extracted_mass)))])
    ax2.plot(np.array(List_t)/(24*3600*365), 100*np.array(List_err1), color='red', label='Relative error', linestyle='--')
    ax2.set_ylabel('Error (%)')
    ax2.set_ylim([-5, 100])
    plt.title('Cumulative mass extracted')
    plt.savefig(Folder + '/Mass_extracted.png', dpi=300, bbox_inches='tight')
    
    plt.subplot(2, 1, 2)
    plt.xlabel('Time (year)') ; plt.ylabel('Mass (Mt/month)')
    plt.plot(np.array(List_t[:-1])/(24*3600*365), abs(np.array(List_found_extracted_mass[1:])-np.array(List_found_extracted_mass[:-1])), color='green', label='Instantaneous extracted mass found')
    plt.plot(np.array(List_t[:-1])/(24*3600*365), abs(np.array(List_expected_extracted_mass[1:])-np.array(List_expected_extracted_mass[:-1])), color='black', label='Instantaneous expected extracted mass', linestyle='dotted')
    plt.xlim([10, T_tot/(365*24*3600)])
    plt.ylim([-0.05*1.3*np.max(abs(np.array(List_expected_extracted_mass[1:])-np.array(List_expected_extracted_mass[:-1]))), 1.3*np.max(abs(np.array(List_expected_extracted_mass[1:])-np.array(List_expected_extracted_mass[:-1])))])
    plt.title('Instantaneous mass extracted')
    plt.legend()
    plt.savefig(Folder + '/Mass_extracted.png', dpi=300, bbox_inches='tight')
    
########################################################################################################################
# Plot initial conditions
# Folder, Sources_g, u, h_field, Out_rsrvr must be provided
def plot_initial_conditions(Folder, Sources_g, u, h_field, Out_rsrvr):
    plt.subplots(figsize=(12,5.5))
    plt.subplot(1,2,1) ; plot_wells(Sources_g, 'black')
    plt.colorbar(plot(u.split()[1], title='Initial gas thickness (m)', mode='color', cmap='rainbow', vmin=np.nanmin(h_field), vmax= np.nanmax(h_field)), orientation='vertical')
    plt.plot(*Out_rsrvr.T, color='black', label='Reservoir outline')
    plt.xlabel("Distance (m)") ; plt.ylabel("Distance (m)")
    plt.subplot(1,2,2) ; plot_wells(Sources_g, 'red')
    plt.colorbar(plot(u.split()[0], title='Initial pressure (Pa)', mode='color', cmap='jet_r'), orientation='vertical') #,vmin=p_0, vmax= p_0-25e5
    plt.plot(*Out_rsrvr.T, color='black', label='Reservoir outline')
    plt.xlabel("Distance (m)") ; plt.ylabel("Distance (m)")
    plt.savefig(Folder + '/Initial_conditions.png', dpi=300, bbox_inches='tight')

########################################################################################################################
# Plot extraction profile
# Folder, Sources_g, T_tot must be provided
def plot_extraction_data(Folder, Sources_g, T_tot):
    Ns = len(Sources_g)
    fig,ax = plt.subplots(1,1,figsize=(11,2.5))
    for i in range(Ns): plot_source(Sources_g[i], T_tot,ax)
    plt.savefig(Folder + '/Extraction profile.png', dpi=300, bbox_inches='tight')

########################################################################################################################
# Plot the pressure profiles at one well
def plot_well_pressure(Folder, List_index, Nt, List_t, Wells, well: str, BottomHP: pd.DataFrame, PArray_, year):
    List_time_measure = [] ; List_p_measure = []
    
    List_time_simu = np.array(List_t)/year + 1956 ; List_p_simu = []

    # Search index of well
    i = 0
    indic = 0
    while indic == 0:
        if Wells[i][0] == well:
            index = i
            indic = 1
        else: i += 1
    
    List_p_simu = PArray_[List_index[index],:]/1e6
    
    for j in range(len(BottomHP)):
        List_time_measure.append(2629800*j/year + 1956)
        List_p_measure.append(BottomHP[Wells[index][0]][j]/1e6)
    
    # Plot
    plt.figure(figsize=(13,5))
    plt.scatter(List_time_measure, List_p_measure, c = List_p_measure, cmap='Spectral', marker='s', s=50, label='Pressure measurements', zorder=1)
    plt.plot(List_time_simu, List_p_simu, color='black', label='VFE-MP', linestyle = '--', zorder=2)
    plt.axhline(y = 0, color = 'grey', linestyle = '--')
    
    plt.legend()
    plt.colorbar(label='Pressure (MPa)')
    plt.ylabel('MPa')
    plt.xlabel('Year')
    plt.title('Pressure at well ' + well)
    plt.savefig(Folder + '/Pressure_profile_' +well +'.png', dpi=300, bbox_inches='tight')


########################################################################################################################
# Plot porosity (phi), permeability (k), reservoir thickness (H) and reservoir depth
# Folder, Sources_g, Out_rsrvr must be provided
def plot_phi_k_H_depth(Folder, Sources_g, Out_rsrvr, k, H, phi, depth):
    plt.subplots(figsize=(12,12))
    ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e3))
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1e3))
    
    plt.subplot(2,2,1) ; plot_wells(Sources_g, 'red')
    plt.colorbar(plot(depth, title='Depth (m)', mode='color'), orientation='vertical')
    plt.plot(*Out_rsrvr.T, color='black', label='Reservoir outline')
    plt.xlabel("Distance (m)") ; plt.ylabel("Distance (m)")
    plt.gca().xaxis.set_major_formatter(ticks_x) ; plt.gca().yaxis.set_major_formatter(ticks_y)
    
    plt.subplot(2,2,2) ; plot_wells(Sources_g, 'white')
    plt.colorbar(plot(H, title='Thickness of the reservoir (m)', mode='color', cmap='plasma'), orientation='vertical');
    plt.plot(*Out_rsrvr.T, color='black', label='Reservoir outline')
    plt.xlabel("Distance (m)") ; plt.ylabel("Distance (m)")
    plt.gca().xaxis.set_major_formatter(ticks_x) ; plt.gca().yaxis.set_major_formatter(ticks_y)
    
    plt.subplot(2,2,3) ; plot_wells(Sources_g, 'red')
    plt.colorbar(plot(phi, title='Porosity', mode='color', cmap='Blues'), orientation='vertical')
    plt.plot(*Out_rsrvr.T, color='black', label='Reservoir outline')
    plt.xlabel("Distance (m)") ; plt.ylabel("Distance (m)")
    plt.gca().xaxis.set_major_formatter(ticks_x) ; plt.gca().yaxis.set_major_formatter(ticks_y)
    
    plt.subplot(2,2,4) ; plot_wells(Sources_g, 'red')
    plt.colorbar(plot(k, title='Permeability (m2)', mode='color'), orientation='vertical');
    plt.plot(*Out_rsrvr.T, color='black', label='Reservoir outline')
    plt.xlabel("Distance (m)") ; plt.ylabel("Distance (m)")
    plt.gca().xaxis.set_major_formatter(ticks_x) ; plt.gca().yaxis.set_major_formatter(ticks_y)
    
    plt.savefig(Folder + '/H_phi_k_depth.png', dpi=300, bbox_inches='tight')

def plot_pressure_comparison(Folder_input, folder_figures, iterations, GasData, mesh, RES, Nt, List_t, x_min, x_max, y_min, y_max):
    plt.rcParams.update({'font.size': 12})
    BottomHP = GasData["PressureMeasurements"]
    N_plot = range(1, Nt)
    p_min_int, p_max_int, h_min, h_max = min_h_p(Folder_input, N_plot)
    offset = 1e3
    x_min, y_min = np.min(RES['Outline']['X'].values)-offset, np.min(RES['Outline']['Y'].values)-offset
    x_max, y_max = np.max(RES['Outline']['X'].values)+offset, np.max(RES['Outline']['Y'].values)+offset

    keys = []
    for key, value in BottomHP.items(): keys.append(key)
    Lists_pressure_simu = [[] for i in range(len(keys))]
    offset = 10

    List_index = []
    for i in range(len(keys)):
        distance = 1e10
        point = Point(int(GasData['WellLocations'][i][1]), int(GasData['WellLocations'][i][2]))
        index = 0
        # Search the index of the point of the mesh which is the closest to (x,y)
        for j in range(len(mesh.coordinates())):
            c = mesh.coordinates()[j]
            if np.sqrt((c[0]-point[0])**2 + (c[1]-point[1])**2) < distance:
                distance = np.sqrt((c[0]-point[0])**2 + (c[1]-point[1])**2)
                index = j
        List_index.append(index)

    fig = plt.figure(figsize=(14,17))
    ax = fig.add_gridspec(12, 3)
    ax1 = fig.add_subplot(ax[0:12, 1:3])

    for i in tqdm(range(Nt+1)):
        p = extract_points_p_h(Folder_input, i)[1] #extract pressure
        T = [] ; P = []
        for j in range(len(keys)): Lists_pressure_simu[j].append(1e-1*p[List_index[j]])    

    for i in range(len(keys)-1):
        ax1.plot(1956+np.array(List_t)/(365*24*3600), np.array(Lists_pressure_simu[i])+np.array(i*offset), color='black', linestyle='--')
        N = [1956+j*2629800/365/24/3600 for j in range(len(BottomHP[str(key)]))]
        P = [BottomHP[str(keys[i])][k] for k in range(len(BottomHP[str(keys[i])]))]
        Nclean = [] ; Pclean = []
        for j in range(len(N)):
            if not np.isnan(P[j]):
                Nclean.append(N[j])
                Pclean.append(P[j])
        im = ax1.scatter(Nclean, 1e-6*np.array(Pclean)+i*offset, c=1e-6*np.array(Pclean), s=35,
                         marker='s', cmap='viridis', vmin=1e-1*p_min_int, vmax=1e-1*p_max_int)
        ax1.text(1956+8, 36+i*offset, keys[i], fontsize=14, color='black')

    ax1.plot(1956+np.array(List_t)/(365*24*3600), np.array(Lists_pressure_simu[len(keys)-1])+np.array(len(keys)-1*offset),
             color='black', linestyle='--')
    N = [1956+j*2629800/365/24/3600 for j in range(len(BottomHP[str(key)]))]
    P = [BottomHP[str(keys[len(keys)-1])][k] for k in range(len(BottomHP[str(keys[len(keys)-1])]))]
    Nclean = [] ; Pclean = []
    for j in range(len(N)):
        if not np.isnan(P[j]):
            Nclean.append(N[j])
            Pclean.append(P[j])
    im = ax1.scatter(Nclean, 1e-6*np.array(Pclean)+len(keys)-1*offset, c=1e-6*np.array(Pclean), s=35,
                     marker='s', cmap='viridis', vmin=1e-1*p_min_int, vmax=1e-1*p_max_int, label='Measurements')

    ax1.text(1956+8, 36+i*offset, keys[i], fontsize=14, color='black')
    ax1.plot([], [], color='black', linestyle='--', label='VFE-MP')
    ax1.get_yaxis().set_visible(False)
    ax1.set_xlabel('Year')
    ax1.plot([1965, 1965], [2, 22], color='black')
    ax1.plot([1964, 1966], [22, 22], color='black')
    ax1.plot([1964, 1966], [2, 2], color='black')
    ax1.text(1967, 11, '20 MPa')
    ax1.legend()
    cbax = ax1.inset_axes([1.05, 0, 0.03, 1], transform=ax1.transAxes)
    fig.colorbar(im, cax=cbax, orientation='vertical', label='Pressure (MPa)')

    fontisze = 9
    p_min = 1e10
    p_max = 0

    Outline_rsrvr = [Point(x,y) for x,y in zip(reversed(RES['Outline']['X'].values),reversed(RES['Outline']['Y'].values))] 
    Outline_rsrvr.append(Point(RES['Outline']['X'].values[0],RES['Outline']['Y'].values[0]))
    Outline_rsrvr = [[point[0], point[1]] for point in Outline_rsrvr]
    Outline_rsrvr_array = np.array(Outline_rsrvr)

    for k in range(3):
        pressure = extract_points_p_h(Folder_input, iterations[k])[1]
        if 1e-1*np.nanmin(pressure) < p_min: p_min=1e-1*np.nanmin(pressure)
        if 1e-1*np.nanmax(pressure) > p_max: p_max=1e-1*np.nanmax(pressure)

    for k in range(3):
        ax2 = fig.add_subplot(ax[4*k:4*(k+1), 0])
        ax2.set_xlabel('Distance (km)') ; ax2.set_ylabel('Distance (km)')
        for i in range(len(keys)):
            point = Point(1e-3*int(GasData['WellLocations'][i][1]), 1e-3*int(GasData['WellLocations'][i][2]))
            if keys[i]=='SP1':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=7)
                ax2.text(point[0]-3, point[1], "  " + "SP1,2", fontsize=fontisze)
            elif keys[i]=='SP2':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=7)
            elif keys[i]=='SDB':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-3.5, point[1]-2, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='ER1':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-1.7, point[1]-2, "  " + "ER1,2", fontsize=fontisze)
            elif keys[i]=='ER2':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
            elif keys[i]=='ZVN':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-2.5, point[1]+1, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='UTB':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-1.2, point[1]-1.7, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='TUS':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-0.5, point[1]-2, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='TJM':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-3, point[1]+0.5, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='MWD':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-1.5, point[1]-2.5, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='TJM':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-1.2, point[1]+0.5, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='EKL':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0], point[1]-1, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='SAP':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-4.7, point[1]-0.5, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='OVS':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-0.3, point[1]+0.2, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='SLO':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-4.5, point[1]-0.7, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='KPD':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-4.5, point[1], "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='FRB':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-1, point[1]+0.8, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='ZW1':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-2, point[1]+0.3, "  " + "ZW1,2", fontsize=fontisze)
            elif keys[i]=='ZW2':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
            elif keys[i]=='NBR':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-1.5, point[1]+0.1, "  " + keys[i], fontsize=fontisze)
            elif keys[i]=='NWS':
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0]-2, point[1]+0.8, "  " + keys[i], fontsize=fontisze)
            else:
                ax2.plot(point[0], point[1], color='white', marker="o", markeredgecolor='black', markersize=5)
                ax2.text(point[0], point[1], "  " + keys[i], fontsize=fontisze)  

        points = extract_points_p_h(Folder_input, iterations[k])[0]
        pressure = extract_points_p_h(Folder_input, iterations[k])[1]
        points = [[points[i][0], points[i][1]] for i in range(len(points))]

        ax2.tricontourf(np.array(points)[:,0]*1e-3, np.array(points)[:,1]*1e-3, 1e-1*np.array(pressure),
                                     100, cmap = 'Spectral', vmin=p_min, vmax=p_max)
        ax2.plot(1e-3*Outline_rsrvr_array[:, 0], 1e-3*Outline_rsrvr_array[:, 1], color='black', label='Reservoir outline')
        ax2.set_ylabel('Distance (km)')
        Outline_rsrvr = [[x,y] for x,y in zip(reversed(RES['Outline']['X'].values[0::20]),reversed(RES['Outline']['Y'].values[0::20]))]
        X = np.concatenate((np.array([[200, 500]]), 1e-3*np.array(Outline_rsrvr)[100:], 1e-3*np.array(Outline_rsrvr)[:100], np.array([[200, 500]]), np.array([[200, 700]]), np.array([[300, 700]]), np.array([[300, 500]]), np.array([[200, 500]])))[:,0]
        Y = np.concatenate((np.array([[200, 500]]), 1e-3*np.array(Outline_rsrvr)[100:], 1e-3*np.array(Outline_rsrvr)[:100], np.array([[200, 500]]), np.array([[200, 700]]), np.array([[300, 700]]), np.array([[300, 500]]), np.array([[200, 500]])))[:,1]
        ax2.fill(X,Y, "white")
        ax2.set_ylim([557, 617])
        ax2.set_xlim([(x_min-1000)*1e-3, (x_max+1000)*1e-3]) ; plt.ylim([(y_min-1000)*1e-3, (y_max+1000)*1e-3])
        ax2.set_title(int(1956+List_t[iterations[k]]/365/24/3600))
        ax2.get_xaxis().set_visible(False)
        
    ax2.get_xaxis().set_visible(True)
    ax2.set_xlabel('Distance (km)')

    ax4  = fig.add_axes([0.05, 0.125, 0.02, 0.755])
    norm = mpl.colors.Normalize(vmin=p_min,vmax=p_max)
    cb1  = mpl.colorbar.ColorbarBase(ax4,cmap='Spectral', norm=norm, orientation='vertical', label='Pressure (MPa)')
    ax4.yaxis.set_ticks_position('left')
    ax4.yaxis.set_label_position('left')

    ax1.set_xlim([1956+5, 1956+70]) ; ax1.set_ylim([0, 320])
    fig.subplots_adjust(wspace=0.1) ; fig.subplots_adjust(hspace=0.5)

    plt.savefig(folder_figures + '/Pressure_profile_gas_extracted.png', dpi=300, bbox_inches='tight')
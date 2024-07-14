from dolfin import*
import numpy as np
from scipy.interpolate import RBFInterpolator

########################################################################################################################
# Class to define the nonlinear problem

class VFE_MP_model(NonlinearProblem):
    def __init__(self, a, L, bcs):
        self.L = L
        self.a = a
        self.bcs = bcs
        NonlinearProblem.__init__(self)
    def F(self, b, x):
        assemble(self.L, tensor=b)
        if self.bcs != []:
            for bc in self.bcs:
                bc.apply(b, x)
    def J(self, A, x):
        assemble(self.a, tensor=A)
        # if self.bcs != []:
            # for bc in self.bcs:
            #     bc.apply(A)

########################################################################################################################
# Class to define initial conditions

# class InitialConditions(UserExpression):
#     def __init__(self, p_points, p_field, h_points, h_field, ME1):
#         super().__init__()
#         self.p_points = p_points
#         self.p_field = p_field
#         self.h_points = h_points
#         self.h_field = h_field
#         self.Coords = ME1.tabulate_dof_coordinates()
#         self.ME1 = ME1
#     def eval(self, values, x):
#         interp_p = RBFInterpolator(self.p_points, self.p_field)
#         interp_h = RBFInterpolator(self.h_points, self.h_field)
#         p = interp_p([x])[0]
#         h = interp_h([x])[0]
#         if np.isnan(p):
#             p_moy = np.nanmean(interp_p(self.Coords))
#             p = p_moy
#             print('Warning : NaN found after interpolation, replacing by mean value')
#         if np.isnan(h):
#             h_moy = np.nanmean(interp_h(self.Coords))
#             h = h_moy
#             print('Warning : NaN found after interpolation, replacing by mean value')
#         values[0] = p
#         values[1] = h
#     def value_shape(self):
#         return (2,)

class InitialConditions(UserExpression):
    def __init__(self, p_init, h_init, Coords, ME1):
        super().__init__()
        self.p_init = p_init
        self.h_init = h_init
        self.Coords = Coords
        self.ME1 = ME1
    def eval(self, values, x):
        i = np.intersect1d(np.where(self.ME1.tabulate_dof_coordinates()[:,0] == x[0]), np.where(self.ME1.tabulate_dof_coordinates()[:,1] == x[1]))
        values[0] = self.p_init[i]
        values[1] = self.h_init[i]
    def value_shape(self):
        return (2,)
    
########################################################################################################################
# Class to define boundary conditions

# class BoundaryConditions(UserExpression):
#     def __init__(self, p_points, p_field, h_points, h_field, ME1):
#         super().__init__()
#         self.p_points = p_points
#         self.p_field = p_field
#         self.h_points = h_points
#         self.h_field = h_field
#         self.Coords = ME1.tabulate_dof_coordinates()
#         self.ME1 = ME1
#     def eval(self, values, x):
#         interp_p = RBFInterpolator(self.p_points, self.p_field)
#         interp_h = RBFInterpolator(self.h_points, self.h_field)
#         p = interp_p([x])[0]
#         h = interp_h([x])[0]
#         if np.isnan(p):
#             p_moy = np.nanmean(interp_p(self.Coords))
#             p = p_moy
#             print('Warning : NaN found after interpolation, replacing by mean value')
#         if np.isnan(h):
#             h_moy = np.nanmean(interp_h(self.Coords))
#             h = h_moy
#             print('Warning : NaN found after interpolation, replacing by mean value')
#         values[0] = p
#         values[1] = h
#     def value_shape(self):
#         return (2,)

class BoundaryConditions(UserExpression):
    def __init__(self, p_init, h_init, Coords, ME1):
        super().__init__()
        self.p_init = p_init
        self.h_init = h_init
        self.Coords = Coords
        self.ME1 = ME1
    def eval(self, values, x):
        i = np.intersect1d(np.where(self.ME1.tabulate_dof_coordinates()[:,0] == x[0]), np.where(self.ME1.tabulate_dof_coordinates()[:,1] == x[1]))
        values[0] = self.p_init[i]
        values[1] = self.h_init[i]
    def value_shape(self):
        return (2,)
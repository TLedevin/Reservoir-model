from dolfin import*
import numpy as np
from scipy.interpolate import RBFInterpolator
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
        if self.bcs != []:
            for bc in self.bcs:
                bc.apply(A)
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
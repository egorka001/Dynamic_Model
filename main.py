import sympy as sym
import time

class Solver:
    def __init__(self):
        #KUKA KR300 R2500 ULTRA
        self.dh = [{"alpha": sym.pi, "a": 0, "theta": 0, "d": -675},
                   {"alpha": sym.pi / 2, "a": 350, "theta": 0, "d": 0},
                   {"alpha": 0, "a": 1150, "theta": -sym.pi / 2, "d": 0},
                   {"alpha": sym.pi / 2, "a": -41, "theta": 0, "d": -1000},
                   {"alpha": -sym.pi / 2, "a": 0, "theta": 0, "d": 0},
                   {"alpha": sym.pi / 2, "a": 0, "theta": 0, "d": 0},
                   {"alpha": sym.pi, "a": 0, "theta": sym.pi, "d": 240}]
        
        self.time = sym.Symbol('t', real=True)
        
        #variabels for robot angle
        self.q = [sym.Symbol('q0', real=True),
                  sym.Symbol('q1', real=True),
                  sym.Symbol('q2', real=True),
                  sym.Symbol('q3', real=True),
                  sym.Symbol('q4', real=True),
                  sym.Symbol('q5', real=True),
                  sym.Symbol('q6', real=True)]
        
        self.q[0] = sym.Function('q0', real=True)(self.time)
        self.q[1] = sym.Function('q1', real=True)(self.time)
        self.q[2] = sym.Function('q2', real=True)(self.time)
        self.q[3] = sym.Function('q3', real=True)(self.time)
        self.q[4] = sym.Function('q4', real=True)(self.time)
        self.q[5] = sym.Function('q5', real=True)(self.time)
        self.q[6] = sym.Function('q6', real=True)(self.time)
        
        
        #variabels for ropot angle velocity
        self.q_dot = [sym.Symbol('q0_dot', real=True),
                      sym.Symbol('q1_dot', real=True),
                      sym.Symbol('q2_dot', real=True),
                      sym.Symbol('q3_dot', real=True),
                      sym.Symbol('q4_dot', real=True),
                      sym.Symbol('q5_dot', real=True),
                      sym.Symbol('q6_dot', real=True)]
        
        self.q_dot[0] = self.q[0].diff(self.time, real=True)
        self.q_dot[1] = self.q[1].diff(self.time, real=True)
        self.q_dot[2] = self.q[2].diff(self.time, real=True)
        self.q_dot[3] = self.q[3].diff(self.time, real=True)
        self.q_dot[4] = self.q[4].diff(self.time, real=True)
        self.q_dot[5] = self.q[5].diff(self.time, real=True)
        self.q_dot[6] = self.q[6].diff(self.time, real=True)
        
        #print(sym.diff(self.q_dot[6], self.time, real=True))

        #rotation matrix for transformation
        self.R = [0]
        for i in range(1, 7):
            self.R.append(self._get_rot_matrix(i))
            
        #move matrix for transformation
        self.T = [0]
        for i in range(1, 7):
            self.T.append(self._get_move_matrix(i))
            
        #mass center in loacl coordinate system
        self.rc_local = [0]
        for i in range(1, 7):
            self.rc_local.append(sym.Matrix([[sym.Symbol('x_rc' + str(i), real=True)], 
                                             [sym.Symbol('y_rc' + str(i), real=True)], 
                                             [sym.Symbol('z_rc' + str(i), real=True)]]))
        
        #mass center in gloval coordinate system
        self.rc_abs = [0]
        for i in range(1, 7):
            to_append = self._get_dh_matrix(1)
            for j in range(2, i + 1):
                to_append *= self._get_dh_matrix(i)
            ext_rc_i = sym.Matrix([[self.rc_local[i][0]],
                                   [self.rc_local[i][1]],
                                   [self.rc_local[i][2]],
                                   [1]])
            to_append *= ext_rc_i
            self.rc_abs.append(sym.Matrix([[to_append[0]],
                                           [to_append[1]],
                                           [to_append[2]]]))
                
        
        #tensor of inertia
        self.I = [0]
        for i in range(1, 7):
            self.I.append(sym.Matrix([['Ixx' + str(i), 'Ixy' + str(i), 'Ixz' + str(i)],
                                      ['Iyx' + str(i), 'Iyy' + str(i), 'Iyz' + str(i)],
                                      ['Izx' + str(i), 'Izy' + str(i), 'Izz' + str(i)]], real=True))

        #mass of joint
        self.m = [0]
        for i in range(1, 7):
            self.m.append(sym.Symbol('m' + str(i), real=True))
            
        self.g = sym.Matrix([[0],
                             ['g'],
                             [0]], real=True)
            
    #get kinematic energy equasions
    def _get_kinetic_eq(self):
        self.w = [sym.Matrix([[0], [0], [0]], real=True)]
        for i in range(1, 7):
            self.w.append(self.R[i].transpose() * (self.w[i - 1] + self.q_dot[i] * (sym.Matrix([[0], [0], [1]], real=True))))
            
        self.v = [sym.Matrix([[0], [0], [0]])]
        for i in range(1, 7):
            self.v.append(self.R[i].transpose() * (self.v[i - 1] + sym.Matrix.cross(self.w[i - 1] + self.q_dot[i] * sym.Matrix([[0], [0], [1]], real=True), self.T[i])))
    
        self.vc = [sym.Matrix([[0], [0], [0]])]
        for i in range(1, 7):
            self.vc.append(self.v[i] + sym.Matrix.cross(self.w[i], self.rc_local[i]))
            
        self.K = [0]
        for i in range(1, 7):
            self.K.append(sym.Rational(1, 2) * self.m[i] * (self.vc[i].norm() ** 2) + (sym.Rational(1, 2) * self.w[i].transpose() * self.I[i] * self.w[i])[0])
            
            
    #get kinematic energy equasions
    def _get_potential_eq(self):
        self.P = [0]
        for i in range(1, 7):
            self.P.append((self.m[i] * (self.g.transpose() * self.rc_abs[i]))[0])
    
    
    #get lagrangian
    def _get_L(self):
        self.L = [0]
        for i in range(1, 7):
            self.L.append(self.K[i] - self.P[i])
        for i in range(1, 7):
            self.L[0] += self.L[i]
            
    #get lagrange equations
    def _get_euler_lagrange_eq(self):
        self.el_eq = [0]
        for i in range(1, 7):
            self.el_eq.append(sym.diff(sym.diff(self.L[0], self.q_dot[i], real=True), self.time, real=True) - sym.diff(self.L[0], self.q[i], real=True))
            with open("output.txt", 'a') as file:
                file.write(str(self.el_eq[i]) + '\n')
                file.write("-------------\n")
                
    #get 4x4 dh matrix of transformation from i-1 to i
    def _get_dh_matrix(self, i):
        Rx = sym.Matrix([[1, 0, 0, 0],
                         [0, sym.cos(self.dh[i-1]["alpha"]), -sym.sin(self.dh[i-1]["alpha"]), 0],
                         [0, sym.sin(self.dh[i-1]["alpha"]), sym.cos(self.dh[i-1]["alpha"]), 0],
                         [0, 0, 0, 1]])
        Tx = sym.Matrix([[1, 0, 0, self.dh[i-1]["a"]],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        Rz = sym.Matrix([[sym.cos(self.dh[i-1]["theta"] + self.q[i]), -sym.sin(self.dh[i-1]["theta"] + self.q[i]), 0, 0],
                         [sym.sin(self.dh[i-1]["theta"] + self.q[i]), sym.cos(self.dh[i-1]["theta"] + self.q[i]), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        Tz = sym.Matrix([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, self.dh[i-1]["d"]],
                         [0, 0, 0, 1]])
        return Rx * Tx * Rz * Tz
        
        
    #get 3x3 rotation matrix of transformation from i-1 to i
    def _get_rot_matrix(self, i):
        dh_matrix = self._get_dh_matrix(i)
        
        rot_matrix = sym.Matrix([[dh_matrix[0], dh_matrix[1], dh_matrix[2]],
                                 [dh_matrix[4], dh_matrix[5], dh_matrix[6]],
                                 [dh_matrix[8], dh_matrix[9], dh_matrix[10]]], real=True)
        
        return rot_matrix
    
    
    #get 3x1 move matrix of transformation from i-1 to i
    def _get_move_matrix(self, i):
        dh_matrix = self._get_dh_matrix(i)
        
        move_matrix = sym.Matrix([[dh_matrix[3]],
                                  [dh_matrix[7]],
                                  [dh_matrix[11]]], real=True)
        
        return move_matrix
    
    
    def print_dh(self):
        print(self._get_dh_matrix(1))
        print(self._get_rot_matrix(1))
        print(self._get_move_matrix(1))


if __name__ == '__main__':
    sym.init_printing(use_unicode=False)
    
    start_time = time.time()
    
    DynamicModel = Solver()
    DynamicModel._get_kinetic_eq()
    DynamicModel._get_potential_eq()
    DynamicModel._get_L()
    DynamicModel._get_euler_lagrange_eq()
    
    end_time = time.time()
    
    execution_time = end_time - start_time

    print(f"Время выполнения: {execution_time} секунд")
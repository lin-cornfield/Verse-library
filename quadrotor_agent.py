# Example agent.
from typing import Tuple, List

import numpy as np
import math
from scipy.integrate import ode
# from ddeint import ddeint #pip install ddeint
# from jitcdde import jitcdde, y, t  # pip install jitcdde
import numpy.linalg as la

from verse import BaseAgent
from verse.map import LaneMap

class quadrotor_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        super().__init__(id, code, file_name)
        # step0-1: start to do some initialization for system params and L1 inputs
        # self.m  = 0.752 # mass for controller design 
        # self.g = 9.81
        # self.J = 1e-3*np.diag([2.5, 2.1, 4.3])
        self.m  = 4.34 # mass for controller design 
        self.g = 9.81
        self.J = 1e-2*np.diag([8.2, 8.45, 13.77])
        self.e3 = np.array([0.,0.,1.])

        """ L1-related parameters """
        self.As_v = -5.0 # parameter for L1
        self.As_omega = -5.0 # parameter for L1
        # self.dt_L1 = 0.00003448 # sample time for L1 AC, for simplicity, set the same as the simulation step size
        # self.dt_L1 = 0.000033 # perfect value for time-delay system
        # self.dt_L1 = 0.015
        self.dt_L1 = 0.005
        """ For large uncertainties ..."""
        self.ctoffq1Thrust = 5*7 # cutoff frequency for thrust channel LPF (rad/s)
        self.ctoffq1Moment = 5*7 # cutoff frequency for moment channels LPF1 (rad/s)
        self.ctoffq2Moment = 5*7 # cutoff frequency for moment channels LPF2 (rad/s)

        self.L1_params = (self.As_v, self.As_omega, self.dt_L1, self.ctoffq1Thrust, self.ctoffq1Moment, self.ctoffq2Moment, self.m, self.g, self.J )

        """ Geometric control gains """
        self.kx = 16*self.m*np.ones((3,)) # position gains
        # self.kx = 16*self.m*np.array([0.5,0.5,2.0]) # position gains
        self.kv = 5.6*self.m*np.ones((3,)) # velocity gains
        self.kR = 8.81*np.ones((3,)) # angular gains
        self.kW = 2.54*np.ones((3,)) # rotational velocity gains
        # self.kW = 2.54*0.1*np.ones((3,)) # rotational velocity gains

        """ Initialization of L1 inputs """
        # self.use_l1ac = True
        self.use_uncertainty = False

        
        
        self.shift_index_prev = int(0)
        self.checker = 0

        # self.start_state = np.random.uniform(-0.5,0.5,20)

        # uncomment above to verify delay-related properties using dde integration (default use of mode 8)
        # self.f = [
        #     y(0, t-self.tau),
        #     y(1, t-self.tau),
        #     y(2, t-self.tau),
        #     y(3, t-self.tau),
        #     y(4, t-self.tau),
        #     y(5, t-self.tau),
        #     y(6, t-self.tau),
        #     y(7, t-self.tau),
        #     y(8, t-self.tau),
        #     y(9, t-self.tau),
        #     y(10, t-self.tau),
        #     y(11, t-self.tau),
        #     y(12, t-self.tau),
        #     y(13, t-self.tau),
        #     y(14, t-self.tau),
        #     y(15, t-self.tau),
        #     y(16, t-self.tau),
        #     y(17, t-self.tau),
        #     y(18, t-self.tau),
        #     y(19, t-self.tau)
        # ]
        


    def L1_AC(self, R, W, x, v, f, M):

        (As_v, As_omega, dt, ctoffq1Thrust, ctoffq1Moment, ctoffq2Moment, kg_vehicleMass, GRAVITY_MAGNITUDE, J ) = self.L1_params
        (v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev,
        u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev,
        lpf1_prev, lpf2_prev) = self.din_L1

        # == begin L1 adaptive control ==
        # first do the state predictor
        # load translational velocity
        v_now = v

        # load rotational velocity
        omega_now = W

        massInverse = 1.0 / kg_vehicleMass

        # compute prediction error (on previous step)
        vpred_error_prev = v_hat_prev - v_prev # computes v_tilde for (k-1) step
        omegapred_error_prev = omega_hat_prev - omega_prev # computes omega_tilde for (k-1) step

        v_hat = v_hat_prev + (self.e3 * GRAVITY_MAGNITUDE - R_prev[:,2]* (u_b_prev[0] + u_ad_prev[0] + sigma_m_hat_prev[0]) * massInverse + R_prev[:,0] * sigma_um_hat_prev[0] * massInverse + R_prev[:,1] * sigma_um_hat_prev[1] * massInverse + vpred_error_prev * As_v) * dt
        Jinv = la.inv(J)
        # temp vector: thrustMomentCmd[1--3] + u_ad_prev[1--3] + sigma_m_hat_prev[1--3]
        # original form
        tempVec = np.array([u_b_prev[1] + u_ad_prev[1] + sigma_m_hat_prev[1], u_b_prev[2] + u_ad_prev[2] + sigma_m_hat_prev[2], u_b_prev[3] + u_ad_prev[3] + sigma_m_hat_prev[3]])
        omega_hat = omega_hat_prev + (-np.matmul(Jinv, np.cross(omega_prev, (np.matmul(J, omega_prev)))) + np.matmul(Jinv, tempVec) + omegapred_error_prev * As_omega) * dt

        # update the state prediction storage
        v_hat_prev = v_hat
        omega_hat_prev = omega_hat

        # compute prediction error (for this step)
        vpred_error = v_hat - v_now
        omegapred_error = omega_hat - omega_now

        # exponential coefficients coefficient for As
        exp_As_v_dt = math.exp(As_v * dt)
        exp_As_omega_dt = math.exp(As_omega * dt)

        # latter part of uncertainty estimation (piecewise constant) (step2: adaptation law)
        PhiInvmu_v = vpred_error / (exp_As_v_dt - 1) * As_v * exp_As_v_dt
        PhiInvmu_omega = omegapred_error / (exp_As_omega_dt - 1) * As_omega * exp_As_omega_dt

        sigma_m_hat = np.array([0.0,0.0,0.0,0.0]) # estimated matched uncertainty
        sigma_m_hat_2to4 = np.array([0.0,0.0,0.0]) # second to fourth element of the estimated matched uncertainty
        sigma_um_hat = np.array([0.0,0.0]) # estimated unmatched uncertainty

        sigma_m_hat[0] = np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass
        # turn np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass to -np.dot(R[:,2], PhiInvmu_v) * kg_vehicleMass
        sigma_m_hat_2to4 = -np.matmul(J, PhiInvmu_omega)
        sigma_m_hat[1] = sigma_m_hat_2to4[0]
        sigma_m_hat[2] = sigma_m_hat_2to4[1]
        sigma_m_hat[3] = sigma_m_hat_2to4[2]

        sigma_um_hat[0] = -np.dot(R[:,0], PhiInvmu_v) * kg_vehicleMass
        sigma_um_hat[1] = -np.dot(R[:,1], PhiInvmu_v) * kg_vehicleMass

        # store uncertainty estimations
        sigma_m_hat_prev = sigma_m_hat
        sigma_um_hat_prev = sigma_um_hat

        # compute lpf1 coefficients
        lpf1_coefficientThrust1 = math.exp(- ctoffq1Thrust * dt)
        lpf1_coefficientThrust2 = 1.0 - lpf1_coefficientThrust1

        lpf1_coefficientMoment1 = math.exp(- ctoffq1Moment * dt)
        lpf1_coefficientMoment2 = 1.0 - lpf1_coefficientMoment1

        # update the adaptive control
        u_ad_int = np.array([0.0,0.0,0.0,0.0])
        u_ad = np.array([0.0,0.0,0.0,0.0])

        # low-pass filter 1 (negation is added to u_ad_prev to filter the correct signal)
        u_ad_int[0] = lpf1_coefficientThrust1 * (lpf1_prev[0]) + lpf1_coefficientThrust2 * sigma_m_hat[0]
        u_ad_int[1:4] = lpf1_coefficientMoment1 * (lpf1_prev[1:4]) + lpf1_coefficientMoment2 * sigma_m_hat[1:4]

        lpf1_prev = u_ad_int # store the current state

        # coefficients for the second LPF on the moment channel
        lpf2_coefficientMoment1 = math.exp(- ctoffq2Moment * dt)
        lpf2_coefficientMoment2 = 1.0 - lpf2_coefficientMoment1

        # low-pass filter 2 (optional)
        u_ad[0] = u_ad_int[0] # only one filter on the thrust channel
        u_ad[1:4] = lpf2_coefficientMoment1 * lpf2_prev[1:4] + lpf2_coefficientMoment2 * u_ad_int[1:4]

        lpf2_prev = u_ad # store the current state

        u_ad = -u_ad

        # store the values for next iteration (negation is added to u_ad_prev to filter the correct signal)
        u_ad_prev = u_ad

        v_prev = v_now
        omega_prev = omega_now
        R_prev = R
        u_b_prev = np.array([f,M[0],M[1],M[2]])

        controlcmd_L1 = np.array([f,M[0],M[1],M[2]]) + u_ad_prev

        self.din_L1 = (v_hat_prev, omega_hat_prev, R_prev, v_prev, omega_prev,
        u_b_prev, u_ad_prev, sigma_m_hat_prev, sigma_um_hat_prev,
        lpf1_prev, lpf2_prev)

        f_L1 = controlcmd_L1[0]
        M_L1 = controlcmd_L1[1:4]
        return (f_L1, M_L1, sigma_m_hat)

    def geometric_control(self, t, R, W, x, v, d_in):
        """there might be minor bug(s) with this geometric controller, not fixed the 'overshoot' issue at the beginning"""

        (xd, xd_dot, xd_2dot, xd_3dot, xd_4dot, b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot) = d_in
        (ex, ev) = self.position_errors( x, xd, v, xd_dot)

        f = np.dot(self.kx*ex + self.kv*ev + self.m*self.g*self.e3 - self.m*xd_2dot, R.dot(self.e3) )
        W_hat = self.hat(W)

        R_dot = R.dot(W_hat)
        x_2dot = self.g*self.e3 - f*R.dot(self.e3)/self.m
        ex_2dot = x_2dot - xd_2dot

        f_dot = ( self.kx*ev + self.kv*ex_2dot - self.m*xd_3dot).dot(R.dot(self.e3)) + ( self.kx*ex + self.kv*ev + self.m*self.g*self.e3 - self.m*xd_2dot).dot(np.dot(R_dot,self.e3))

        x_3dot = -1/self.m*( f_dot*R + f*R_dot ).dot(self.e3)
        ex_3dot = x_3dot - xd_3dot

        A = -self.kx*ex - self.kv*ev - self.m*self.g*self.e3 + self.m*xd_2dot
        A_dot = -self.kx*ev - self.kv*ex_2dot + self.m*xd_3dot
        A_2dot = -self.kx*ex_2dot - self.kv*ex_3dot + self.m*xd_4dot

        (Rd, Wd, Wd_dot) = self.get_Rc(A, A_dot, A_2dot , b1d, b1d_dot, b1d_ddot)

        (eR, eW) = self.attitude_errors( R, Rd, W, Wd )
        M= -self.kR*eR - self.kW*eW + np.cross(W, self.J.dot(W)) - self.J.dot(W_hat.dot(R.T.dot(Rd.dot(Wd))) - R.T.dot(Rd.dot(Wd_dot)))
        return (f, M, Rd)

    def geometric_control_new(self, t, x, v, R, W):
        """bug-free geometric controller"""

        GeoCtrl_Kpx = 16.*self.m # 4.512
        GeoCtrl_Kpy = 16.*self.m #4.512

        GeoCtrl_Kpz = 16.*self.m
        GeoCtrl_Kvx = 5.6*self.m
        GeoCtrl_Kvy = 5.6*self.m # 0.5
        GeoCtrl_Kvz = 5.6*self.m
        GeoCtrl_KRx = 8.81
        GeoCtrl_KRy = 8.81
        GeoCtrl_KRz = 8.81
        GeoCtrl_KOx = 2.54/10

        GeoCtrl_KOy = 2.54/10 # 0.073
        GeoCtrl_KOz = 2.54/10
    
        GRAVITY_MAGNITUDE = 9.81
        


        # phi = state[6]
        # theta = state[7]
        # psi = state[8]

        zeros2 = [0.0,0.0]
        zeros3 = [0.0,0.0,0.0]

        # targetPos.x = radius * sinf(currentRav bte * netTime)
        # targetPos.y = radius * (1 - cosf(currentRate * netTime))
        # targetPos.z = 1
        targetPos = np.array([2*(1-math.cos(t)), 2*math.sin(t), 1-math.cos(t)])

        # targetVel.x = radius * currentRate * cosf(currentRate * netTime)
        # targetVel.y = radius * currentRate * sinf(currentRate * netTime)
        # targetVel.z = 0
        targetVel = np.array([2*math.sin(t), 2*math.cos(t), math.sin(t)])

        # targetAcc.x = -radius * currentRate * currentRate * sinf(currentRate * netTime)
        # targetAcc.y = radius * currentRate * currentRate * cosf(currentRate * netTime)
        # targetAcc.z = 0
        targetAcc = np.array([2*math.cos(t), -2*math.sin(t), math.cos(t)])


        # targetJerk.x = -radius * powF(currentRate,3) * cosf(currentRate * netTime)
        # targetJerk.y = -radius * powF(currentRate,3) * sinf(currentRate * netTime)
        # targetJerk.z = 0
        targetJerk = np.array([-2*math.sin(t), -2*math.cos(t), -math.sin(t)])


        # targetSnap.x = radius * powF(currentRate,4) * sinf(currentRate * netTime)
        # targetSnap.y = -radius * powF(currentRate,4) * cosf(currentRate * netTime)
        # targetSnap.z = 0
        targetSnap = np.array([-2*math.cos(t), 2*math.sin(t), -math.cos(t)])


        targetYaw = np.array([1.0,0.0]) # represent the orientation vector (Algo 1 in the supplementary)
        targetYaw_dot = np.array(zeros2) # represent derivative of the orientation vector
        targetYaw_ddot = np.array(zeros2)
        # targetYaw = np.array([math.cos(t), math.sin(t)])
        # targetYaw_dot = np.array([-math.sin(t), math.cos(t)])
        # targetYaw_ddot = np.array([-math.cos(t), -math.sin(t)])

        # begin geometric control
        # Position Error (ep)
        statePos = x
        r_error = statePos - targetPos
        # print(r_error)

        # Velocity Error (ev)
        stateVel = v
        v_error = stateVel - targetVel
        # print(v_error)

        target_force = np.array(zeros3)
        target_force[0] = self.m * targetAcc[0] - GeoCtrl_Kpx * r_error[0] - GeoCtrl_Kvx * v_error[0]
        target_force[1] = self.m * targetAcc[1] - GeoCtrl_Kpy * r_error[1] - GeoCtrl_Kvy * v_error[1]
        target_force[2] = self.m * (targetAcc[2] - GRAVITY_MAGNITUDE) - GeoCtrl_Kpz * r_error[2] - GeoCtrl_Kvz * v_error[2]
        # target_force[2] = kg_vehicleMass * (targetAcc[2] + GRAVITY_MAGNITUDE) - GeoCtrl_Kpz * r_error[2] - GeoCtrl_Kvz * v_error[2]

        # r = Rot.from_euler('zyx', [phi, theta, psi], degrees=True) # ? I changed this, since no attitude (w,x,y,z) tuple is available
        # R = r.as_matrix()
        # Rvec = state[6:15]
        # R = Rvec.reshape(3,3)

        z_axis = R[:,2]

        # target thrust [F] (z-positive)
        target_thrust = -np.dot(target_force,z_axis)
        # target_thrust = np.dot(target_force,z_axis)
        # Calculate axis [zB_des] (z-positive)
        z_axis_desired = -target_force/np.linalg.norm(target_force)
        # z_axis_desired = target_force/np.linalg.norm(target_force)

        # [xC_des]
        # x_axis_desired = z_axis_desired x [cos(yaw), sin(yaw), 0]^T, b_int in the supplementary document
        x_c_des = np.array(zeros3)
        x_c_des[0] = targetYaw[0]
        x_c_des[1] = targetYaw[1]
        x_c_des[2] = 0

        x_c_des_dot = np.array(zeros3)
        x_c_des_dot[0] = targetYaw_dot[0]
        x_c_des_dot[1] = targetYaw_dot[1]
        x_c_des_dot[2] = 0

        x_c_des_ddot = np.array(zeros3)
        x_c_des_ddot[0] = targetYaw_ddot[0]
        x_c_des_ddot[1] = targetYaw_ddot[1]
        x_c_des_ddot[2] = 0
        # [yB_des]
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired = y_axis_desired/la.norm(y_axis_desired)

        # [xB_des]
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        # [eR]
        # Slow version
        Rdes = np.empty(shape=(3,3))
        Rdes[:,0] = x_axis_desired
        Rdes[:,1] = y_axis_desired
        Rdes[:,2] = z_axis_desired
        # Matrix3f Rdes(Vector3f(x_axis_desired.x, y_axis_desired.x, z_axis_desired.x),
        #               Vector3f(x_axis_desired.y, y_axis_desired.y, z_axis_desired.y),
        #               Vector3f(x_axis_desired.z, y_axis_desired.z, z_axis_desired.z));

        eRM = (np.matmul(Rdes.transpose(),R) - np.matmul(R.transpose(), Rdes)) / 2

        # Matrix3<T>(const T ax, const T ay, const T az,
        #  const T bx, const T by, const T bz,
        #  const T cx, const T cy, const T cz)
        # eR.x = eRM.c.y;
        # eR.y = eRM.a.z;
        # eR.z = eRM.b.x;
        eR = np.array(zeros3)
        eR = self.vee(eRM)
        # print(eR)
        # eR[0] = eRM[2,1]
        # eR[1] = eRM[0,2]
        # eR[2] = eRM[1,0]

        Omega = W
        # print(Omega)

        #compute Omegad
        a_error = np.array(zeros3) # error on acceleration
        # a_error = [0,0,-GRAVITY_MAGNITUDE] + R[:,2]* target_thrust / kg_vehicleMass - targetAcc
        a_error = [0,0,GRAVITY_MAGNITUDE] - R[:,2]* target_thrust / self.m - targetAcc
        # ? turn GRAVITY_MAGNITUDE to - GRAVITY_MAGNITUDE
        # ? turn - R[:,2]* target_thrust / kg_vehicleMass to + R[:,2]* target_thrust / kg_vehicleMass

        target_force_dot = np.array(zeros3) # derivative of target_force
        target_force_dot[0] = - GeoCtrl_Kpx * v_error[0] - GeoCtrl_Kvx * a_error[0] + self.m * targetJerk[0]
        target_force_dot[1] = - GeoCtrl_Kpy * v_error[1] - GeoCtrl_Kvy * a_error[1] + self.m * targetJerk[1]
        target_force_dot[2] = - GeoCtrl_Kpz * v_error[2] - GeoCtrl_Kvz * a_error[2] + self.m * targetJerk[2]

        b3_dot = np.matmul(np.matmul(R, self.hat(Omega)),np.array([0,0,1])) #derivative of (Re3) in eq (2)
        target_thrust_dot = - np.dot(target_force_dot,R[:,2]) - np.dot(target_force, b3_dot)
        # target_thrust_dot = + np.dot(target_force_dot,R[:,2]) + np.dot(target_force, b3_dot)
        # ? turn the RHS from - to +

        j_error = np.array(zeros3) # error on jerk
        # j_error = np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass + b3_dot * target_thrust / kg_vehicleMass - targetJerk
        j_error = -np.dot(R[:,2], target_thrust_dot) / self.m - b3_dot * target_thrust / self.m - targetJerk
        # ? turn - np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass to np.dot(R[:,2], target_thrust_dot) / kg_vehicleMass
        # ? turn - b3_dot * target_thrust / kg_vehicleMass to + b3_dot * target_thrust / kg_vehicleMass

        target_force_ddot = np.array(zeros3) # derivative of target_force_dot
        target_force_ddot[0] = - GeoCtrl_Kpx * a_error[0] - GeoCtrl_Kvx * j_error[0] + self.m * targetSnap[0]
        target_force_ddot[1] = - GeoCtrl_Kpy * a_error[1] - GeoCtrl_Kvy * j_error[1] + self.m * targetSnap[1]
        target_force_ddot[2] = - GeoCtrl_Kpz * a_error[2] - GeoCtrl_Kvz * j_error[2] + self.m * targetSnap[2]


        b3cCollection = np.array([zeros3,zeros3,zeros3]) # collection of three three-dimensional vectors b3c, b3c_dot, b3c_ddot
        b3cCollection = self.unit_vec(-target_force, -target_force_dot, -target_force_ddot) # unit_vec function is from geometric controller's git repo: https://github.com/fdcl-gwu/uav_geometric_control/blob/master/matlab/aux_functions/deriv_unit_vector.m
        
        b3c = np.array(zeros3)
        b3c_dot = np.array(zeros3)
        b3c_ddot = np.array(zeros3)

        b3c[0] = b3cCollection[0]
        b3c[1] = b3cCollection[1]
        b3c[2] = b3cCollection[2]

        b3c_dot[0] = b3cCollection[3]
        b3c_dot[1] = b3cCollection[4]
        b3c_dot[2] = b3cCollection[5]

        b3c_ddot[0] = b3cCollection[6]
        b3c_ddot[1] = b3cCollection[7]
        b3c_ddot[2] = b3cCollection[8]

        """some changes start here"""
        A2 = - np.matmul(self.hat(x_c_des), b3c)
        A2_dot = - np.matmul(self.hat(x_c_des_dot),b3c) - np.matmul(self.hat(x_c_des), b3c_dot)
        A2_ddot = - np.matmul(self.hat(x_c_des_ddot), b3c) - np.matmul(self.hat(x_c_des_dot), b3c_dot) * 2 - np.matmul(self.hat(x_c_des), b3c_ddot)

        b2cCollection = np.array([zeros3,zeros3,zeros3]) # collection of three three-dimensional vectors b2c, b2c_dot, b2c_ddot
        b2cCollection = self.unit_vec(A2, A2_dot, A2_ddot) # unit_vec function is from geometric controller's git repo: https://github.com/fdcl-gwu/uav_geometric_control/blob/master/matlab/aux_functions/deriv_unit_vector.m

        b2c = np.array(zeros3)
        b2c_dot = np.array(zeros3)
        b2c_ddot = np.array(zeros3)

        b2c[0] = b2cCollection[0]
        b2c[1] = b2cCollection[1]
        b2c[2] = b2cCollection[2]

        b2c_dot[0] = b2cCollection[3]
        b2c_dot[1] = b2cCollection[4]
        b2c_dot[2] = b2cCollection[5]

        b2c_ddot[0] = b2cCollection[6]
        b2c_ddot[1] = b2cCollection[7]
        b2c_ddot[2] = b2cCollection[8]

        b1c_dot = np.matmul(self.hat(b2c_dot), b3c) + np.matmul(self.hat(b2c), b3c_dot)
        b1c_ddot = np.matmul(self.hat(b2c_ddot),b3c) + np.matmul(self.hat(b2c_dot), b3c_dot) * 2 + np.matmul(self.hat(b2c), b3c_ddot)

        Rd_dot = np.empty(shape=(3,3)) # derivative of Rdes
        Rd_ddot = np.empty(shape=(3,3)) # derivative of Rd_dot

        # print("-----")
        # print(b1c_dot)
        # print(b2c_dot)
        # print(b3c_dot)
        Rd_dot[0,:] = b1c_dot
        Rd_dot[1,:] = b2c_dot
        Rd_dot[2,:] = b3c_dot
        # print(Rd_dot)
        Rd_dot = Rd_dot.transpose()
        # print(Rd_dot)

        Rd_ddot[0,:] = b1c_ddot
        Rd_ddot[1,:] = b2c_ddot
        Rd_ddot[2,:] = b3c_ddot
        Rd_ddot = Rd_ddot.transpose()

        Omegad = self.vee(np.matmul(Rdes.transpose(), Rd_dot))
        # print(t)
        # print(Omegad)
        Omegad_dot = self.vee(np.matmul(Rdes.transpose(), Rd_ddot) - np.matmul(self.hat(Omegad), self.hat(Omegad)))

        # these two lines are remedy which is not supposed to exist in the code. There might be an error in the code above.
        # Omegad[1] = -Omegad[1]
        # Omegad_dot[1] = -Omegad_dot[1]
        # temporarily use zero Omegad
        Omegad_check = np.matmul(np.matmul(R.transpose(), Rdes), Omegad)
        ew = Omega -  np.matmul(np.matmul(R.transpose(), Rdes), Omegad)
        # Moment: simple version
        M = np.array(zeros3)
        M[0] = -GeoCtrl_KRx * eR[0] - GeoCtrl_KOx * ew[0]
        M[1] = -GeoCtrl_KRy * eR[1] - GeoCtrl_KOy * ew[1]
        M[2] = -GeoCtrl_KRz * eR[2] - GeoCtrl_KOz * ew[2]
        # Moment: full version
        # M = M - np.matmul(J, (np.matmul(hatOperator(Omega), np.matmul(R.transpose(),np.matmul(Rdes, Omegad))) - np.matmul(R.transpose(), np.matmul(Rdes, Omegad_dot))))
        # LS: check version
        M = M - np.matmul(self.J, (- np.matmul(R.transpose(), np.matmul(Rdes, Omegad_dot))))
        # ShengC: an additive term is the following
        # momentAdd = np.cross(Omegad, (np.matmul(J, Omegad))) # J is the inertia matrix
        # LS: check version
        momentAdd = np.cross(Omegad_check, (np.matmul(self.J, Omegad_check))) # J is the inertia matrix
        M = M +  momentAdd

        thrustMomentCmd = np.array([0.0,0.0,0.0,0.0])
        thrustMomentCmd[0] = target_thrust
        thrustMomentCmd[1] = M[0]
        thrustMomentCmd[2] = M[1]
        thrustMomentCmd[3] = M[2]

        # u = np.array([0.0,0.0,0.0,0.0])
        # motorAssignMatrix = np.array([[1, 1, 1, 1],
        #                               [-0.1, 0.1,-0.1, 0.1],
        #                               [-0.075, 0.075, 0.075, -0.075],
        #                               [-0.022, -0.022, 0.022, 0.022]])
        # u = np.matmul(LA.inv(motorAssignMatrix),thrustMomentCmd) # no need to re-assign to every motor in simulation
        u = thrustMomentCmd
        return (target_thrust, M, Rdes)




    # def dynamic(t, state):
    #     x, y = state
    #     x = float(x)
    #     y = float(y)
    #     x_dot = y
    #     y_dot = (1-x**2)*y - x
    #     return [x_dot, y_dot]
    def dynamic_mode1(self, t, state):
        # print(state)
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = state[18:19] # mass for simulation
        t = state[19:]
        self.use_l1ac = False

        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        (f, M, Rd) = self.geometric_control_new(t, x, v, R, W) # computes the control commands
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print(f_L1)
        # print('------------------------')
        # print(M_L1)
        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)

        

        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance


        x_dot = v
        v_dot = self.g*self.e3 - f/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
        mass_dot = np.zeros(1,)
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))

        return X_dot
    def dynamic_mode2(self, t, state):
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = self.m
        # print(mass)
        t = state[19:]
        self.use_l1ac = True

        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        (f, M, Rd) = self.geometric_control_new(t, x, v, R, W)
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print(f_L1)
        # print('------------------------')
        # print(M_L1)
 


        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)

        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance

        x_dot = v
        v_dot = self.g*self.e3 - f/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
        mass_dot = np.zeros(1,)
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))

        return X_dot

    def dynamic_mode3(self, t, state):
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = self.m*3
        # print(mass)
        t = state[19:]
        self.use_l1ac = True

        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        (f, M, Rd) = self.geometric_control_new(t, x, v, R, W)
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print(f_L1)
        # print('------------------------')
        # print(M_L1)
 


        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)

        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance

        x_dot = v
        v_dot = self.g*self.e3 - f/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
        mass_dot = np.zeros(1,)
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))

        return X_dot

    def dynamic_mode4(self, t, state):
        # print(state)
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = state[18:19] # mass for simulation
        self.use_l1ac = False
        t = state[19:]

        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        (f, M, Rd) = self.geometric_control_new(t, x, v, R, W) # computes the control commands
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print(f_L1)
        # print('------------------------')
        # print(M_L1)
        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)

        

        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance


        x_dot = v
        v_dot = self.g*self.e3 - f/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
        mass_dot = np.zeros(1,)
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))

        return X_dot

    def dynamic_mode5(self, t, state):
        # print(state)
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = state[18:19] # mass for simulation
        self.use_l1ac = True
        t = state[19:]

        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        (f, M, Rd) = self.geometric_control_new(t, x, v, R, W) # computes the control commands
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print(f_L1)
        # print('------------------------')
        # print(M_L1)
        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)

        

        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance


        x_dot = v
        v_dot = self.g*self.e3 - f/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
        mass_dot = np.zeros(1,)
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))

        return X_dot
    
    def dynamic_mode6(self, t, state):
        # print(state)
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = state[18:19] # mass parameter

        
        t = state[19:]
        if (t > 60):
            self.use_l1ac = True
        else:
            self.use_l1ac = False
        # if (t >= 2 and t < 4):
        #     mass = 1.3*state[18:19] # mass for simulation, changed to a new parameter after 5 seconds
        # elif (t >= 6):
        #     mass = 0.65*state[18:19]
        # else:
        #     mass = state[18:19] # mass for simulation




        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        (f, M, Rd) = self.geometric_control_new(t, x, v, R, W) # computes the control commands
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print(f_L1)
        # print('------------------------')
        # print(M_L1)
        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)



        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance


        x_dot = v
        v_dot = self.g*self.e3 - f/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
        mass_dot = np.zeros(1,)
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))

        return X_dot

      
    def dynamic_mode7(self, t, state):
        # print(state)
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = state[18:19] # mass parameter
        self.tau = 0.0 # no time delay
        self.use_l1ac = False
        t = state[19:]
        # if (t >= 2 and t < 4):
        #     mass = 1.3*state[18:19] # mass for simulation, changed to a new parameter after 5 seconds
        # elif (t >= 6):
        #     mass = 0.65*state[18:19]
        # else:
        #     mass = state[18:19] # mass for simulation




        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        (f, M, Rd) = self.geometric_control_new(t, x, v, R, W) # computes the control commands
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print(f_L1)
        # print('------------------------')
        # print(M_L1)
        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)



        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance


        x_dot = v
        v_dot = self.g*self.e3 - f/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M - np.cross(W, np.dot(self.J, W)))
        mass_dot = np.array([-3*math.sin(t)*math.cos(t)-64*math.sin(8*t)*math.cos(8*t)**3]) # drama_change and drama_change_aggres
        # mass_dot = -0.05*t-4*np.array([math.sin(2*t)]) - 0.25 # undrama_change
        # mass_dot = np.zeros(1,)
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))

        return X_dot

    def dynamic_mode8(self, t, state):
        # print(self.itr)
        # print(state)
        x = state[:3] # position
        v = state[3:6] # velocity
        R = np.reshape(state[6:15], (3,3)) # rotation matrix from body to inertial
        W = state[15:18] # angular velocity
        mass = state[18:19] # mass parameter

        self.use_l1ac = True
        self.tau = 0.15 # injected time delay
        curr_t = state[19:]
        # print(curr_t)
        # if (t >= 2 and t < 4):
        #     mass = 1.3*state[18:19] # mass for simulation, changed to a new parameter after 5 seconds
        # elif (t >= 6):
        #     mass = 0.65*state[18:19]
        # else:
        #     mass = state[18:19] # mass for simulation




        # xd = np.array([2*(1-np.cos(t)), 2*np.sin(t), 1+np.sin(t)]) #desired position
        # xd_dot = np.array([2*np.sin(t), 2*np.cos(t), np.cos(t)]) #desired velocity
        # xd_ddot = np.array([2*np.cos(t), -2*np.sin(t), -np.sin(t)]) #desired acceleration
        # xd_dddot= np.array([-2*np.sin(t), -2*np.cos(t), -np.cos(t)]) #desired jerk
        # xd_ddddot = np.array([-2*np.cos(t), 2*np.sin(t), np.sin(t)]) #desired snap

        # b1d = np.array([1., 0., 0.]) #desired orientation vector (bint in Algorithm 1)
        # b1d_dot=np.array([0., 0., 0.]) #(bint_dot)
        # b1d_ddot=np.array([0., 0., 0.]) #(bint_ddot)

        # Rd = np.eye(3)
        # Wd = np.array([0.,0.,0.])
        # Wd_dot = np.array([0.,0.,0.])

        f = np.array([0])
        M = np.array([0,0,0])

        # d_in = (xd, xd_dot, xd_ddot, xd_dddot, xd_ddddot,
        #             b1d, b1d_dot, b1d_ddot, Rd, Wd, Wd_dot)
        (f, M, Rd) = self.geometric_control_new(curr_t, x, v, R, W) # computes the control commands
        # (f, M, Rd) = self.geometric_control(t, R, W, x, v, d_in) # computes the control commands
        # print(f)
        (f_L1, M_L1, sigma_m_hat) = self.L1_AC(R, W, x, v, f, M)
        # print('----this is the thrust comp w/ L1.------')
        # print(f_L1)
        # print('----this is the thrust comp wo L1.------')
        # print(f)
        # print('------------------------')
        # print(M_L1)
        """ Uncertain dynamics (use geometric + L1 adaptive control) """
        if self.use_uncertainty:
            sigma_m_thrust = 1.4*math.sin(0.5*(t-1.5)) + 3.7*math.sin(0.75*(t-2.5))
            sigma_m_roll = 0.9*math.sin(0.75*(t-1.5))
            sigma_m_pitch = 0.85*(math.sin(t-0.5) + math.sin(0.5*(t-3.5)))
        else:
            sigma_m_thrust = 0.0
            sigma_m_roll = 0.0
            sigma_m_pitch = 0.0 # tuning knobs (uncertainty/disturbances on the control channel)



        if self.use_l1ac:
            f = f_L1 + sigma_m_thrust # L1 + geometric control + disturbance
            M = M_L1 + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance
        else:
            f = f + sigma_m_thrust # L1 + geometric control + disturbance
            M = M + np.array([sigma_m_roll, sigma_m_pitch, 0.0]) # geometric control + disturbance

        # self.shift_i-p-ndex_curr = int(math.floor((t-self.tau)/0.05))

        self.f_seq.append([f])
        self.M_seq.append(M)

        curr_length = int(len(self.f_seq))
        # print('----current length of sequence----------')
        # print(curr_length)
        # print(curr_t)
        # curr_t = t
        if self.itr_time > self.tau + 0.0:
            index = curr_length  - self.itr 
            # print('---- this is the index after shift back -----')
            # print(index)
            f_apply = self.f_seq[index]
            # print(self.f_seq)
            M_apply = self.M_seq[index]
            # print(self.M_seq)
        else:
            f_apply = 0.0
            M_apply = np.array([0.0,0.0,0.0])
            self.itr = self.itr+1
            # print(self.itr)


        x_dot = v
        v_dot = self.g*self.e3 - f_apply/mass*R.dot(self.e3)
        R_dot = np.dot(R, self.hat(W))
        W_dot = np.dot(la.inv(self.J), M_apply - np.cross(W, np.dot(self.J, W)))
        # mass_dot = 2*np.array([math.cos(t)])
        # mass_dot = np.zeros(1,)
        mass_dot = -0.05*t-4*np.array([math.sin(2*t)]) - 0.25 # test the delay margin
        time_dot = np.ones(1,)
        X_dot = np.concatenate((x_dot, v_dot, R_dot.flatten(), W_dot, mass_dot, time_dot))
        
        return X_dot




    """ Some utility functions """
    def position_errors( self, x, xd, v, vd):
        ex = x - xd
        ev = v - vd
        return (ex, ev)

    def attitude_errors( self, R, Rd, W, Wd ):
        eR = 0.5*self.vee(Rd.T.dot(R) - R.T.dot(Rd))
        eW = W - R.T.dot(Rd.dot(Wd))
        return (eR, eW)

    def vee(self, M):
        return np.array([M[2,1], M[0,2], M[1,0]])

    def unit_vec(self, q, q_dot, q_ddot):
        """unit vector function provides three different normalization method/scale to the three three-element vectors"""

        collection = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        nq = la.norm(q)
        u = q / nq
        u_dot = q_dot / nq - q * np.dot(q, q_dot) / pow(nq,3)

        u_ddot = q_ddot / nq - q_dot / pow(nq,3) * (2 * np.dot(q, q_dot)) - q / pow(nq,3) * (np.dot(q_dot, q_dot) + np.dot(q, q_ddot)) + 3 * q / pow(nq,5) * pow(np.dot(q, q_dot),2)
        # breakpoint()
        collection[0 : 3] = u
        collection[3 : 6] = u_dot
        collection[6 : 9] = u_ddot

        # breakpoint()

        return collection

    def hat(self, x):
        hat_x = [0, -x[2], x[1],
                x[2], 0, -x[0],
                -x[1], x[0], 0]
        return np.reshape(hat_x,(3,3))

    def get_Rc(self, A, A_dot, A_2dot, b1d, b1d_dot, b1d_ddot):

        norm_A = la.norm(A)
        b3c = - A/norm_A
        b3c_dot = - A_dot/norm_A + ( np.dot(A, A_dot)*A )/norm_A**3 #eq (4)
        b3c_2dot = - A_2dot/norm_A + ( 2*np.dot(A*A_dot,A_dot) )/norm_A**3 + np.dot( A_dot* A_dot + A*A_2dot ,A)/norm_A**3 - 3*np.dot((A*A_dot)**2,A)/norm_A**5 #eq (7)

        b_ = np.cross(b3c, b1d)
        b_norm = la.norm(b_)
        b_dot = np.cross(b3c_dot, b1d) + np.cross(b3c, b1d_dot)
        b_2dot = np.cross(b3c_2dot, b1d) + 2*np.cross(b3c_dot, b1d_dot) + np.cross(b3c, b1d_ddot)

        b1c =  - np.cross( b3c, b_ )/b_norm
        """b1c = b2c x b3c, equivalently, b2c = b3c x b1c"""
        b1c_dot = ( np.cross(b3c_dot, b_) - np.cross(b3c, b_dot) )/b_norm - np.cross(b3c, b_)*(b_dot* b_)/b_norm**3

        # intermediate steps to calculate b1c_2dot
        m_1 = ( np.cross(b3c_2dot, b_) + 2*np.cross(b3c_dot, b_dot) + np.cross(b3c, b_2dot) )/b_norm
        m_2 = ( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )*np.dot(b_dot, b_)/b_norm**3
        m_dot = m_1 - m_2
        n_1 = np.cross(b3c, b_)*np.dot(b_dot, b_)
        n_1dot = ( np.cross(b3c_dot, b_) + np.cross(b3c, b_dot) )*np.dot(b_dot, b_) + np.cross(b3c, b_)*( np.dot(b_2dot, b_)+np.dot(b_dot, b_dot) )
        n_dot = n_1dot/b_norm**3 - 3*n_1*np.dot(b_dot, b_)/b_norm**5
        b1c_2dot = (-m_dot + n_dot)

        Rc = np.reshape([b1c, np.cross(b3c, b1c), b3c],(3,3)).T
        Rc_dot = np.reshape([b1c_dot, ( np.cross(b3c_dot, b1c) + np.cross(b3c, b1c_dot) ), b3c_dot],(3,3)).T
        Rc_2dot = np.reshape( [b1c_2dot, ( np.cross(b3c_2dot, b1c) + np.cross(b3c_dot, b1c_dot) + np.cross(b3c_dot, b1c_dot) + np.cross(b3c, b1c_2dot) ), b3c_2dot],(3,3)).T
        Wc = self.vee(Rc.T.dot(Rc_dot))
        # Wc_dot= self.vee( Rc_dot.T.dot(Rc_dot) + Rc.T.dot(Rc_2dot))
        Wc_hat = self.hat(Wc)
        Wc_dot= self.vee( Rc.T.dot(Rc_2dot) - Wc_hat.T.dot(Wc_hat)) # check version
        return (Rc, Wc, Wc_dot)

    def action_handler(self, mode):
        if mode == 'Mode1':
            return ode(self.dynamic_mode1)
        elif mode == 'Mode2':
            return ode(self.dynamic_mode2)
        elif mode == 'Mode3':
            return ode(self.dynamic_mode3)
        elif mode == 'Mode4':
            return ode(self.dynamic_mode4)
        elif mode == 'Mode5':
            return ode(self.dynamic_mode5)
        elif mode == 'Mode6':
            return ode(self.dynamic_mode6)
        elif mode == 'Mode7':
            return ode(self.dynamic_mode7)
        elif mode == 'Mode8':
            # DDE = jitcdde(self.f)
            # DDE.add_past_point(-self.tau, self.start_state, np.zeros(20))
            # DDE.add_past_point(0.0   , self.start_state, np.zeros(20)) 
            # return DDE
            return ode(self.dynamic_mode8)
        else:
            raise ValueError
    
    # def initial_history_func_0(self, t):
    #     return 0

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:

        print(mode[0])
        # print(time_step)
        # print('-----------time step shown above------------')
        self.f_seq = []
        self.M_seq = []
        self.itr = 0
        self.itr_time = 0

        self.v_hat_prev = np.array([0.0, 0.0, 0.0])
        self.omega_hat_prev = np.array([0.0, 0.0, 0.0])
        self.R_prev = np.zeros((9,)).reshape(3,3)
        self.v_prev = np.array([0.0,0.0,0.0])
        self.omega_prev = np.array([0.0,0.0,0.0])

        self.u_b_prev = np.array([0.0,0.0,0.0,0.0])
        self.u_ad_prev = np.array([0.0,0.0,0.0,0.0])
        self.sigma_m_hat_prev = np.array([0.0,0.0,0.0,0.0])
        self.sigma_um_hat_prev = np.array([0.0,0.0])
        self.lpf1_prev = np.array([0.0,0.0,0.0,0.0])
        self.lpf2_prev = np.array([0.0,0.0,0.0,0.0])

        self.din_L1 = (self.v_hat_prev, self.omega_hat_prev, self.R_prev, self.v_prev, self.omega_prev,
                       self.u_b_prev, self.u_ad_prev, self.sigma_m_hat_prev, self.sigma_um_hat_prev, 
                       self.lpf1_prev, self.lpf2_prev)

        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        print('---same call of tc simulate----')
        print(self.checker)
        # print(t)


        init = initialCondition
        trace = [[0]+init]
        r = self.action_handler(mode[0])
        # print('---length of time duration----')
        # r.set_integrator('dopri5', max_step=0.1).set_initial_value(init)
        r.set_integrator('dopri5', nsteps=6000).set_initial_value(init)  
        # print(r.t)
        # r.set_integrator('zvode', method='bdf').set_initial_value(init)
        for i in range(len(t)):
            # if i > 100:
            #     print("stop")
            # print('---same call of tc simulate---')
            # print('---within the inner loop-----')
            print(r.t)
            self.itr_time = r.t
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
            # print(len(self.M_seq))
        # print(np.array(trace))
        self.checker = self.checker + 1
        return np.array(trace)

    # def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:

    #     # print(mode[0])
    #     time_bound = float(time_bound)
    #     number_points = int(np.ceil(time_bound/time_step))
    #     t = [round(i*time_step, 10) for i in range(0, number_points)]

    #     init = initialCondition
    #     trace = [[0]+init]
    #     r = self.action_handler(mode[0])
    #     # print('---length of time duration----')
    #     # print(len(t))
    #     # r.set_integrator('dopri5', nsteps=4000).set_initial_value(init) 
    #     times = np.arange(0, time_bound, time_step)
    #     i = 0
    #     for tt in times:
    #         # if i > 100:
    #         #     print("stop")
    #         # print('---same call of tc simulate---')
    #         res: np.ndarray = r.integrate(tt)
    #         init = res.flatten().tolist()
    #         trace.append([t[i] + time_step] + init)
    #         i = i+1
    #     # print(np.array(trace))
    #     return np.array(trace)





if __name__ == '__main__':
    aquad = quadrotor_agent('agent1')

    # Initialize simulation states
    trace = aquad.TC_Simulate(['none'], [1.25, 2.25], 7, 0.05)
    print(trace)

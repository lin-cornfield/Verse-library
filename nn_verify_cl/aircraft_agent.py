# aircraft agent.
from typing import Tuple, List, NamedTuple

import numpy as np
from scipy.integrate import ode
import torch


import numpy.linalg as la
import random

# from tutorial_utils import drone_params
from verse import BaseAgent
from verse import LaneMap
from verse.map.lane_map_3d import LaneMap_3d
from verse.analysis.utils import wrap_to_pi
from verse.analysis.analysis_tree import TraceType



from functools import lru_cache
import time
import math
import argparse

from scipy import ndimage
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib import patches, animation
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.lines import Line2D
from verse.parser.parser import ControllerIR

import onnxruntime as ort
from numba import njit
import functools as ft

import ipdb
import jax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
import tqdm # for progress bar
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.functional.aero_prop_new import FuncAeroProp
from jax_guam.functional.lc_control import LCControl, LCControlState
from jax_guam.functional.surf_engine import SurfEngine, SurfEngineState
from jax_guam.functional.vehicle_eom_simple import VehicleEOMSimple
from jax_guam.guam_types import AircraftState, AircraftStateVec, EnvData, PwrCmd, RefInputs, Failure_Engines
from jax_guam.subsystems.environment.environment import Environment
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import *
from jax_guam.subsystems.vehicle_model_ref.power_system import power_system
from jax_guam.utils.ode import ode3
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger



class AircraftAgent(BaseAgent):
    def __init__(
        self,
        id,
        code=None,
        file_name=None,
        initial_state=None,
        initial_mode=None
    ):
        super().__init__(
            id, code, file_name, initial_state=initial_state, initial_mode=initial_mode
        )
        self._environment = Environment()

        self.controller = LCControl()
        self.veh_eom = VehicleEOMSimple()
        self.surf_eng = SurfEngine()
        self.aero_prop = FuncAeroProp()
        self.failure = Failure_Engines(
            F_Fail_Initiate=jnp.array([0,0,0,0,0,0,0,0,0]), #jnp.zeros(9)
            F_Hold_Last=jnp.zeros(9),
            F_Pre_Scale=jnp.ones(9),
            F_Post_Scale=jnp.ones(9),
            F_Pos_Bias=jnp.zeros(9),
            F_Pos_Scale=jnp.ones(9),
            F_Up_Plim=jnp.zeros(9) + jnp.inf,
            F_Lwr_Plim=jnp.zeros(9) - jnp.inf,
            F_Rate_Bias=jnp.zeros(9),
            F_Rate_Scale=jnp.ones(9),
            F_Up_Rlim=jnp.zeros(9) + jnp.inf,
            F_Lwr_Rlim=jnp.zeros(9) - jnp.inf,
            F_Accel_Bias=jnp.zeros(9),
            F_Accel_Scale=jnp.ones(9),
            F_Up_Alim=jnp.zeros(9) + jnp.inf,
            F_Lwr_Alim=jnp.zeros(9) - jnp.inf,
            F_Gen_Sig_Sel=jnp.zeros(15),
        )

        self.dt = 0.2
        
        self.decision_logic = ControllerIR.empty()
        
    @property
    def env_data(self) -> EnvData:
        return self._environment.Env
    
    def deriv(self, state: GuamState, ref_inputs: RefInputs) -> GuamState:
        ref_inputs.assert_shapes()

        sensor, aeroprop_body_data, alt_msl = self.veh_eom.get_sensor_aeroprop_altmsl(state.aircraft)
        atmosphere = self._environment.get_env_atmosphere(alt_msl)
        env_data = self.env_data._replace(Atmosphere=atmosphere)
        control, cache = self.controller.get_control(state.controller, sensor, ref_inputs)

        d_state_controller = self.controller.state_deriv(cache)
        pwr_cmd = PwrCmd(CtrlSurfacePwr=control.Cmd.CtrlSurfacePwr, EnginePwr=control.Cmd.EnginePwr)
        power = power_system(pwr_cmd)

        surf_act, prop_act = self.surf_eng.get_surf_prop_act(state.surf_eng, control.Cmd, power, self.failure)
        d_state_surf_eng = self.surf_eng.surf_engine_state_deriv(control.Cmd, state.surf_eng)

        fm = self.aero_prop.aero_prop(prop_act, surf_act, env_data, aeroprop_body_data)
        fm_total = self.veh_eom.get_fm_with_gravity(state.aircraft, fm)
        d_state_aircraft = self.veh_eom.state_deriv(fm_total, state.aircraft)

        return GuamState(d_state_controller, d_state_aircraft, d_state_surf_eng)
    
    def step(self, dt: float, state: GuamState, ref_inputs: RefInputs) -> GuamState:

        deriv_fn = ft.partial(self.deriv, ref_inputs=ref_inputs)
        state_new = ode3(deriv_fn, dt, state)
        # We also need to clip the state.
        state_new = state_new._replace(surf_eng=self.surf_eng.clip_state(state_new.surf_eng))
        return state_new
    
    def array2GuamState(self, state_arr):
        # print("---for debug purpose----")
        # print(state_arr)
        e_long = state_arr[0:3]
        e_lat = state_arr[3:6]
        aircraft_state = np.array(state_arr[6:19])
        surf_state = state_arr[19:24]
        random_attr = state_arr[-1]
        
        lc_control_state = LCControlState(int_e_long=np.array(e_long), int_e_lat=np.array(e_lat))
        surf_engine_state = SurfEngineState(ctrl_surf_state=np.array(surf_state))
        # Aircraft_state = AircraftState(np.array(aircraft_state[0:3]), np.array(aircraft_state[3:6]), np.array(aircraft_state[6:9]), np.array(aircraft_state[9:13]))
        
        # print("----for debug purpose---")
        # print(test)
        return GuamState(
            controller=lc_control_state,
            aircraft=aircraft_state,
            surf_eng=surf_engine_state
        )
        
    def GuamState2array(self, state_guam, index):
    
        int_e_long = state_guam.controller.int_e_long
        int_e_lat = state_guam.controller.int_e_lat
        aircraft = state_guam.aircraft
        ctrl_surf_state = state_guam.surf_eng.ctrl_surf_state
        index= np.array([index])
        
        
        return np.concatenate((int_e_long, int_e_lat, aircraft, ctrl_surf_state, index)).tolist()
    
    def cmd2ref(self, curr_time, time_bound, init_state, curr_state, cmd):
        if cmd == 0:
            return lift_cruise_reference_inputs_coc(curr_time, time_bound, init_state)
            # return lift_cruise_reference_inputs_coc(self.dt, state)
        else:
            # return None
            return lift_cruise_reference_inputs_turn_right(curr_time, time_bound, init_state, cmd)
            # return lift_cruise_reference_inputs_turn_random(curr_time, time_bound, init_state, cmd_list)


    def action_handler(self, mode: str, state, lane_map: LaneMap) -> int:
        # x1, y1, theta1, _ = state
        ego_mode = mode[0]
        a1=0
        # print(ego_mode)
        if ego_mode == "Coc":
            a1= 0
        elif ego_mode == "Weak_left":
            a1= 1
        elif ego_mode == "Weak_right":
            a1= 2
        elif ego_mode == "Strong_left":
            a1= 3
        elif ego_mode == "Strong_right":
            a1= 4
        else:
            raise ValueError(f"Invalid mode: {ego_mode}")
        return a1
    
    

    def TC_simulate(
        self, mode: str, init, time_bound, time_step, lane_map: LaneMap = None
    ) -> np.ndarray:
        
        jax_use_double()
        # time_bound = float(time_bound)
        T = int(np.ceil(time_bound/self.dt))
        # print(time_bound, self.dt, T)
        
        # num_points = int(np.ceil(time_bound / time_step))
        # trace = np.zeros((num_points + 1, 1 + len(init)))
        # trace[1:, 0] = [round(i * time_step, 10) for i in range(num_points)]
        # trace[0, 1:] = init
        state_arr = init
        state = self.array2GuamState(state_arr)
        initGuamState = state
        # print("---for testing purpose----")
        # print(init)
        dt_acas=1.0
        """ determine whether to apply advisories to one agent """
        ''' ego vehicle: initial_x < 0; intruder vehicle: initial_x > 0 '''
        # if init[12] < 1e-6:
        #     decisions = np.load("test.npy")
        #     decisions=decisions.tolist()
        #     # decisions = []
        #     # for _ in range(T):
        #     #     decisions.append(random.randint(0,4))
        #     # decisions = decisions.tolist()     
        # else:
        #     decisions =[0]*T
        # decisions = np.load("test.npy") # working example 1 of ACAS Xu advisory
        decisions = np.load("test1.npy") # working example 2 of ACAS Xu advisory
        decisions = decisions.tolist()
        # decisions =[0]*T    
        trace = [[0]+state_arr]
        # time_elapse_mats = init_time_elapse_mats(dt_acas)
        cmd_list = []
        # length = int(time_bound / 0.01) + 1
        length = len(decisions)
        cmd_list = [0]*length
        
        # for i in range(length):
        #     if i < length/3:
        #         cmd_list.append(1)
        #     elif i < 7*length/8:
        #         cmd_list.append(3)
        #     else:
        #         cmd_list.append(0)
        
        
        # spl_vel_bIc, spl_pos_bii = initialize_reference_inputs(time_bound, initGuamState, cmd_list)
        for kk in range(T):
            # print(mode)
            cmd = decisions[kk]
            curr_t = (kk) * self.dt
            # print(cmd)
            # ref_input = self.cmd2ref(cmd, state)
            # ref_input = self.cmd2ref(curr_t, time_bound, initGuamState, state, cmd) # for the single command for the horizon
            if init[-1] < 0: # intruder vehicle
                """ correspond to the control of intruder """
                # ref_input = lift_cruise_reference_inputs_turn_right(curr_t, time_bound, initGuamState, 0)
                ref_input = lift_cruise_reference_inputs_turn_random(self.dt, curr_t, time_bound, initGuamState, cmd_list)
            else: # ownship vehicle
                """ correspond to the control of ego vehicle """
                # ref_input = lift_cruise_reference_inputs_turn_right(curr_t, time_bound, initGuamState, 0)
                ref_input = lift_cruise_reference_inputs_turn_random(self.dt, curr_t, time_bound, initGuamState, decisions)
            state = self.step(self.dt, state, ref_input)
            # print(vec[1])
            state_arr = self.GuamState2array(state, kk+1)
            trace.append([curr_t + self.dt] + state_arr)
        return np.array(trace)

if __name__ == '__main__':
    aguam = AircraftAgent('agent1')

    # Initialize simulation states
    trace = aguam.TC_Simulate(['none'], [1.25, 2.25], 7, 0.05) # this line does not need to be accurate
    print(trace)
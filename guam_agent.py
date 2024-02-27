# Example agent.
from typing import Tuple, List, NamedTuple

import numpy as np
import math
from scipy.integrate import ode
import numpy.linalg as la

from verse import BaseAgent
from verse.map import LaneMap

import functools as ft

import ipdb
import jax
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import tqdm # for progress bar
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.functional.aero_prop_new import FuncAeroProp
from jax_guam.functional.lc_control import LCControl, LCControlState
from jax_guam.functional.surf_engine import SurfEngine, SurfEngineState
from jax_guam.functional.vehicle_eom_simple import VehicleEOMSimple
from jax_guam.guam_types import AircraftState, AircraftStateVec, EnvData, PwrCmd, RefInputs
from jax_guam.subsystems.environment.environment import Environment
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs
from jax_guam.subsystems.vehicle_model_ref.power_system import power_system
from jax_guam.utils.ode import ode3
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format
from loguru import logger



class guam_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        # TODO: some initialization of the GUAM model (parameters, etc)
        
        super().__init__(id, code, file_name)
        self._environment = Environment()

        self.controller = LCControl()
        self.veh_eom = VehicleEOMSimple()
        self.surf_eng = SurfEngine()
        self.aero_prop = FuncAeroProp()

        self.dt = 0.1
        # self.state = GuamState.create()
        self.checker = 0
        
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

        surf_act, prop_act = self.surf_eng.get_surf_prop_act(state.surf_eng, control.Cmd, power)
        d_state_surf_eng = self.surf_eng.surf_engine_state_deriv(control.Cmd, state.surf_eng)

        fm = self.aero_prop.aero_prop(prop_act, surf_act, env_data, aeroprop_body_data)
        fm_total = self.veh_eom.get_fm_with_gravity(state.aircraft, fm)
        d_state_aircraft = self.veh_eom.state_deriv(fm_total, state.aircraft)

        return GuamState(d_state_controller, d_state_aircraft, d_state_surf_eng)

    # def dynamic_mode1(self, t, state):
    #     #TODO: fill in the EOM/dynamics of the GUAM model

    #     return X_dot
    def step(self, dt: float, state: GuamState, ref_inputs: RefInputs) -> GuamState:
        deriv_fn = ft.partial(self.deriv, ref_inputs=ref_inputs)
        state_new = ode3(deriv_fn, dt, state)
        # We also need to clip the state.
        state_new = state_new._replace(surf_eng=self.surf_eng.clip_state(state_new.surf_eng))
        return state_new


    def action_handler(self, mode, state_guam, ref):
        if mode == 'Mode1':
            return self.step(self.dt, state_guam, ref)
        else:
            raise ValueError
    
    def array2GuamState(self, state_arr):
      
        e_long = state_arr[0:3]
        e_lat = state_arr[3:6]
        aircraft_state = np.array(state_arr[6:19])
        surf_state = state_arr[19:24]
        
        lc_control_state = LCControlState(int_e_long=np.array(e_long), int_e_lat=np.array(e_lat))
        surf_engine_state = SurfEngineState(ctrl_surf_state=np.array(surf_state))
        # Aircraft_state = AircraftState(np.array(aircraft_state[0:3]), np.array(aircraft_state[3:6]), np.array(aircraft_state[6:9]), np.array(aircraft_state[9:13]))
        
        
        
        return GuamState(
            controller=lc_control_state,
            aircraft=aircraft_state,
            surf_eng=surf_engine_state
        )
    
    def GuamState2array(self, state_guam):
        
        int_e_long = state_guam.controller.int_e_long
        int_e_lat = state_guam.controller.int_e_lat
        aircraft = state_guam.aircraft
        ctrl_surf_state = state_guam.surf_eng.ctrl_surf_state
       
        
        return np.concatenate((int_e_long, int_e_lat, aircraft, ctrl_surf_state)).tolist()
    


    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:


        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        


        state_arr = initialCondition
        state = self.array2GuamState(state_arr)
        trace = [[0]+state_arr]
        r = self.action_handler(mode[0], state, lift_cruise_reference_inputs(0))
        # r.set_integrator('dopri5', nsteps=6000).set_initial_value(init)  
        for i in range(len(t)):
            curr_t = i * self.dt
            print(curr_t)
            # res: np.ndarray = r.integrate(r.t + time_step)
            # init = res.flatten().tolist()
            ref_input = lift_cruise_reference_inputs(curr_t)
            state = self.step(self.dt, state, ref_input)

            state_arr = self.GuamState2array(state)
            trace.append([t[i] + time_step] + state_arr)
        self.checker = self.checker + 1
        # print(trace)
        return np.array(trace)



if __name__ == '__main__':
    aguam = guam_agent('agent1')

    # Initialize simulation states
    trace = aguam.TC_Simulate(['none'], [1.25, 2.25], 7, 0.05) # this line does not need to be accurate
    print(trace)

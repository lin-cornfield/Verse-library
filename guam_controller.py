from enum import Enum, auto
import copy
import numpy as np

import functools as ft
from typing import NamedTuple

import jax
import tqdm
from jax_guam.functional.aero_prop_new import FuncAeroProp
from jax_guam.functional.lc_control import LCControl, LCControlState
from jax_guam.functional.surf_engine import SurfEngine, SurfEngineState
from jax_guam.functional.vehicle_eom_simple import VehicleEOMSimple
from jax_guam.guam_types import AircraftState, AircraftStateVec, EnvData, PwrCmd, RefInputs
from jax_guam.subsystems.environment.environment import Environment
from jax_guam.subsystems.genctrl_inputs.genctrl_inputs import lift_cruise_reference_inputs
from jax_guam.subsystems.vehicle_model_ref.power_system import power_system
from jax_guam.utils.ode import ode3


class AgentMode(Enum):
    Mode1 = auto()
    

# class GuamState(NamedTuple):
#     # [   0:3   ,   3:6     ]
#     # [   long  ,   lat     ]
#     controller: LCControlState 
#     # [   0:3  ,   3:6    ,   6:9  ,  9:13 ]
#     # [ vel_bEb, Omega_BIb, pos_bii, Q_i2b ]
#     aircraft: AircraftStateVec
#     # [     0:5   ]
#     surf_eng: SurfEngineState

#     @property
#     def pos_ned(self):
#         return self.aircraft[..., 6:9]

#     @staticmethod
#     def create():
#         return GuamState(
#             controller=LCControlState.create(),
#             aircraft=AircraftState.GetDefault13(),
#             surf_eng=SurfEngineState.create(),
#         )
        
class State:
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    x5 = 0.0
    x6 = 0.0
    x7 = 0.0
    x8 = 0.0
    x9 = 0.0
    x10 = 0.0
    x11 = 0.0
    x12 = 0.0
    x13 = 0.0
    x14 = 0.0
    x15 = 0.0
    x16 = 0.0
    x17 = 0.0
    x18 = 0.0
    x19 = 0.0
    x20 = 0.0
    x21 = 0.0
    x22 = 0.0
    x23 = 0.0
    x24 = 0.0
    
    agent_mode: AgentMode = AgentMode.Mode1
    
    def __init_(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, agent_mode: AgentMode):
        pass
    
def decisionLogic(ego: State, lane_map):
    output = copy.deepcopy(ego)
    
    return output

# Example agent.
from typing import Tuple, List

import numpy as np
import math
from scipy.integrate import ode
import numpy.linalg as la

from verse import BaseAgent
from verse.map import LaneMap

class guam_agent(BaseAgent):
    def __init__(self, id, code=None, file_name=None):
        # Calling the constructor of tha base class
        # TODO: some initialization of the GUAM model (parameters, etc)
        super().__init__(id, code, file_name)
        self.checker = 0

    def dynamic_mode1(self, t, state):
        #TODO: fill in the EOM/dynamics of the GUAM model

        return X_dot


    def action_handler(self, mode):
        if mode == 'Mode1':
            return ode(self.dynamic_mode1)
        else:
            raise ValueError


    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, lane_map: LaneMap = None) -> np.ndarray:


        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        print('---same call of tc simulate----')
        print(self.checker)


        init = initialCondition
        trace = [[0]+init]
        r = self.action_handler(mode[0])
        r.set_integrator('dopri5', nsteps=6000).set_initial_value(init)  
        for i in range(len(t)):
            print(r.t)
            res: np.ndarray = r.integrate(r.t + time_step)
            init = res.flatten().tolist()
            trace.append([t[i] + time_step] + init)
        self.checker = self.checker + 1
        return np.array(trace)



if __name__ == '__main__':
    aguam = guam_agent('agent1')

    # Initialize simulation states
    trace = aguam.TC_Simulate(['none'], [1.25, 2.25], 7, 0.05) # this line does not need to be accurate
    print(trace)

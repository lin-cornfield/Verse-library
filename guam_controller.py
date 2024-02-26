from enum import Enum, auto
import copy


class AgentMode(Enum):
    Mode1 = auto()
    

class State:
    # TODO: Guam State
    agent_mode: AgentMode = AgentMode.Mode1 # mode initialization

    def __init__(self, agent_mode: AgentMode): #TODO. pass the guam model states
        pass


def decisionLogic(ego: State, lane_map):
    output = copy.deepcopy(ego)
    
    return output

from enum import Enum, auto
import copy


class AgentMode(Enum):
    Mode1 = auto()
    Mode2 = auto()
    Mode3 = auto()
    Mode4 = auto()
    Mode5 = auto()
    Mode6 = auto()
    Mode7 = auto()
    Mode8 = auto()


class State:
    x1 = 0.0
    x2 = 0.0
    x3 = 0.0
    x4 = 0.0
    x5 = 0.0
    x6 = 0.0
    x7 = 1.0
    x8 = 0.0
    x9 = 0.0
    x10 = 0.0
    x11 = 1.0
    x12 = 0.0
    x13 = 0.0
    x14 = 0.0
    x15 = 1.0
    x16 = 0.0
    x17 = 0.0
    x18 = 0.0
    mass = 0.7
    cycle_time = 0.0
    agent_mode: AgentMode = AgentMode.Mode8 # mode initialization

    def __init__(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, mass, cycletime, agent_mode: AgentMode):
        pass


def decisionLogic(ego: State, lane_map):
    output = copy.deepcopy(ego)
    # print(ego.cycle_time)
    # if ego.agent_mode == AgentMode.Mode1:
    #     if ego.cycle_time>6:
    #         output.agent_mode = AgentMode.Mode2
    #         output.cycle_time = 0.0

    # if ego.agent_mode == AgentMode.Mode2:
    #     if ego.cycle_time>6:
    #         output.agent_mode = AgentMode.Mode3
    #         output.cycle_time = 0.0
    # if ego.agent_mode == AgentMode.Mode8:
    #     if ego.cycle_time>50+0.05:
    #         output.agent_mode = AgentMode.Mode4
    #         output.cycle_time = 0.0

    # if ego.agent_mode == AgentMode.Mode2:
    #     if ego.cycle_time>6:
    #         output.agent_mode = AgentMode.Mode3
    #         output.cycle_time = 0.0
    return output

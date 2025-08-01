from pysc2.lib import actions, features, units
import random

class BaseSetup():
    def __init__(self, build_order=None, tech=None, units=None):
        self.build_order = build_order.copy()
        self.tech = tech.copy()
        self.units = units.copy()

    def get_units_by_type(self, observations, unit_type, ready_only=False, same_side=True):
        if ready_only:
            if same_side:
                return [unit for unit in observations.observation.raw_units if ((unit.unit_type == unit_type) and (unit.alliance == features.PlayerRelative.SELF) and (unit.build_progress == 100))]
            else:
                return [unit for unit in observations.observation.raw_units if ((unit.unit_type == unit_type) and (unit.build_progress == 100))]
        else:
            if same_side:
                return [unit for unit in observations.observation.raw_units if ((unit.unit_type == unit_type) and (unit.alliance == features.PlayerRelative.SELF))]
            else:
                return [unit for unit in observations.observation.raw_units if (unit.unit_type == unit_type)]

    def base_setup_step(self, observations):
        base_setup_complete = True
        base_setup_action = actions.FUNCTIONS.no_op()

        return base_setup_complete, base_setup_action

    def research_tech(self, observations):
        research_complete = True
        research_action = actions.FUNCTIONS.no_op()

        return research_complete, research_action

    def build_units_step(self, observations):
        build_units_complete = True
        build_units_action = actions.FUNCTIONS.no_op()

        return build_units_complete, build_units_action

    def position_units(self, observations, position, num_agents):
        positions_complete = False
        position_action = actions.FUNCTIONS.no_op()

        return positions_complete, position_action

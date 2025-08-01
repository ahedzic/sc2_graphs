from battle_setup.base_setup import BaseSetup
from pysc2.lib import actions, features, units
import random

BUILDING_COSTS = {
    units.Terran.SupplyDepot: { 'minerals': 100, 'gas': 0 },
    units.Terran.Barracks:  { 'minerals': 150, 'gas': 0 }
}

UNIT_COSTS = {
    units.Terran.Marine:  { 'minerals': 150, 'gas': 0 }
}

class TerranSetup(BaseSetup):
    def __init__(self, build_order=None, tech=None, units=None):
        super(TerranSetup, self).__init__(build_order, tech, units)

    def get_terran_action(self, unit, tag=None, position=None):
        action = actions.FUNCTIONS.no_op()

        if unit == units.Terran.SupplyDepot:
            action = actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", tag, position)
        elif unit == units.Terran.Barracks:
            action = actions.RAW_FUNCTIONS.Build_Barracks_pt("now", tag, position)
        elif unit == units.Terran.Marine:
            action = actions.RAW_FUNCTIONS.Train_Marine_quick("now", tag)

        return action

    def base_setup_step(self, observations):
        base_setup_complete = True
        base_setup_action = actions.FUNCTIONS.no_op()

        if len(self.build_order) > 0:
            base_setup_complete = False
            next_order = self.build_order[0]
            next_units = self.get_units_by_type(observations, next_order[0], True)

            if len(next_units) == next_order[1]:
                self.build_order.pop(0)
            else:
                command_center = self.get_units_by_type(observations, units.Terran.CommandCenter)[0]

                if command_center.x <= 31:
                    self.attack_coordinates = (49, 49)
                    self.first_depot = (60, 60)
                else:
                    self.attack_coordinates = (12, 16)
                    self.first_depot = (10, 10)

                minerals = observations.observation.player.minerals
                vespene = observations.observation.player.vespene

                if minerals >= BUILDING_COSTS[next_order[0]]['minerals']:
                    if command_center.x <= 31:
                        x = random.randint(0, 39)
                        y = random.randint(0, 39)
                    else:
                        x = random.randint(50, 83)
                        y = random.randint(50, 83)

                    scvs = self.get_units_by_type(observations, units.Terran.SCV)
                    scv = random.choice(scvs)
                    base_setup_action = self.get_terran_action(next_order[0], scv.tag, (x, y))

        return base_setup_complete, base_setup_action

    def build_units_step(self, observations):
        build_units_complete = True
        build_units_action = actions.FUNCTIONS.no_op()

        if len(self.units) > 0:
            build_units_complete = False
            next_order = self.units[0]
            minerals = observations.observation.player.minerals
            vespene = observations.observation.player.vespene

            if minerals >= UNIT_COSTS[next_order[0]]['minerals']:
                buildings = self.get_units_by_type(observations, units.Terran.Barracks, True)

                if len(buildings) == 2:
                    building = min(buildings, key=lambda x: x.order_length)
                    build_units_action = self.get_terran_action(next_order[0], building.tag)
                    self.units[0] = (next_order[0], next_order[1] - 1)
                    
                    if self.units[0][1] == 0:
                        self.units.pop(0)

        return build_units_complete, build_units_action

    def position_units(self, observations, position):
        positions_complete = False
        position_actions = []
        marines = self.get_units_by_type(observations, units.Terran.Marine, True)
        
        if len(marines) == 10:
            for marine in marines:
                position_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, position))
            
            positions_complete = True

        if len(position_actions) == 0:
            position_actions.append(actions.FUNCTIONS.no_op())

        return positions_complete, position_actions

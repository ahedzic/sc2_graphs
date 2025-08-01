from battle_setup.base_setup import BaseSetup
from pysc2.lib import actions, features, units
from sc2_env_util_protoss import get_distance
import random

BUILDING_COSTS = {
    units.Protoss.Probe:  { 'minerals': 50, 'gas': 0 },
    units.Protoss.Assimilator:  { 'minerals': 75, 'gas': 0 },
    units.Protoss.Pylon: { 'minerals': 100, 'gas': 0 },
    units.Protoss.Gateway:  { 'minerals': 150, 'gas': 0 },
    units.Protoss.CyberneticsCore:  { 'minerals': 150, 'gas': 0 },
    units.Protoss.TwilightCouncil:  { 'minerals': 150, 'gas': 100 },
    units.Protoss.TemplarArchive:  { 'minerals': 150, 'gas': 200 }
}

RESEARCH_COSTS = {
    "Research_PsiStorm_quick": { 'minerals': 200, 'gas': 200 }
}

UNIT_COSTS = {
    units.Protoss.Stalker:  { 'minerals': 125, 'gas': 50 },
    units.Protoss.Sentry:  { 'minerals': 50, 'gas': 100 },
    units.Protoss.HighTemplar:  { 'minerals': 50, 'gas': 150 }
}

class ProtossSetup(BaseSetup):
    def __init__(self, build_order=None, tech=None, units=None):
        super(ProtossSetup, self).__init__(build_order, tech, units)
        self.available_units = {}
        self.assimilator_capacity = {}

    def get_research_action(self, tag=None, research=None):
        action = actions.FUNCTIONS.no_op()

        if research == "Research_PsiStorm_quick":
            action = actions.RAW_FUNCTIONS.Research_PsiStorm_quick("now", tag)

        return action

    def get_protoss_action(self, unit, tag=None, position=None):
        action = actions.FUNCTIONS.no_op()

        if unit == units.Protoss.Probe:
            action = actions.RAW_FUNCTIONS.Train_Probe_quick("now", tag)
        elif unit == units.Protoss.Assimilator:
            action = actions.RAW_FUNCTIONS.Build_Assimilator_unit("now", tag, position)
        elif unit == units.Protoss.Pylon:
            action = actions.RAW_FUNCTIONS.Build_Pylon_pt("now", tag, position)
        elif unit == units.Protoss.Gateway:
            action = actions.RAW_FUNCTIONS.Build_Gateway_pt("now", tag, position)
        elif unit == units.Protoss.CyberneticsCore:
            action = actions.RAW_FUNCTIONS.Build_CyberneticsCore_pt("now", tag, position)
        elif unit == units.Protoss.TwilightCouncil:
            action = actions.RAW_FUNCTIONS.Build_TwilightCouncil_pt("now", tag, position)
        elif unit == units.Protoss.TemplarArchive:
            action = actions.RAW_FUNCTIONS.Build_TemplarArchive_pt("now", tag, position)
        elif unit == units.Protoss.Stalker:
            action = actions.RAW_FUNCTIONS.Train_Stalker_quick("now", tag)
        elif unit == units.Protoss.Sentry:
            action = actions.RAW_FUNCTIONS.Train_Sentry_quick("now", tag)
        elif unit == units.Protoss.HighTemplar:
            action = actions.RAW_FUNCTIONS.Train_HighTemplar_quick("now", tag)

        return action

    def base_setup_step(self, observations):
        base_setup_complete = True
        base_setup_action = actions.FUNCTIONS.no_op()

        assimilators = self.get_units_by_type(observations, units.Protoss.Assimilator, True)

        if len(assimilators):
            for assimilator in assimilators:
                if assimilator.tag not in self.assimilator_capacity.keys():
                    self.assimilator_capacity[assimilator.tag] = 0
                    
                if self.assimilator_capacity[assimilator.tag] < 3:
                    probes = self.get_units_by_type(observations, units.Protoss.Probe)
                    idle_probes = [probe for probe in probes if probe.order_length == 0]

                    if (len(idle_probes)):
                        probe = random.choice(idle_probes)
                    else:
                        probe = random.choice(probes)

                    self.assimilator_capacity[assimilator.tag] += 1
                    return False, actions.RAW_FUNCTIONS.Harvest_Gather_Probe_unit("now", probe.tag, assimilator.tag)

        if len(self.build_order) > 0:
            base_setup_complete = False
            next_order = self.build_order[0]
            next_units = self.get_units_by_type(observations, next_order[0], False, True)
            left_to_build = next_order[1] - len(next_units)

            if left_to_build <= 0:
                self.build_order.pop(0)
            else:
                nexus = self.get_units_by_type(observations, units.Protoss.Nexus)[0]

                if nexus.x <= 31:
                    self.attack_coordinates = (49, 49)
                    self.first_depot = (60, 60)
                else:
                    self.attack_coordinates = (12, 16)
                    self.first_depot = (10, 10)

                minerals = observations.observation.player.minerals
                vespene = observations.observation.player.vespene

                if minerals >= BUILDING_COSTS[next_order[0]]['minerals'] and vespene >= BUILDING_COSTS[next_order[0]]['gas']:
                    if nexus.x <= 31:
                        x = random.randint(0, 39)
                        y = random.randint(0, 39)
                    else:
                        x = random.randint(50, 83)
                        y = random.randint(50, 83)

                    probes = self.get_units_by_type(observations, units.Protoss.Probe)
                    idle_probes = [probe for probe in probes if probe.order_length == 0]

                    if (len(idle_probes)):
                        probe = random.choice(idle_probes)
                    else:
                        probe = random.choice(probes)

                    if next_order[0] == units.Protoss.Assimilator:
                        geysers = self.get_units_by_type(observations, units.Neutral.VespeneGeyser, False, False)

                        if len(geysers):
                            closest_geyser = 100
                            closest_tag = geysers[0].tag
                            
                            for geyser in geysers:
                                distance = get_distance((probe.x, probe.y), (geyser.x, geyser.y))

                                if distance < closest_geyser:
                                    closest_geyser = distance
                                    closest_tag = geyser.tag

                            base_setup_action = self.get_protoss_action(next_order[0], probe.tag, closest_tag)
                    elif next_order[0] == units.Protoss.Probe:
                        base_setup_action = self.get_protoss_action(next_order[0], nexus.tag, (x, y))
                    else:
                        base_setup_action = self.get_protoss_action(next_order[0], probe.tag, (x, y))

        return base_setup_complete, base_setup_action

    def research_tech(self, observations):
        research_complete = True
        research_action = actions.FUNCTIONS.no_op()

        if len(self.tech) > 0:
            research_complete = False
            next_order = self.tech[0]
            minerals = observations.observation.player.minerals
            vespene = observations.observation.player.vespene

            if minerals >= RESEARCH_COSTS[next_order[1]]['minerals'] and vespene >= RESEARCH_COSTS[next_order[1]]['gas']:
                buildings = self.get_units_by_type(observations, next_order[0], True)

                if len(buildings) > 0:
                    research_action = self.get_research_action(buildings[0].tag, next_order[1])
                    self.tech.pop(0)

        return research_complete, research_action

    def build_units_step(self, observations):
        build_units_complete = True
        build_units_action = actions.FUNCTIONS.no_op()

        if len(self.units) > 0:
            build_units_complete = False
            next_order = self.units[0]
            minerals = observations.observation.player.minerals
            vespene = observations.observation.player.vespene

            if minerals >= UNIT_COSTS[next_order[0]]['minerals'] and vespene >= UNIT_COSTS[next_order[0]]['gas']:
                buildings = self.get_units_by_type(observations, units.Protoss.Gateway, True)

                if len(buildings) >= 3:
                    building = min(buildings, key=lambda x: x.order_length)
                    build_units_action = self.get_protoss_action(next_order[0], building.tag)
                    self.units[0] = (next_order[0], next_order[1] - 1)
                    
                    if next_order[0] in self.available_units.keys():
                        self.available_units[next_order[0]] += 1
                    else:
                        self.available_units[next_order[0]] = 1
                    
                    if self.units[0][1] <= 0:
                        self.units.pop(0)
        else:
            for unit_type in self.available_units.keys():
                unit_list = self.get_units_by_type(observations, unit_type, True)

                if len(unit_list) != self.available_units[unit_type]:
                    build_units_complete = False

        return build_units_complete, build_units_action

    def position_units(self, observations, position, unit_config, num_agents):
        positions_complete = False
        position_actions = []
        unit_lists = []
        total_units = 0

        for unit_type in self.available_units.keys():
            unit_list = self.get_units_by_type(observations, unit_type, True)
            expected_count = 0

            for expected_unit in unit_config:
                if expected_unit[0] == unit_type:
                    expected_count = expected_unit[1]

            if len(unit_list) == expected_count:
                total_units += len(unit_list)
                unit_lists.append(unit_list)
        
        if total_units == num_agents:
            for unit_list in unit_lists:
                for unit in unit_list:
                    position_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, position))
            
            positions_complete = True

        if len(position_actions) == 0:
            position_actions.append(actions.FUNCTIONS.no_op())

        return positions_complete, position_actions

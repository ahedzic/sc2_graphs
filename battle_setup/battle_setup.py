from pysc2.env import sc2_env
from pysc2.lib import actions

from battle_setup.terran_setup import TerranSetup
from battle_setup.protoss_setup import ProtossSetup

class BattleSetup():
    def __init__(self, config=None):
        self.config = config
        self.player_setups = []
        self.ready_bases = []
        self.ready_research = []
        self.ready_units = []
        self.ready_positions = []
        self.ready_actions = {}

        for player in config:
            self.player_setups.append(self.get_race_setup(player['race'], player['build_order'], player['research'], player['units']))

    def get_race_setup(self, race, build_order, tech, units):
        if race == sc2_env.Race.terran:
            return TerranSetup(build_order, tech, units)
        if race == sc2_env.Race.protoss:
            return ProtossSetup(build_order, tech, units)

    def base_setup_step(self, observations):
        base_setup_complete = False
        base_setup_actions = []
        
        for i in range(len(observations)):
            base_setup_actions.append([])

            if i not in self.ready_bases:
                base_setup = self.player_setups[i]
                base_setup_ready, base_setup_action = base_setup.base_setup_step(observations[i])
                base_setup_actions[i].append(base_setup_action)

                if base_setup_ready:
                    self.ready_bases.append(i)
            else:
                base_setup_actions[i].append(actions.FUNCTIONS.no_op())

        if len(self.ready_bases) == len(self.player_setups):
            base_setup_complete = True

        return base_setup_complete, base_setup_actions

    def research_tech(self, observations):
        research_complete = False
        research_actions = []
        
        for i in range(len(observations)):
            research_actions.append([])

            if i not in self.ready_research:
                research_setup = self.player_setups[i]
                research_ready, research_action = research_setup.research_tech(observations[i])
                research_actions[i].append(research_action)

                if research_ready:
                    self.ready_research.append(i)
            else:
                research_actions[i].append(actions.FUNCTIONS.no_op())

        if len(self.ready_research) == len(self.player_setups):
            research_complete = True

        return research_complete, research_actions

    def build_units_step(self, observations):
        build_units_complete = False
        build_units_actions = []
        
        for i in range(len(observations)):
            build_units_actions.append([])

            if i not in self.ready_units:
                build_units_setup = self.player_setups[i]
                build_units_ready, build_units_action = build_units_setup.build_units_step(observations[i])
                build_units_actions[i].append(build_units_action)

                if build_units_ready:
                    self.ready_units.append(i)
            else:
                build_units_actions[i].append(actions.FUNCTIONS.no_op())

        if len(self.ready_units) == len(self.player_setups):
            build_units_complete = True

        return build_units_complete, build_units_actions

    def position_units(self, observations, position, units_config, num_agents):
        positions_complete = False
        all_position_actions = []
        
        for i in range(len(observations)):
            all_position_actions.append([])

            if i not in self.ready_positions:
                position_setup = self.player_setups[i]
                positions_ready, position_actions = position_setup.position_units(observations[i], position, units_config[i]['units'], num_agents)
                all_position_actions[i] = position_actions

                if positions_ready:
                    self.ready_positions.append(i)
                    self.ready_actions[i] = position_actions
            else:
                all_position_actions[i] = self.ready_actions[i]

        if len(self.ready_positions) == len(self.player_setups):
            positions_complete = True

        return positions_complete, all_position_actions
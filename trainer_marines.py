from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
from BiCNet_agent import BiCNetAgent
from battle_setup.battle_setup import BattleSetup
from pysc2.agents import base_agent
from sc2_env_util_marines import get_agent_units, get_state_space, get_action_position, get_actions, get_rewards, check_end, get_distance
from sc2_graph import get_deltas, create_graph
from graph_image import create_graph_image
import numpy as np
import torch as th
import time

DEFAULT_CONFIG = [
    {
        'id': 'p1',
        'race': sc2_env.Race.terran,
        'build_order': [
            (units.Terran.SupplyDepot, 1),
            (units.Terran.Barracks, 2)
        ],
        'units': [
            (units.Terran.Marine, 10)
        ]
    },
    {
        'id': 'p2',
        'race': sc2_env.Race.terran,
        'build_order': [
            (units.Terran.SupplyDepot, 1),
            (units.Terran.Barracks, 2)
        ],
        'units': [
            (units.Terran.Marine, 10)
        ]
    },
]

STATE_SPACE_SIZE = 60
ACTION_SPACE_SIZE = 3
NUM_AGENTS = 10
BICNET_CONFIG = {
    'algo': 'BiCNet',
    'model_episode': 1,
    'batch_size': 32,
    'memory_length': int(1e5),
    'a_lr': 0.002,
    'c_lr': 0.002,
    'ou_theta': 0.15,
    'ou_mu': 0.0,
    'ou_sigma': 0.2,
    'tau': 0.001,
    'epsilon_decay': 1000000,
    'gamma': 0.95
}
MAX_STEP_COUNT = 800

def main(unused_argv):
    config = DEFAULT_CONFIG
    agent = BiCNetAgent(STATE_SPACE_SIZE, ACTION_SPACE_SIZE, NUM_AGENTS, BICNET_CONFIG)
    agent.load_model()
    agent.prep_train()

    try:
        while True:
            player_list = []

            for player in config:
                player_list.append(sc2_env.Agent(player['race']))

            with sc2_env.SC2Env(
                map_name ="Simple64",
                players=player_list,
                agent_interface_format = features.AgentInterfaceFormat(
                    feature_dimensions = features.Dimensions(screen=84, minimap=64),
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units = True
                ),
                step_mul = 8,
                game_steps_per_episode = 0,
                visualize=False
            ) as env:
                episode = 0

                while True:
                    battle_setup = BattleSetup(config)
                    agent.setup(env.observation_spec(), env.action_spec())
                    observations = env.reset()
                    p1_last_state = np.zeros((10, 60))
                    p2_last_state = np.zeros((10, 60))
                    p1_last_health_map = {}
                    p2_last_health_map = {}
                    p1_last_action = np.zeros((10, 3))
                    p2_last_action = np.zeros((10, 3))
                    p1_centroid = (0, 0)
                    p2_centroid = (0, 0)
                    agent.reset()
                    base_setup = False
                    unit_setup = False
                    positioned = False
                    step_count = 0
                    episode_start = False

                    # Need to setup environment for battle first
                    # Build base
                    while not base_setup and (step_count < MAX_STEP_COUNT):
                        base_setup, setup_actions = battle_setup.base_setup_step(observations)
                        observations = env.step(setup_actions)
                        step_count += 1

                    # Train units
                    while not unit_setup and (step_count < MAX_STEP_COUNT):
                        unit_setup, setup_actions = battle_setup.build_units_step(observations)
                        observations = env.step(setup_actions)
                        step_count += 1

                    # Position units
                    while not positioned and (step_count < MAX_STEP_COUNT):
                        positioned, setup_actions = battle_setup.position_units(observations, (55, 44))
                        observations = env.step(setup_actions)
                        step_count += 1

                    # Now start the agent loop
                    p1_units, p1_health_map = get_agent_units(observations[0], DEFAULT_CONFIG[0]['units'], NUM_AGENTS)
                    p2_units, p2_health_map = get_agent_units(observations[1], DEFAULT_CONFIG[1]['units'], NUM_AGENTS)
                    done = False
                    too_far = False

                    while (not check_end(p1_health_map, p2_health_map) and (step_count < MAX_STEP_COUNT)) and (not too_far):
                        # Grab state space for each player
                        p1_state_space, p1_health_map, p1_centroid, p1_units = get_state_space(observations[0], p1_units, p2_units)
                        p2_state_space, p2_health_map, p2_centroid, p2_units = get_state_space(observations[1], p2_units, p1_units)
                        
                        if (not episode_start) and (get_distance(p1_centroid, p2_centroid) <= 13):
                            episode_start = True

                        if episode_start:
                            # Update agent
                            p1_done = []
                            p2_done = []

                            for p1_unit in p1_units:
                                if (p1_unit.tag not in p1_health_map.keys()) or (p1_health_map[p1_unit.tag] == 0.0):
                                    p1_done.append(True)
                                else:
                                    p1_done.append(False)

                            for p2_unit in p2_units:
                                if (p2_unit.tag not in p2_health_map.keys()) or (p2_health_map[p2_unit.tag] == 0.0):
                                    p2_done.append(True)
                                else:
                                    p2_done.append(False)

                            p2_actions = []
                            for marine in p2_units:
                                if step_count % 3 == 0:
                                    p2_actions.append(actions.RAW_FUNCTIONS.Attack_pt("now", marine.tag, p1_centroid))
                                else:
                                    p2_actions.append(actions.RAW_FUNCTIONS.Move_pt("now", marine.tag, p1_centroid))

                            p1_rewards = get_rewards(p1_units, p1_health_map, p2_health_map, p1_last_health_map, p2_last_health_map)
                            #p2_rewards = get_rewards(p2_units, p2_health_map, p1_health_map, p2_last_health_map, p1_last_health_map)
                            agent.memory(p1_last_state, p1_last_action, p1_rewards, p1_state_space, np.array(p1_done))
                            #agent.memory(p2_last_state, p2_last_action, p2_rewards, p2_state_space, np.array(p1_done))
                            agent.update(step_count)

                            # Get agent action states for each player
                            p1_action_states = agent.choose_action(p1_state_space)
                            #p2_action_states = agent.choose_action(p2_state_space)
                            p1_actions = get_actions(observations[0], p1_action_states, p1_units)
                            #p2_actions = get_actions(observations[1], p2_action_states, p2_units)
                            
                            # Keep camera on units
                            p1_actions.append(actions.RAW_FUNCTIONS.raw_move_camera(p1_centroid))
                            p2_actions.append(actions.RAW_FUNCTIONS.raw_move_camera(p2_centroid))

                            # Step through environment with agent actions
                            observations = env.step([p1_actions, p2_actions])
                            #time.sleep(1 / (22.4 / 8))

                            # Track actions for last time step
                            p1_last_action = p1_action_states
                            #p2_last_action = p2_action_states

                            if get_distance(p1_centroid, p2_centroid) > 32:
                                too_far = True
                        else:
                            if (p1_centroid[0] != 0) and (p1_centroid[1] != 0):
                                observations = env.step([[actions.RAW_FUNCTIONS.raw_move_camera(p1_centroid)], [actions.RAW_FUNCTIONS.raw_move_camera(p2_centroid)]])
                            else:
                                observations = env.step([[actions.RAW_FUNCTIONS.no_op()], [actions.RAW_FUNCTIONS.no_op()]])

                        # Track variables for last time step
                        p1_last_state = p1_state_space
                        p2_last_state = p2_state_space
                        p1_last_health_map = p1_health_map
                        p2_last_health_map = p2_health_map
                        
                        if episode_start:
                            step_count += 1

                    # Final agent update
                    p1_state_space, p1_health_map, p1_centroid, p1_units = get_state_space(observations[0], p1_units, p2_units)
                    p2_state_space, p2_health_map, p2_centroid, p2_units = get_state_space(observations[1], p2_units, p1_units)
                    p1_rewards = get_rewards(p1_units, p1_health_map, p2_health_map, p1_last_health_map, p2_last_health_map)
                    p2_rewards = get_rewards(p2_units, p2_health_map, p1_health_map, p2_last_health_map, p1_last_health_map)
                    agent.memory(p1_last_state, p1_last_action, p1_rewards, p1_state_space, np.array([True] * NUM_AGENTS))
                    agent.memory(p2_last_state, p2_last_action, p2_rewards, p2_state_space, np.array([True] * NUM_AGENTS))

                    if (episode % 1 == 0):
                        print("SAVING MODEL")
                        agent.save_model()
                    episode += 1
                    #env. save_replay("D:/sc2_rl/replays", prefix="episode_" + str(episode))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    app.run(main)
from pysc2.lib import actions, units as sc2_units
from statistics import mean
import unit_data as ut
import math
import numpy as np
import torch as th

class AgentUnit():
    def __init__(self, tag, unit_type, position):
        self.tag = tag
        self.unit_type = unit_type
        self.position = position

def get_agent_units(observations, units, num_agents):
    obs_units = observations.observation.raw_units
    agent_units = [AgentUnit(0, sc2_units.Protoss.Stalker, (0, 0))] * num_agents
    health_map = {}
    total_count = 0

    for unit in units:
        unit_type = unit[0]
        required_count = unit[1]
        current_count = 0

        for obs_unit in obs_units:
            if (obs_unit.unit_type == unit_type):
                if (current_count < required_count) and (total_count < num_agents):
                    agent_units[total_count] = AgentUnit(obs_unit.tag, obs_unit.unit_type, (obs_unit.x, obs_unit.y))
                    #print(total_count, obs_unit.tag, obs_unit.unit_type, (obs_unit.x, obs_unit.y))
                    current_count += 1
                    total_count += 1
                    health_map[obs_unit.tag] = (45.0, {}, {})

    #for agent in agent_units:
    #    print(agent.tag, agent.unit_type, agent.position)
    #print(total_count)

    return agent_units, health_map

def get_distance(pos1, pos2):
    position1 = th.tensor([pos1[0], pos1[1]])
    position2 = th.tensor([pos2[0], pos2[1]])

    return (position1 - position2).pow(2).sum().sqrt().item()

def get_centroid(positions):
    if len(positions) > 0:
        xs = []
        ys = []

        for pos in positions:
            xs.append(pos[0])
            ys.append(pos[1])

        return (mean(xs), mean(ys))
    else:
        return (0, 0)

def in_range(position, unit_type, target_position):
    return get_distance(position, target_position) <= ut.unit_data[unit_type]["sight"]

def get_state_space(observations, unit_tags, enemy_tags):
    state_spaces = np.zeros((10, 120))
    health_map = {}
    positions = []

    for i in range(len(unit_tags)):
        current_position = (0, 0)
        friend_index = 1 # Start at one cause agents point of view is itself at index 0
        enemy_index = 0
        closest_friends = {}
        closest_enemies = {}
        health = 0.0

        for unit in observations.observation.raw_units:
            if unit.tag == unit_tags[i].tag:
                current_position = (unit.x, unit.y)
                unit_tags[i].position = current_position
                positions.append(current_position)
                state_spaces[i][0] = unit.x / 64.0
                state_spaces[i][1] = unit.y / 64.0
                state_spaces[i][2] = unit.health / 45.0

                if unit.unit_type == sc2_units.Protoss.Stalker:
                    state_spaces[i][3] = 1.0
                    state_spaces[i][4] = 0.0
                    state_spaces[i][5] = 0.0
                elif unit.unit_type == sc2_units.Protoss.Sentry:
                    state_spaces[i][3] = 0.0
                    state_spaces[i][4] = 1.0
                    state_spaces[i][5] = 0.0
                elif unit.unit_type == sc2_units.Protoss.HighTemplar:
                    state_spaces[i][3] = 0.0
                    state_spaces[i][4] = 0.0
                    state_spaces[i][5] = 1.0

                health = unit.health
                break
        
        for unit in observations.observation.raw_units:
            if unit.tag != unit_tags[i].tag:
                position = (unit.x, unit.y)

                if in_range(current_position, unit_tags[i].unit_type, position):
                    for j in range(len(unit_tags)):
                        if (unit_tags[i].tag != unit_tags[j].tag) and (unit.tag == unit_tags[j].tag):
                            offset = friend_index * 3
                            state_spaces[i][offset] = unit.x / 64.0
                            state_spaces[i][offset + 1] = unit.y / 64.0
                            state_spaces[i][offset + 2] = unit.health / 45.0

                            if unit.unit_type == sc2_units.Protoss.Stalker:
                                state_spaces[i][offset + 3] = 1.0
                                state_spaces[i][offset + 4] = 0.0
                                state_spaces[i][offset + 5] = 0.0
                            elif unit.unit_type == sc2_units.Protoss.Sentry:
                                state_spaces[i][offset + 3] = 0.0
                                state_spaces[i][offset + 4] = 1.0
                                state_spaces[i][offset + 5] = 0.0
                            elif unit.unit_type == sc2_units.Protoss.HighTemplar:
                                state_spaces[i][offset + 3] = 0.0
                                state_spaces[i][offset + 4] = 0.0
                                state_spaces[i][offset + 5] = 1.0

                            closest_friends[unit.tag] = unit.health
                            friend_index += 1
                    for j in range(len(enemy_tags)):
                        if (unit.tag == enemy_tags[j].tag):
                            offset = enemy_index * 3 + 30
                            state_spaces[i][offset] = unit.x / 64.0
                            state_spaces[i][offset + 1] = unit.y / 64.0
                            state_spaces[i][offset + 2] = unit.health / 45.0

                            if unit.unit_type == sc2_units.Protoss.Stalker:
                                state_spaces[i][offset + 3] = 1.0
                                state_spaces[i][offset + 4] = 0.0
                                state_spaces[i][offset + 5] = 0.0
                            elif unit.unit_type == sc2_units.Protoss.Sentry:
                                state_spaces[i][offset + 3] = 0.0
                                state_spaces[i][offset + 4] = 1.0
                                state_spaces[i][offset + 5] = 0.0
                            elif unit.unit_type == sc2_units.Protoss.HighTemplar:
                                state_spaces[i][offset + 3] = 0.0
                                state_spaces[i][offset + 4] = 0.0
                                state_spaces[i][offset + 5] = 1.0

                            closest_enemies[unit.tag] = unit.health
                            enemy_index += 1

        if health > 0.0:
            health_map[unit_tags[i].tag] = (health, closest_friends, closest_enemies)

    return state_spaces, health_map, get_centroid(positions), unit_tags

def get_action_position(current_position, heading, magnitude, unit_type):
    agent_distance = ut.unit_data[unit_type]["sight"]
    hypotenuse = magnitude * agent_distance
    degrees = math.radians(heading * 360.0)
    x = hypotenuse * math.cos(degrees)
    y = hypotenuse * math.sin(degrees)

    return (current_position[0] + x, current_position[1] + y)

def get_actions(observations, action_states, units):
    action_list = []

    for i in range(len(action_states)):
        action_state = action_states[i]
        agent_unit = units[i]
        agent_attack = action_state[0]
        agent_heading = action_state[1]
        agent_magnitude = action_state[2]
        agent_ability = action_state[3]
        new_position = get_action_position(agent_unit.position, agent_heading, agent_magnitude, agent_unit.unit_type)

        if agent_ability >=  0.5:
            if agent_unit.unit_type == sc2_units.Protoss.Sentry:
                action_list.append(actions.RAW_FUNCTIONS.Effect_GuardianShield_quick("now", agent_unit.tag))
            elif agent_unit.unit_type == sc2_units.Protoss.HighTemplar:
                action_list.append(actions.RAW_FUNCTIONS.Effect_PsiStorm_pt("now", agent_unit.tag, new_position))
            else:
                if action_state[0] >= 0.5:
                    action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", agent_unit.tag, new_position))
        else:
            if agent_attack >= 0.5:
                action_list.append(actions.RAW_FUNCTIONS.Attack_pt("now", agent_unit.tag, new_position))
            else:
                action_list.append(actions.RAW_FUNCTIONS.Move_pt("now", agent_unit.tag, new_position))

    return action_list

def get_rewards(unit_tags, health_t, enemy_health_t, health_t1, enemy_health_t1):
    rewards = np.zeros(10)

    for i in range(len(unit_tags)):
        tag = unit_tags[i].tag

        if (tag in health_t.keys()) and (tag in health_t1.keys()):
            # Own health first
            prev_health = health_t1[tag][0]
            curr_health = health_t[tag][0]
            friend_delta = prev_health - curr_health
            enemy_delta = 0
            friend_count = 1
            enemy_count = 0

            # Friendly health deltas
            for friend in health_t1[tag][1].keys():
                if friend in health_t[tag][1].keys():
                    friend_delta += health_t1[tag][1][friend] - health_t[tag][1][friend]
                    friend_count += 1

            # Enemy health deltas
            for enemy in health_t1[tag][2].keys():
                if enemy in health_t[tag][2].keys():
                    if (enemy in enemy_health_t1.keys()) and (enemy in enemy_health_t.keys()):
                        enemy_delta += enemy_health_t1[enemy][0] - enemy_health_t[enemy][0]
                        enemy_count += 1

            if enemy_count > 0:
                rewards[i] = (enemy_delta / enemy_count) - (friend_delta / friend_count)
            else:
                rewards[i] = -1 * (friend_delta / friend_count)

    return rewards

def check_end(p1_health, p2_health):
    p1_total = 0.0
    p2_total = 0.0

    for tag in p1_health.keys():
        p1_total += p1_health[tag][0]
    
    for tag in p2_health.keys():
        p2_total += p2_health[tag][0]

    if (p1_total == 0.0) or (p2_total == 0.0):
        return True

    return False

def get_nearest_enemy_position(observations, self_centroid, enemy_unit, enemy_centroid):
    obs_units = observations.observation.raw_units
    position = enemy_centroid
    closest = 100

    for obs_unit in obs_units:
        for enemy in enemy_unit:
            if obs_unit.tag == enemy.tag:
                distance = get_distance(self_centroid, (obs_unit.x, obs_unit.y))

                if distance < closest:
                    closest = distance
                    position = (obs_unit.x, obs_unit.y)

    return position

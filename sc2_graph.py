import dgl
import torch as th
import unit_attack_types as ut

from pysc2.lib import units as sc2_units

def check_battle(units):
  # Find battle
  unit_count = 0
  sides = set()
  battle_found = False
      
  for unit in units:
    if unit.is_on_screen and unit.is_active:
      unit_count += 1
      sides.add(unit.alliance)
      		
  if len(sides) >= 2 and unit_count >= 10:
    battle_found = True
    print("FOUND BATTLE BETWEEN", sides, "WITH UNIT COUNT", unit_count)
    
  return battle_found
  
def attackable(attack_type, target_type):
  return True
  
def in_range(attack_pos, target_pos, attack_range):
  return ((attack_pos - target_pos).pow(2).sum(3).sqrt() >= attack_range)
  
def add_unit_meta_data(graph, units):
  unit_tag = []
  unit_type = []
  unit_alliance = []
  unit_facing = []
  unit_radius = []
  unit_cloaked = []
  unit_pos = []
  unit_health = []
  unit_shield = []
  unit_energy = []
  unit_flying = []
  unit_burrowed = []
  unit_attack_upgrade = []
  unit_armor_upgrade = []
  unit_shield_upgrade = []
  
  for unit in units:
    unit_tag.append(th.tensor(unit.tag))
    unit_type.append(th.tensor(sc2_units.get_unit_type(unit.unit_type)))
    unit_alliance.append(th.tensor(unit.alliance))
    unit_facing.append(th.tensor(unit.facing))
    unit_radius.append(th.tensor(unit.radius))
    unit_cloaked.append(th.tensor(unit.cloak))
    unit_pos.append(th.tensor([unit.pos.x, unit.pos.y, unit.pos.z]))
    unit_health.append(th.tensor(unit.health))
    unit_shield.append(th.tensor(unit.shield))
    unit_energy.append(th.tensor(unit.energy))
    unit_flying.append(th.tensor(unit.is_flying))
    unit_burrowed.append(th.tensor(unit.is_burrowed))
    unit_attack_upgrade.append(th.tensor(unit.attack_upgrade_level)) 
    unit_armor_upgrade.append(th.tensor(unit.armor_upgrade_level)) 
    unit_shield_upgrade.append(th.tensor(unit.shield_upgrade_level))
    
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_type"] = th.stack(unit_type) 
  graph.ndata["unit_alliance"] = th.stack(unit_alliance) 
  graph.ndata["unit_facing"] = th.stack(unit_facing) 
  graph.ndata["unit_radius"] = th.stack(unit_radius) 
  graph.ndata["unit_cloaked"] = th.stack(unit_cloaked) 
  graph.ndata["unit_pos"] = th.stack(unit_pos) 
  graph.ndata["unit_health"] = th.stack(unit_health) 
  graph.ndata["unit_shield"] = th.stack(unit_shield) 
  graph.ndata["unit_energy"] = th.stack(unit_energy) 
  graph.ndata["unit_flying"] = th.stack(unit_flying) 
  graph.ndata["unit_burrowed"] = th.stack(unit_burrowed) 
  graph.ndata["unit_attack_upgrade"] = th.stack(unit_attack_upgrade) 
  graph.ndata["unit_armor_upgrade"] = th.stack(unit_armor_upgrade) 
  graph.ndata["unit_shield_upgrade"] = th.stack(unit_shield_upgrade) 
  
def add_attackable_edges(graph, units, unit_map):
  for i in range(len(units)):
    for j in range(len(units)):
      if (i != j) and (graph.ndata['unit_alliance'][i] != graph.ndata['unit_alliance'][j]):
        # Check if i->j can attack and is in-range to attack
        if (i != j) and attackable(graph.ndata['unit_type'][i], graph.ndata['unit_type'][j]) and in_range(graph.ndata['unit_pos'][i], graph.ndata['unit_radius'][j], graph.ndata['unit_pos'][i]):
          graph.add_edges([i], [j], {'e': th.tensor([1., 0., 0., 0., 0., 0.])})
        # Check if j->i can attack and is in-range to attack
        if (i != j) and attackable(graph.ndata['unit_type'][j], graph.ndata['unit_type'][i]) and in_range(graph.ndata['unit_pos'][j], graph.ndata['unit_radius'][i], graph.ndata['unit_pos'][j]):
          graph.add_edges([j], [i], {'e': th.tensor([1., 0., 0., 0., 0., 0.])})
        
def add_attacking_edges(graph, units, unit_map):
  pass
  
def add_supporting_edges(graph, units, unit_map):
  pass
  
def add_visible_edges(graph, units, unit_map):
  pass
  
def add_grouped_edges(graph, units, unit_map):
  pass
  
def add_transporting_edges(graph, units, unit_map):
  pass

def create_graph(units):
  unit_to_node = {}
  visible_units = []
  
  for unit in units:
    if unit.is_on_screen and unit.is_active:
      unit_to_node[unit.tag] = len(visible_units)
      visible_units.append(unit)
      
  print("Total Units", visible_units)
  print("Mapping", unit_to_node)

  graph = dgl.graph(([], []), num_nodes=len(visible_units))
  
  add_unit_meta_data(graph, visible_units)
  add_attackable_edges(graph, visible_units, unit_to_node)
  add_attacking_edges(graph, visible_units, unit_to_node)
  add_supporting_edges(graph, visible_units, unit_to_node)
  add_visible_edges(graph, visible_units, unit_to_node)
  add_grouped_edges(graph, visible_units, unit_to_node)
  add_transporting_edges(graph, visible_units, unit_to_node)
  
  return graph


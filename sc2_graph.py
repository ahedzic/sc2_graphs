import dgl
import torch as th

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
    unit_tag.append(unit.tag)
    unit_type.append(unit.unit_type)
    unit_alliance.append(unit.alliance)
    unit_facing.append(unit.facing)
    unit_radius.append(unit.radius)
    unit_cloaked.append(unit.cloaked)
    unit_pos.append(th.tensor([unit.unit_type.x, unit.unit_type.y, unit.unit_type.z)])
    unit_health.append(unit.health)
    unit_shield.append(unit.shield)
    unit_energy.append(unit.energy) 
    unit_flying.append(unit.is_flying) 
    unit_burrowed.append(unit.is_burrowed) 
    unit_attack_upgrade.append(unit.attack_upgrade_level) 
    unit_armor_upgrade.append(unit.armor_upgrade_level) 
    unit_shield_upgrade.append(unit.shield_upgrade_level)
    
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  graph.ndata["unit_tag"] = th.stack(unit_tag) 
  
def add_attackable_edges(graph, units, unit_map):
  for i in range(len(units)):
    for j in range(len(units)):
      if (i != j) and (graph.ndata['unit_alliance'][units[i]] != graph.ndata['unit_alliance'][units[j]]):
        # Check if i->j can attack and is in-range to attack
        if (i != j) and attackable(graph.ndata['unit_type'][units[i]], graph.ndata['unit_type'][units[j]]) and in_range(graph.ndata['unit_pos'][units[i]], graph.ndata['unit_radius'][units[j]], graph.ndata['unit_pos'][units[i]]):
          graph.add_edges([i], [j], {'e': th.tensor([1., 0., 0., 0., 0., 0.])})
        # Check if j->i can attack and is in-range to attack
        if (i != j) and attackable(graph.ndata['unit_type'][units[j]], graph.ndata['unit_type'][units[i]]) and in_range(graph.ndata['unit_pos'][units[j]], graph.ndata['unit_radius'][units[i]], graph.ndata['unit_pos'][units[j]]):
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


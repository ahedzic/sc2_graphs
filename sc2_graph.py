import dgl
import torch as th
import unit_data as ut

from pysc2.lib import units as sc2_units

def get_deltas(units, prev_units):
  deltas = {"total_health": 0, "total_shield": 0, "total_energy": 0}
  
  for unit in units:
    for p_unit in prev_units:
      if unit.tag == p_unit.tag:
        h_delta = unit.health - p_unit.health
        s_delta = unit.shield - p_unit.shield
        e_delta = unit.energy - p_unit.energy
        deltas[unit.tag] = {"health": h_delta, "shield": s_delta, "energy": e_delta}
        deltas["total_health"] += h_delta
        deltas["total_shield"] += s_delta
        deltas["total_energy"] += e_delta
    
  return deltas

def check_battle(units, deltas):
  # Find battle
  unit_count = 0
  sides = set()
  battle_found = False
      
  for unit in units:
    if unit.is_on_screen and unit.is_active:
      unit_count += 1
      sides.add(unit.alliance)
      		
  if (len(sides) >= 2) and (unit_count >= 20) and ((deltas["total_health"] + deltas["total_shield"]) < 0):
    battle_found = True
    print("FOUND BATTLE BETWEEN", sides, "WITH UNIT COUNT", unit_count)
    
  return battle_found
  
def get_range(attack_pos, target_pos):
  return (attack_pos - target_pos).pow(2).sum(3).sqrt()
  
def add_unit_meta_data(graph, units, deltas):
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
  unit_health_delta = []
  unit_shield_delta = []
  unit_energy_delta = []
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
    unit_health_delta.append(th.tensor(unit.deltas[unit.tag]["health"]))
    unit_shield_delta.append(th.tensor(unit.deltas[unit.tag]["shield"]))
    unit_energy_delta.append(th.tensor(unit.deltas[unit.tag]["energy"]))
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
  graph.ndata["unit_health_delta"] = th.stack(unit_health_delta) 
  graph.ndata["unit_shield_delta"] = th.stack(unit_shield_delta) 
  graph.ndata["unit_energy_delta"] = th.stack(unit_energy_delta) 
  graph.ndata["unit_flying"] = th.stack(unit_flying) 
  graph.ndata["unit_burrowed"] = th.stack(unit_burrowed) 
  graph.ndata["unit_attack_upgrade"] = th.stack(unit_attack_upgrade) 
  graph.ndata["unit_armor_upgrade"] = th.stack(unit_armor_upgrade) 
  graph.ndata["unit_shield_upgrade"] = th.stack(unit_shield_upgrade) 
  

def add_ability_edges(graph, units, unit_map):
  for i in range(len(units)):
    for order in units.orders:
      if hasattr(order, 'target_unit_tag'):
        edge_type = None
        if order.ability_id in ut.attack_abilities:
          edge_type = th.tensor([0., 1., 0., 0., 0., 0.])
        if order.ability_id in ut.attack_abilities:
          edge_type = th.tensor([0., 1., 0., 0., 0., 0.])
          
        if edge_type != None:
          for j in range(len(units)):
            if graph.ndata["unit_tag"] == order.target_unit_tag:
              graph.add_edges([i], [j], {'e': edge_type})
      else:
        # TODO calculate AoE ability targets
        pass
  
def add_edges(graph, units, unit_map):
  for i in range(len(units)):     
    for j in range(len(units)):
      if i != j:
        # Get range for edge determinations
        target_range = get_range(graph.ndata["unit_pos"][i], graph.ndata["unit_pos"][j])
        
        # Opponent edges first
        if graph.ndata["unit_alliance"][i] != graph.ndata["unit_alliance"][j]:
          # Add "Targetable" edges
          
          # Add "Attacking" edges
          
          # Add "Visible" edges
        else:
          # Add "Supporting" edges
      
    # Add "Transporting" edges  
    if hasattr(units[i], "passengers"):
      for passenger in units[i].passengers:
        graph.add_edges([i], [unit_map[passenger.tag]], {"e": th.tensor([0., 0., 0., 0., 0., 1.])}) 
             

def create_graph(units, deltas):
  unit_to_node = {}
  visible_units = []
  
  for unit in units:
    if unit.is_on_screen and unit.is_active:
      unit_to_node[unit.tag] = len(visible_units)
      visible_units.append(unit)
      
  #print("Total Units", visible_units)
  #print("Mapping", unit_to_node)

  graph = dgl.graph(([], []), num_nodes=len(visible_units))
  
  add_unit_meta_data(graph, visible_units, deltas)
  add_edges(graph, visible_units, unit_to_node)
  
  return graph


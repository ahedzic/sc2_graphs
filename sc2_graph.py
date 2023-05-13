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
  return (attack_pos - target_pos).pow(2).sum().sqrt().item()
  
def add_unit_meta_data(graph, units, deltas):
  unit_tag = []
  unit_type = []
  unit_alliance = []
  unit_facing = []
  unit_radius = []
  unit_display = []
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
    unit_display.append(th.tensor(unit.display_type))
    unit_cloaked.append(th.tensor(unit.cloak))
    unit_pos.append(th.tensor([unit.pos.x, unit.pos.y, unit.pos.z]))
    
    if unit.health_max > 0:
      unit_health.append(th.tensor(unit.health / unit.health_max))
    else:
      unit_health.append(th.tensor(0.0))
    if unit.shield_max > 0:
      unit_shield.append(th.tensor(unit.shield / unit.shield_max))
    else:
      unit_shield.append(th.tensor(0.0))
    if unit.energy_max > 0:
      unit_energy.append(th.tensor(unit.energy / unit.energy_max))
    else:
      unit_energy.append(th.tensor(0.0))
    
    if unit.tag in deltas.keys():
      unit_health_delta.append(th.tensor(deltas[unit.tag]["health"]))
      unit_shield_delta.append(th.tensor(deltas[unit.tag]["shield"]))
      unit_energy_delta.append(th.tensor(deltas[unit.tag]["energy"]))
    else:
      unit_health_delta.append(th.tensor(0.0))
      unit_shield_delta.append(th.tensor(0.0))
      unit_energy_delta.append(th.tensor(0.0))
      
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
  graph.ndata["unit_display"] = th.stack(unit_display) 
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
  # TODO
  graph.ndata["feat"] = None
  
def add_edges(graph, units, unit_map):
  # Index 0 Targeteable
  # Index 1 Normal Attacking
  # Index 2 Single Target Ability Attacking
  # Index 3 AoE Ability Attacking
  # Index 4 Single Target Supporting
  # Index 5 AoE Supporting
  # Index 6 Visible
  # Index 7 Transporting
  for i in range(len(units)):     
    for j in range(len(units)):
      if i != j:
        # Get range for edge determinations
        target_range = get_range(graph.ndata["unit_pos"][i], graph.ndata["unit_pos"][j])
        
        # Opponent edges first
        if graph.ndata["unit_alliance"][i] != graph.ndata["unit_alliance"][j]:
          # Add enemy "Targetable" edges
          attacker_types = ut.unit_data[graph.ndata["unit_type"][i].item()]["attack_type"]
          targeteable = False
          
          if 1 in attacker_types:
            ground_range = ut.unit_data[graph.ndata["unit_type"][i].item()]["attack_range"][1]
            is_ground = len(ut.unit_data[graph.ndata["unit_type"][j].item()]["unit_type"].intersection(set([1, 3, 8]))) > 0
            
            if (target_range <= ground_range) and is_ground:
              targeteable = True
              
          if 2 in attacker_types: 
            air_range = ut.unit_data[graph.ndata["unit_type"][i].item()]["attack_range"][2]
            is_air = len(ut.unit_data[graph.ndata["unit_type"][j].item()]["unit_type"].intersection(set([2, 4, 9]))) > 0
            
            if (target_range <= air_range) and is_air:
              targeteable = True
          
          if targeteable:
            graph.add_edges([i], [j], {"e": th.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])})
          
          # Add "Attacking" edges
          for order in units[i].orders:
            if order.ability_id in ut.attack_commands:
              if hasattr(order, 'target_unit_tag') and (order.target_unit_tag in unit_map.keys()):
                # Normal attacking
                graph.add_edges([i], [unit_map[order.target_unit_tag]], {"e": th.tensor([[0., 1., 0., 0., 0., 0., 0., 0.]])})
            if order.ability_id in ut.attack_abilities:
              if hasattr(order, 'target_unit_tag') and (order.target_unit_tag in unit_map.keys()):
                # Single target ability attacking
                graph.add_edges([i], [unit_map[order.target_unit_tag]], {"e": th.tensor([[0., 0., 1., 0., 0., 0., 0., 0.]])})
              if hasattr(order, 'target_pos') and (order.target_unit_tag in unit_map.keys()):
                # AoE ability attacking
                target_pos = th.tensor([order.target_pos.x, order.target_pos.y, 0.0])
                target_range = get_range(target_pos, graph.ndata["unit_pos"][j])
                
                if target_range <= ut.attack_ability_aoe_radius[order.ability_id]:
                  graph.add_edges([i], [j], {"e": th.tensor([[0., 0., 0., 1., 0., 0., 0., 0.]])})
          
          # Add "Visible" edges
          # Visible = 1, Snapshot = 2, Hidden = 3
          # Cloaked = 1, CloakedDetected = 2, NotCloaked = 3
          if (target_range <= ut.unit_data[graph.ndata["unit_type"][i].item()]["sight"]) and (graph.ndata["unit_display"][j] == 1) and (graph.ndata["unit_cloaked"][j] != 1):
            graph.add_edges([i], [j], {"e": th.tensor([[0., 0., 0., 0., 0., 0., 1., 0.]])})
        else:
          # Add friendly "Targetable" edges
          if 1 in ut.unit_data[graph.ndata["unit_type"][i].item()]["support_type"]:
            support_range = ut.unit_data[graph.ndata["unit_type"][i].item()]["support_range"][1]
            
            if (target_range <= support_range):
              graph.add_edges([i], [j], {"e": th.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])})
           
          # Add "Supporting" edges   
          # AoE vision supporting
          if 6 in ut.unit_data[graph.ndata["unit_type"][i].item()]["unit_type"]:
            vision_range = ut.unit_data[graph.ndata["unit_type"][i].item()]["sight"][1]
            
            if (target_range <= vision_range):
              graph.add_edges([i], [j], {"e": th.tensor([[0., 0., 0., 0., 0., 1., 0., 0.]])})
          
          for order in units[i].orders:
            if order.ability_id in ut.support_abilities:
              if hasattr(order, 'target_unit_tag') and (order.target_unit_tag in unit_map.keys()):
                # Single target ability supporting
                graph.add_edges([i], [unit_map[order.target_unit_tag]], {"e": th.tensor([[0., 0., 0., 0., 1., 0., 0., 0.]])})
              if hasattr(order, 'target_pos') and (order.target_unit_tag in unit_map.keys()):
                # AoE ability supporting
                target_pos = th.tensor([order.target_pos.x, order.target_pos.y, 0.0])
                target_range = get_range(target_pos, graph.ndata["unit_pos"][j])
                
                if target_range <= ut.support_ability_aoe_radius[order.ability_id]:
                  graph.add_edges([i], [j], {"e": th.tensor([[0., 0., 0., 0., 0., 1., 0., 0.]])})
      
    # Add "Transporting" edges  
    if hasattr(units[i], "passengers"):
      for passenger in units[i].passengers:
        graph.add_edges([i], [unit_map[passenger.tag]], {"e": th.tensor([[0., 0., 0., 0., 0., 0., 0., 1.]])}) 
             

def create_graph(units, deltas):
  unit_to_node = {}
  visible_units = []
  
  for unit in units:
    if unit.is_on_screen and unit.is_active and (unit.health > 0.0):
      unit_to_node[unit.tag] = len(visible_units)
      visible_units.append(unit)
      
  #print("Total Units", visible_units)
  #print("Mapping", unit_to_node)

  graph = dgl.graph(([], []), num_nodes=len(visible_units))
  
  add_unit_meta_data(graph, visible_units, deltas)
  add_edges(graph, visible_units, unit_to_node)
  
  return graph


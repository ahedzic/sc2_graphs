import networkx as nx
import dgl
import matplotlib.pyplot as plt
import os
import unit_data as ut

def create_graph_image(graph_images_dir, graph, graph_name):
  graph_nx = dgl.to_networkx(graph, node_attrs=["unit_tag", "unit_type", "unit_alliance", "unit_pos"], edge_attrs=["e"])
       	  
  ground_nodes = {"nodes": [], "nodes_pos": {}, "colors": []}
  flying_nodes = {"nodes": [], "nodes_pos": {}, "colors": []}
  static_nodes = {"nodes": [], "nodes_pos": {}, "colors": []}
  alliance_colors = {}
  edge_colors = []
  node_pos = {}
  first = True
  
  for edge in range(len(graph_nx.edges())):
    color = (0.0, 0.0, 0.0, 1.0)
    
    if graph.edata["e"][edge][0].item() > 0:
      color = (0.0, 1.0, 0.0, 1.0)
    if graph.edata["e"][edge][1].item() > 0:
      color = (1.0, 0.0, 0.0, 1.0)
    if graph.edata["e"][edge][2].item() > 0:
      color = (0.0, 0.0, 1.0, 1.0)
    if graph.edata["e"][edge][3].item() > 0:
      color = (1.0, 1.0, 0.0, 1.0)
    if graph.edata["e"][edge][4].item() > 0:
      color = (1.0, 0.65, 0.0, 1.0)
      
    edge_colors.append(color)
       	  
  for node in list(graph_nx):
    alliance = graph.ndata["unit_alliance"][node].item()
    unit_type = ut.unit_data[graph.ndata["unit_type"][node].item()]["unit_type"]
       	    
    if alliance not in alliance_colors.keys():
      if first:
        alliance_colors[alliance] = (1.0, 0.0, 0.0)
        first = False
      else:
        alliance_colors[alliance] = (0.0, 0.0, 1.0)
       	    
    pos = graph.ndata["unit_pos"][node]
    color = (alliance_colors[alliance][0], alliance_colors[alliance][1], alliance_colors[alliance][2], graph.ndata["unit_health"][node].item())
    node_pos[node] = [pos[0].item(), pos[1].item()]
    
    if len(unit_type.intersection(set([1, 3]))) > 0:
      ground_nodes["nodes"].append(node)
      ground_nodes["nodes_pos"][node] = [pos[0].item(), pos[1].item()]
      ground_nodes["colors"].append(color)
    if len(unit_type.intersection(set([2, 4]))) > 0:
      flying_nodes["nodes"].append(node)
      flying_nodes["nodes_pos"][node] = [pos[0].item(), pos[1].item()]
      flying_nodes["colors"].append(color)
    if len(unit_type.intersection(set([8, 9]))) > 0:
      static_nodes["nodes"].append(node)
      static_nodes["nodes_pos"][node] = [pos[0].item(), pos[1].item()]
      static_nodes["colors"].append(color)

  nx.draw_networkx_nodes(graph_nx, ground_nodes["nodes_pos"], nodelist=ground_nodes["nodes"], node_color=ground_nodes["colors"], node_size=20, node_shape="v")
  nx.draw_networkx_nodes(graph_nx, flying_nodes["nodes_pos"], nodelist=flying_nodes["nodes"], node_color=flying_nodes["colors"], node_size=20, node_shape="^")
  nx.draw_networkx_nodes(graph_nx, static_nodes["nodes_pos"], nodelist=static_nodes["nodes"], node_color=static_nodes["colors"], node_size=20, node_shape="o")
  nx.draw_networkx_edges(graph_nx, node_pos, edge_color=edge_colors, arrowsize=5)
  graph_image_path = os.path.join(graph_images_dir, graph_name+".png")
  plt.savefig(graph_image_path, format="PNG")

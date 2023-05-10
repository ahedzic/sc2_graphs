import networkx as nx
import dgl
import matplotlib.pyplot as plt
import os

def create_graph_image(graph_images_dir, graph, graph_name):
  graph_nx = dgl.to_networkx(graph, node_attrs=["unit_tag", "unit_type", "unit_alliance", "unit_pos"], edge_attrs=["e"])
       	  
  draw_nodes = {"nodes_pos": {}, "colors": []}
  alliance_colors = {}
  first = True
       	  
  for node in list(graph_nx):
    alliance = graph.ndata["unit_alliance"][node].item()
       	    
    if alliance not in alliance_colors.keys():
      if first:
        alliance_colors[alliance] = (1.0, 0.0, 0.0)
        first = False
      else:
        alliance_colors[alliance] = (0.0, 0.0, 1.0)
       	    
    pos = graph.ndata["unit_pos"][node]
    draw_nodes["nodes_pos"][node] = [pos[0].item(), pos[1].item()]
    color = (alliance_colors[alliance][0], alliance_colors[alliance][1], alliance_colors[alliance][2], graph.ndata["unit_health"][node].item())
    draw_nodes["colors"].append(color)

  nx.draw_networkx(graph_nx, draw_nodes["nodes_pos"], nodelist=list(graph_nx), node_color=draw_nodes["colors"], node_size=20, arrowsize=1, with_labels=False)
  graph_image_path = os.path.join(graph_images_dir, graph_name+".png")
  plt.savefig(graph_image_path, format="PNG")

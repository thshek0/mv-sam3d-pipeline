"""Articulated part container; shared definitions supply its geometry and joints."""
import numpy as np
import trimesh

def translation(vector):
 result=np.eye(4);result[:3,3]=vector;return result

class Model:
 def __init__(self):
  self.parts={};self.info={};self.links={}

 def add(self,name,mesh,group,color,evidence,hardware=False):
  assert mesh.is_volume and mesh.is_convex and np.isfinite(mesh.vertices).all(),name
  if color is not None:mesh.visual.vertex_colors=color
  self.parts[name]=mesh
  self.info[name]=dict(group=group,evidence=evidence,hinge_hardware=hardware,triangles=len(mesh.faces))

 def relative(self,name,state):
  link=self.links[name];transform=translation(link['origin_m']);value=state.get(name,0)
  if link['type']=='revolute':transform=transform@trimesh.transformations.rotation_matrix(np.deg2rad(value),link['axis'])
  elif link['type']=='prismatic':transform=transform@translation(np.array(link['axis'])*value)
  return transform

 def transforms(self,state):
  result={'world':np.eye(4)}
  for name,link in self.links.items():result[name]=result[link['parent']]@self.relative(name,state)
  return result

 def scene(self,state=None,tower_only=False,selected=None):
  state=state or {};scene=trimesh.Scene(base_frame='world')
  for name,link in self.links.items():scene.graph.update(frame_to=name,frame_from=link['parent'],matrix=self.relative(name,state))
  for name,mesh in self.parts.items():
   if tower_only and name in ('right_housing','rear_housing','base_support'):continue
   if selected is not None and name not in selected:continue
   scene.add_geometry(mesh.copy(),node_name=name,geom_name=name,parent_node_name=self.info[name]['group'])
  scene.metadata=dict(units='metres',frame='X right, Y rear, Z up',joint_positions=state,kinematics=self.links,
   authoring='Shared CAD/simulation definition',physical_contact_accuracy_validated=False)
  return scene

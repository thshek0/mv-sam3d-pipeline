"""Small shared solid/parameter representation for CAD and convex simulation parts.

Expressions are evaluated with a restricted AST interpreter, never Python eval.
The same expressions are emitted as native OpenSCAD primitives. Meshes retain
separate convex pieces so an assembly's openings are not filled by convexification.
"""
import ast
from dataclasses import dataclass
import json
import math
import operator
import re
import numpy as np
import trimesh


def evaluate(expression, values):
    ops={ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
         ast.Div:operator.truediv, ast.Lt:operator.lt, ast.LtE:operator.le,
         ast.Gt:operator.gt, ast.GtE:operator.ge, ast.Eq:operator.eq}
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
        if isinstance(node,ast.Name):return values[node.id]
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('min','max','abs'):
            return {'min':min,'max':max,'abs':abs}[node.func.id](*[visit(x) for x in node.args])
        if isinstance(node,ast.BinOp) and type(node.op) in ops:return ops[type(node.op)](visit(node.left),visit(node.right))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -visit(node.operand)
        if isinstance(node,ast.Compare):
            terms=[node.left,*node.comparators]
            return all(ops[type(op)](visit(a),visit(b)) for a,op,b in zip(terms,node.ops,terms[1:]))
        if isinstance(node,ast.BoolOp) and isinstance(node.op,(ast.And,ast.Or)):
            return (all if isinstance(node.op,ast.And) else any)(visit(x) for x in node.values)
        raise ValueError('Unsupported parameter expression: '+ast.dump(node))
    return visit(ast.parse(expression,mode='eval').body)


@dataclass(frozen=True)
class Expr:
    code: str
    def _op(self, other, op):return Expr(f'({self.code}{op}{scad(other)})')
    def __add__(self, other):return self._op(other,'+')
    def __sub__(self, other):return self._op(other,'-')
    def __mul__(self, other):return self._op(other,'*')
    def __truediv__(self, other):return self._op(other,'/')
    def __neg__(self):return Expr(f'(-{self.code})')
    def __radd__(self, other):return Expr(scad(other)).__add__(self)
    def __rsub__(self, other):return Expr(scad(other)).__sub__(self)
    def __rmul__(self, other):return self.__mul__(other)


def scad(value):
    if isinstance(value,Expr):return value.code
    if isinstance(value,(list,tuple)):return '['+','.join(scad(x) for x in value)+']'
    return json.dumps(value)


def resolve(value, values):
    if isinstance(value,Expr):return evaluate(value.code,values)
    if isinstance(value,(list,tuple)):return [resolve(x,values) for x in value]
    return value


def solid_scad(kind, args):
    if kind == 'subtract':
        base, cutters = args
        return 'difference() { '+solid_scad(*base)+' '+ ' '.join(solid_scad(*c) for c in cutters)+' }'
    return kind+'('+','.join(scad(x) for x in args)+');'


def convex_difference(base, cutters):
    """Partition a convex solid around convex cutters; retain real empty volumes.

    CAD performs native difference(). These nonoverlapping convex pieces are
    derived for simulation; they do not define the CAD shell as adjoining walls.
    Coordinates are millimetres to keep clipping tolerances meaningful.
    """
    def primitive(node):
        kind,args=node
        if kind=='block':
            lo,hi=np.asarray(args[0]),np.asarray(args[1])
            if np.any(hi<=lo):raise ValueError('Non-positive Boolean block')
            return trimesh.creation.box(hi-lo,transform=trimesh.transformations.translation_matrix((lo+hi)/2))
        if kind=='xy_prism':
            points,z0,z1=args;vertices=[[x,y,z] for x,y in points for z in (z0,z1)]
        elif kind=='yz_prism':
            points,x0,x1=args;vertices=[[x,y,z] for x in (x0,x1) for y,z in points]
        else:raise ValueError('Unsupported convex Boolean primitive: '+kind)
        return trimesh.convex.convex_hull(vertices)
    def clip(mesh, normal, offset, keep_inside):
        vertices=mesh.vertices;dist=vertices@normal-offset
        if not keep_inside:dist=-dist
        if dist.max()<=1e-8:return mesh
        if dist.min()>=-1e-8:return None
        points=list(vertices[dist<=1e-8])
        for i,j in mesh.edges_unique:
            if dist[i]*dist[j]<-1e-16:
                points.append(vertices[i]+dist[i]/(dist[i]-dist[j])*(vertices[j]-vertices[i]))
        points=np.unique(np.round(points,9),axis=0)
        if len(points)<4:return None
        # Merge nearly coplanar facets within 1e-6 mm. Unmerged numerical
        # slivers otherwise become invalid after conversion to metre units.
        hull=trimesh.convex.convex_hull(points,qhull_options='Qt Qx C-0.000001')
        return hull if hull.volume>1e-8 else None
    pieces=[primitive(base)]
    for cutter in cutters:
        tool=primitive(cutter)
        raw_planes=np.column_stack((tool.face_normals,np.einsum('ij,ij->i',tool.face_normals,tool.triangles_center)))
        _,indices=np.unique(np.round(raw_planes,9),axis=0,return_index=True)
        planes=raw_planes[indices]
        result=[]
        for piece in pieces:
            inside=piece
            for plane in planes:
                outside=clip(inside,plane[:3],plane[3],False)
                if outside is not None:result.append(outside)
                inside=clip(inside,plane[:3],plane[3],True)
                if inside is None:break
            # Remaining inside is the material removed by this cutter.
        pieces=result
    return pieces


def read_parameter_include(path, expected):
    """Import only named finite numeric assignments; reject code or unknown names."""
    text=re.sub(r'//[^\n]*','',path.read_text())
    result={}
    for statement in text.split(';'):
        if not statement.strip():continue
        match=re.fullmatch(r'\s*([a-zA-Z_]\w*)\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*',statement)
        if not match:raise ValueError('Only numeric parameter assignments can be imported: '+statement.strip()[:100])
        key,raw=match.groups();value=float(raw)
        if key not in expected or key in result or not math.isfinite(value):raise ValueError('Unknown, duplicate or non-finite parameter: '+key)
        result[key]=value
    if set(result)!=set(expected):raise ValueError('Parameter include is incomplete: '+', '.join(sorted(set(expected)-set(result))))
    return result


@dataclass
class Part:
    name: str
    kind: str
    args: list
    group: str
    tags: tuple
    color: list
    local: bool = False
    hardware: bool = False


class Assembly:
    def __init__(self, values, constraints, travel=0, axis=(0,1,0), origin=(0,0,0), joint_type='prismatic'):
        self.values=values;self.constraints=constraints;self.travel=travel;self.axis=axis;self.origin=origin;self.joint_type=joint_type;self.parts=[]
        self.links={};self.states={}
        for key,value in values.items():
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):raise ValueError('Invalid parameter '+key)
        for expression,message in constraints:
            if not evaluate(expression,values):raise ValueError(message)

    def add(self,name,kind,args,group='base_link',tags=('body',),color=(224,224,215,255),local=False,hardware=False):
        self.parts.append(Part(name,kind,args,group,tags,list(color),local,hardware))

    def link(self,name,parent,kind,origin=(0,0,0),axis=(0,0,1),limit=0):
        self.links[name]=dict(parent=parent,type=kind,origin=origin,axis=axis,limit=limit)

    def model(self,p):
        from .geometry import Model
        m=Model(p)
        p.update(joint_type=self.joint_type,joint_origin_m=(np.array(resolve(self.origin,self.values))/1000).tolist(),joint_axis=list(self.axis),
                 lid_preview_stroke_m=resolve(self.travel,self.values)/1000)
        for part in self.parts:
            args=resolve(part.args,self.values);kw=dict(group=part.group,color=part.color,evidence='Shared CAD/simulation definition; unmeasured local dimensions remain estimates')
            if part.kind in ('block','rounded_x'):
                lo,hi=np.array(args[0])/1000,np.array(args[1])/1000
                if part.kind=='block':m.box(part.name,lo,hi,**kw)
                else:m.rounded(part.name,lo,hi,r=args[2]/1000,**kw)
            elif part.kind=='rod':
                center,d,length,axis=args
                q=trimesh.creation.cylinder(radius=d/2000,height=length/1000,sections=16)
                q.apply_transform(trimesh.geometry.align_vectors([0,0,1],{'x':[1,0,0],'y':[0,1,0],'z':[0,0,1]}[axis]))
                q.apply_translation(np.array(center)/1000);m.add(part.name,q,**kw)
            elif part.kind=='xy_prism':
                points,z0,z1=args
                q=trimesh.convex.convex_hull(np.array([[x,y,z] for x,y in points for z in (z0,z1)])/1000)
                m.add(part.name,q,**kw)
            else:
                for name,q in solid_meshes(part.name,part.kind,args):m.add(name,q,**kw)
        if self.links:
            from .qy.kinematics import Model as Articulated
            articulated=Articulated()
            articulated.links={}
            for name,link in self.links.items():
                kind=link['type'];limit=resolve(link['limit'],self.values)
                articulated.links[name]=dict(parent=link['parent'],type=kind,
                    origin_m=(np.array(resolve(link['origin'],self.values))/1000).tolist(),axis=list(link['axis']),
                    preview_range=[0,limit/1000 if kind=='prismatic' else limit],
                    position_units='metres' if kind=='prismatic' else 'degrees',measured_limits=None)
            zeros=articulated.transforms({})
            for part in self.parts:
                names=[n for n in m.parts if n==part.name or n.startswith(part.name+'__')]
                for name in names:
                    q=m.parts[name].copy()
                    if not part.local:q.apply_transform(np.linalg.inv(zeros[part.group]))
                    articulated.add(name,q,part.group,part.color,m.evidence[name],part.hardware)
            return articulated
        return m

    def export_scad(self,path):
        lines=['// GENERATED from the shared part/joint definition. Do not edit this file.',
               '// Edit parameters_native.scad; import its values with build --from-cad-parameters.',
               'include <parameters_native.scad>','use <common.scad>',
               '/* [Motion] */',
               'pose="closed"; // ['+','.join(dict.fromkeys(['closed','open',*self.states,'custom']))+']',
               'animate_motion=false;',
               '// Custom position: 0 closed, 1 open. Animation opens then closes.',
               'opening=0; // [0:0.01:1]',
               'part="all"; // all, body, cover, holder, socket, or an individual part name',
               ]
        for name,link in self.links.items():
            if link['type']!='fixed':lines.append(f'{name}_opening=0; // [0:0.01:1]')
        lines += ['/* [Hidden] */', 'function motion_fraction()=1-abs(2*$t-1);',
                  f'function cover_position_mm()=(animate_motion ? motion_fraction() : pose=="custom" ? opening : pose=="open" ? 1:0)*({scad(self.travel)});', '$fn=64;']
        origins={'world':[0,0,0]}
        if self.links:
            for name,link in self.links.items():
                origin=link['origin'];parent=link['parent'];origins[name]=[a+b for a,b in zip(origins[parent],origin)]
                choices='0'
                for state,positions in self.states.items():
                    if name in positions:choices=f'pose=="{state}" ? {scad(positions[name])}:({choices})'
                if link['type']!='fixed':
                    choices=f'(animate_motion ? motion_fraction() : pose=="custom" ? {name}_opening : -1)>=0 ? (animate_motion ? motion_fraction() : {name}_opening)*({scad(link["limit"])}) : ({choices})'
                lines.append(f'function pos_{name}()={choices};')
                transform=''
                if link['type']=='prismatic':transform=f'translate({scad(link["axis"])}*pos_{name}()) '
                if link['type']=='revolute':transform=f'rotate(a=pos_{name}(),v={scad(link["axis"])}) '
                chain='' if parent=='world' else f'link_{parent}() '
                lines.append(f'module link_{name}() {{ {chain}translate({scad(origin)}) {transform}children(); }}')
        for expression,message in self.constraints:
            expression=expression.replace(' and ',' && ').replace(' or ',' || ')
            lines.append(f'assert({expression},{json.dumps(message)});')
        for part in self.parts:
            selected=' || '.join(f'part=={json.dumps(x)}' for x in ('all',part.name,*part.tags))
            transform=''
            if self.links:
                transform=f'link_{part.group}() '
                if not part.local:transform+=f'translate(-{scad(origins[part.group])}) '
            elif part.group=='cover':
                if self.joint_type=='prismatic':transform=f'translate({scad(self.axis)}*cover_position_mm()) '
                else:transform=f'translate({scad(self.origin)}) rotate(a=cover_position_mm(),v={scad(self.axis)}) translate(-{scad(self.origin)}) '
            call=solid_scad(part.kind,part.args)
            # Evaluate static cuts before applying moving transforms. OpenCSG can
            # otherwise display phantom cutter faces as doors animate (F5).
            if part.kind=='subtract':call='render(convexity=10) '+call
            color=scad([x/255 for x in part.color[:3]])
            lines.append(f'if({selected}) color({color}) {transform}{call}')
        path.write_text('\n'.join(lines)+'\n')


def solid_meshes(name,kind,args):
    """Convex decomposition for additional native solids. Input millimetres."""
    from .openings import circle_plate
    def scaled(q):q.apply_scale(.001);return q
    if kind=='subtract':
        return [(f'{name}__{i:02}',scaled(q)) for i,q in enumerate(convex_difference(*args))]
    if kind=='yz_prism':
        points,x0,x1=args
        return [(name,scaled(trimesh.convex.convex_hull([[x,y,z] for x in (x0,x1) for y,z in points])))]
    if kind=='plate_hole':
        lo,hi,center,d,z0,z1=args
        return [(f'{name}__{i:02}',q) for i,q in enumerate(circle_plate(np.array(lo)/1000,np.array(hi)/1000,np.array(center)/1000,d/2000,z0/1000,z1/1000))]
    if kind=='plate_holes':
        from .openings import circular_holes_plate
        return [(f'{name}__{i:03}',scaled(q)) for i,q in enumerate(circular_holes_plate(*args))]
    if kind=='radial_profile':
        center,profile,axis,segments=args;parts=[]
        for i in range(segments):
            vertices=[[r*np.cos(a),r*np.sin(a),z] for a in (2*np.pi*i/segments,2*np.pi*(i+1)/segments) for r,z in profile]
            q=trimesh.convex.convex_hull(vertices)
            q.apply_transform(trimesh.geometry.align_vectors([0,0,1],{'x':[1,0,0],'y':[0,1,0],'z':[0,0,1]}[axis]))
            q.apply_translation(center);parts.append((f'{name}__{i:02}',scaled(q)))
        return parts
    if kind=='quarter_disc':
        edge,r,z0,z1,side=args
        points=[[edge,0]]+[[edge+side*r*np.cos(a),-r*np.sin(a)] for a in np.linspace(0,np.pi/2,17)]
        return [(name,scaled(trimesh.convex.convex_hull([[x,y,z] for x,y in points for z in (z0,z1)])))]
    if kind=='ellipsoid':
        center,radii=args;q=trimesh.creation.icosphere(subdivisions=1);q.apply_scale(radii);q.apply_translation(center)
        return [(name,scaled(q))]
    if kind=='capsule_y':
        center,width,height,depth=args;r=width/2;span=(height-width)/2
        pts=[[x,y,z] for y in (-depth/2,depth/2) for sign in (-1,1) for x,z in
             [[r*np.cos(a),sign*span+r*np.sin(a)] for a in np.linspace(0,2*np.pi,13)]]
        q=trimesh.convex.convex_hull(pts);q.apply_translation(center)
        return [(name,scaled(q))]
    raise ValueError('Unsupported solid '+kind)

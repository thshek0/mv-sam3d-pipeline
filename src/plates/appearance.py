"""Reproducible plate artwork and UV mapping, independent of collision geometry."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import trimesh
from trimesh.visual.material import PBRMaterial

ROOT = Path(__file__).resolve().parents[2]


def pattern_texture(recipe, size=1024):
    """Render the reference-derived radial dot/stem design, with a plain underside tile."""
    kind = recipe.get('pattern_type', 'radial_dots')
    if kind in ('rim_flutes', 'leaf_wreath'):
        return rim_texture(recipe, size)
    if kind != 'radial_dots':
        raise ValueError(f'Unknown plate pattern type: {kind}')
    scale = 2
    n = size*scale
    image = Image.new('RGB', (n*2, n), tuple(recipe['background_rgb']))
    draw = ImageDraw.Draw(image)
    center = np.array([n/2, n/2])
    radius = n*.48  # Margin prevents the atlas seam bleeding onto the rim.
    for ring in recipe['rings']:
        for index in range(recipe['motifs_per_ring']):
            angle = np.deg2rad(ring['phase_degrees'] + index*360/recipe['motifs_per_ring'])
            direction = np.array([np.cos(angle), -np.sin(angle)])
            colour = recipe['blue_rgb'] if index % 2 == 0 else recipe['gold_rgb']
            outline = recipe['blue_outline_rgb'] if index % 2 == 0 else recipe['gold_outline_rgb']
            end = center + direction*ring['radius']*radius
            start = center + direction*(ring['radius']-ring['stem_length'])*radius
            draw.line([tuple(start), tuple(end)], fill=tuple(colour),
                      width=max(2, round(ring['stem_width']*radius)))
            dot = ring['dot_radius']*radius
            draw.ellipse([tuple(end-dot), tuple(end+dot)], fill=tuple(colour),
                         outline=tuple(outline), width=max(1, round(radius*.003)))
    # Artwork beyond the plate circle is unused; the right atlas tile is plain.
    draw.rectangle((n, 0, n*2, n), fill=tuple(recipe['background_rgb']))
    return image.resize((size*2, size), Image.Resampling.LANCZOS)


def rim_texture(recipe, size=1024):
    """Bake a reference-derived rim motif into colour; not physical relief or a normal map."""
    n = size*2
    image = Image.new('RGB', (2*n, n), tuple(recipe['background_rgb']))
    draw = ImageDraw.Draw(image)
    center, radius = np.array([n/2, n/2]), n*.48
    highlight, shadow = tuple(recipe['highlight_rgb']), tuple(recipe['shadow_rgb'])

    def polar(r, angle):
        return center + radius*r*np.array([np.cos(angle), -np.sin(angle)])

    def leaf(base, tip, width):
        delta = tip-base
        perpendicular = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
        t = np.linspace(0, 1, 33)
        mid = base + t[:, None]*delta
        offset = np.sin(np.pi*t)[:, None]*perpendicular*width*radius
        polygon = np.vstack([mid+offset, (mid-offset)[::-1]])
        # A subtle light edge and darker body convey the photographed moulded motif.
        draw.polygon([tuple(p) for p in polygon], fill=shadow)
        draw.line([tuple(p) for p in (mid+offset)], fill=highlight, width=max(2, round(radius*.003)))

    if recipe['pattern_type'] == 'rim_flutes':
        for i in range(recipe['motif_count']):
            angle = i*2*np.pi/recipe['motif_count']
            inner = polar(recipe['inner_radius'], angle)
            outer = polar(recipe['outer_radius'], angle)
            tangent = np.array([np.sin(angle), np.cos(angle)])
            shift = tangent*radius*.003
            draw.line([tuple(inner+shift), tuple(outer+shift)], fill=shadow, width=max(2, round(radius*.008)))
            draw.line([tuple(inner), tuple(outer)], fill=highlight, width=max(2, round(radius*.004)))
            # The rim has small loop/teardrop ends, rather than simple stripes.
            theta = np.linspace(0, 2*np.pi, 40)
            r = recipe['outer_radius'] + recipe['loop_length']*(1-np.cos(theta))/2
            a = angle + recipe['loop_half_angle']*np.sin(theta)
            loop = [tuple(polar(rr, aa)) for rr, aa in zip(r, a)]
            draw.line(loop, fill=shadow, width=max(2, round(radius*.006)))
            draw.line(loop, fill=highlight, width=max(2, round(radius*.003)))
    else:
        for i in range(recipe['motif_count']):
            angle = i*2*np.pi/recipe['motif_count']
            base = polar(recipe['stem_radius'], angle)
            leaf(base, polar(recipe['outer_radius'], angle+recipe['leaf_sweep']), recipe['leaf_half_width'])
            leaf(base, polar(recipe['inner_radius'], angle+recipe['leaf_sweep']), recipe['leaf_half_width']*.9)
    draw.rectangle((n, 0, n*2, n), fill=tuple(recipe['background_rgb']))
    return image.resize((size*2, size), Image.Resampling.LANCZOS)


def apply_appearance(mesh, name, params, destination):
    """Keep triangle coordinates/order; duplicate only vertices at the UV seam."""
    if 'appearance_recipe' not in params:
        mesh.visual = trimesh.visual.TextureVisuals(material=PBRMaterial(
            name=name+'_plain', baseColorFactor=params['rgba'], metallicFactor=0, roughnessFactor=.5))
        return mesh, {'rgba': params['rgba'], 'images': 0, 'textures': 0,
                      'physical_rim_detail': False, 'material': 'Uniform plain material'}
    recipe_path = ROOT/params['appearance_recipe']
    recipe = json.loads(recipe_path.read_text())
    texture = pattern_texture(recipe)
    texture_path = Path(destination)/(name+'_pattern.png')
    texture.save(texture_path)
    upper = mesh.face_normals[:, 2] > 0
    keys = np.c_[mesh.faces.ravel(), np.repeat(upper, 3)]
    keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    indices = keys[:, 0]
    result = trimesh.Trimesh(vertices=mesh.vertices[indices], faces=inverse.reshape(-1, 3),
                             vertex_normals=mesh.vertex_normals[indices], process=False)
    uv = np.tile([.75, .5], (len(indices), 1))
    top = keys[:, 1].astype(bool)
    radius = np.linalg.norm(mesh.vertices[:, :2], axis=1).max()
    xy = result.vertices[top, :2]/radius
    uv[top] = np.c_[.25 + xy[:, 0]*.24, .5 + xy[:, 1]*.48]
    result.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(
        name=name+'_pattern', baseColorFactor=[255]*4, baseColorTexture=texture,
        metallicFactor=0, roughnessFactor=.5))
    np.testing.assert_array_equal(result.triangles, mesh.triangles)
    return result, {'rgba': params['rgba'], 'images': 1, 'textures': 1,
                    'physical_rim_detail': False,
                    'material': recipe.get('material_description', 'UV-mapped blue/gold radial print; plain underside'),
                    'texture': texture_path.name, 'recipe': str(recipe_path.relative_to(ROOT)),
                    'pattern_provenance': recipe['provenance'],
                    'geometry_preserved': True, 'uv_seam_vertices': len(result.vertices)-len(mesh.vertices)}

"""Refresh plate review pages and their local-file manifest."""
from pathlib import Path
import hashlib
import json
import sys
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.output_layout import navigation
NAMES=('plate_maehata_pop_blue_23cm','plate_strimmig_pale_grey_green_21cm','plate_paradisisk_off_white_20cm')


def run():
    review=ROOT/'output/comparisons/plates';review.mkdir(parents=True,exist_ok=True)
    files={};summaries=[];links=[]
    board=Image.new('RGB',(1600,1500),'white');draw=ImageDraw.Draw(board)
    for i,name in enumerate(NAMES):
        folder=ROOT/'output/objects/hestia-rl'/name/'codex/products'
        report=json.loads((folder/'validation.json').read_text());check=report['mesh_checks']
        summaries.append({'name':name,'triangles':check['triangles'],'extents_mm':check['extents_mm'],
                          'appearance':report['appearance_check'],
                          'collision':'Family ZIP includes separate convex pieces and a textured MuJoCo assembly'})
        relative=folder.relative_to(ROOT/'output').as_posix()
        links.append(f'<li><a href="../../{relative}/index.html">{name}</a></li>')
        for path in sorted(folder.rglob('*')):
            if path.is_file():files[path.relative_to(ROOT/'output').as_posix()]=path
        with Image.open(folder/'review.png') as im:board.paste(im,(0,i*500+45))
        draw.text((20,i*500+12),f'{name} | {check["triangles"]:,} triangles',fill='black')
    board.save(review/'preview.png')
    manifest={'schema_version':1,'path_base':'output root, or extracted ZIP root','models':summaries,
              'units':{'glb':'metres','obj':'metres','stl':'millimetres','scad':'millimetres'},
              'files':{name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}}
    (review/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    readme='Current simplified plate products. Open index.html. Local GLB/OBJ use metres; local STL/SCAD/CSV use millimetres. The family download uses metres for both OBJ and STL plate files, with separate convex collision pieces and textured MuJoCo XML. All three plate decorations are reconstructed from reference photographs; see each validation.json for provenance and approximation limits. Rim relief on the green and off-white plates is represented by baked colour, not physical embossing. Rebuild with src/plates/rebuild.py, then render.py and review.py. ZIP downloads are snapshots and are not rebuilt by this review command. Parameters and profile CSVs remain editable inputs; pattern recipes are in dataset/hestia-rl/<plate>/appearance/.\n'
    (review/'README.txt').write_text(readme)
    page='<!doctype html><meta charset="utf-8"><title>Plate products</title><h1>Plate products</h1><ul>'+''.join(links)+'</ul><a href="../../packages/">Exported ZIP snapshots</a><p><img style="max-width:100%" src="preview.png"></p>'
    (review/'index.html').write_text(page)
    navigation(ROOT/'output')
    print('Updated plate review:',review)


if __name__=='__main__':run()

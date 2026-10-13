import numpy as np
from graphite.render.canvas import GraphiteCanvas


def s(p, g='HB'):
    xs=np.linspace(0,1,64,dtype=np.float32)
    x=20+xs*80
    y=32+np.zeros_like(x,dtype=np.float32)
    pts=np.column_stack([x,y]).astype(np.float32)
    stroke={'pts':pts,'width':np.full(64,14.0,dtype=np.float32),'pressure':np.full(64,p,dtype=np.float32),'grade':g,'layer':'L1_tone','kind':'hatch','footprint':'side','seed':13}
    c=GraphiteCanvas((96,96),seed=7,paper_preset='cold_press')
    c.add_stroke(stroke)
    return float(np.mean(c.snapshot('L1_tone')))

for g in ['2H','H','HB','B','2B','4B','6B','8B']:
    vals=[]
    for p in [0.1,0.2,0.3,0.4,0.5,0.7,0.9,1.0]:
        vals.append((p,s(p,g)))
    print(g, vals)

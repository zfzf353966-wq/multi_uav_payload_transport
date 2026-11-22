"""Offline geometry check from actual Gazebo poses; never a policy input."""
import json
import csv
from pathlib import Path
import numpy as np
import yaml


def audit_geometry(output,config_path):
    output=Path(output)
    config=yaml.safe_load(Path(config_path).read_text())
    specs=json.loads((output/'obstacles.json').read_text())['obstacles']
    boxes=[]
    for o in config['static_obstacles']:
        c,h=np.array(o['center']),np.array(o['size'])/2
        boxes.append([c-h,c+h])
    crowns=[]
    for x,y,z in config['tree_positions']:
        c=np.array([x,y,z+1.5]);h=np.array([.28,.28,1.5])
        boxes.append([c-h,c+h]);crowns.append([x,y,z+4.])
    boxes=np.array(boxes);centres=boxes.mean(axis=1);half=(boxes[:,1]-boxes[:,0])/2
    minimum=[float('inf')]*3;samples=[0]*3
    windows={}
    trace=output/'control_trace.csv'
    if trace.exists():
        with trace.open() as stream:
            for row in csv.DictReader(stream):
                if 'monotonic_s' not in row:continue
                i=int(row['vehicle'])-1;t=float(row['monotonic_s'])
                if i not in windows:windows[i]=[t,t]
                else:windows[i][1]=t
    path=output/'gazebo_poses.jsonl'
    if path.exists():
        with path.open() as stream:
            for line in stream:
                try:frame=json.loads(line)
                except json.JSONDecodeError:continue
                poses={p['name']:p for p in frame['poses']}
                points=list(crowns);radii=[1.55]*len(crowns)
                for s in specs:
                    if s['name'] in poses:
                        points.append(poses[s['name']]['position']);radii.append(s['radius'])
                if 'moving_obstacle_1' in poses and 'sphere_link' in poses:
                    points.append((np.array(poses['moving_obstacle_1']['position'])+poses['sphere_link']['position']).tolist());radii.append(.45)
                points=np.array(points);radii=np.array(radii)
                for i in range(3):
                    if i not in windows or not windows[i][0]<=frame['monotonic_s']<=windows[i][1]:continue
                    name=f'x500_{i+1}'
                    if name not in poses:continue
                    p=np.array(poses[name]['position'])
                    q=np.abs(p-centres)-half
                    box_distance=(np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)).min()-.45
                    sphere_distance=(np.linalg.norm(points-p,axis=1)-radii-.45).min()
                    minimum[i]=min(minimum[i],float(box_distance),float(sphere_distance));samples[i]+=1
    result=dict(method='cruise timestamps matched to sampled Gazebo root poses; UAV radius .45 m; exact building boxes/crowns, conservative trunk boxes; takeoff and landing contact excluded',
                samples=samples,minimum_clearance_m=[v if np.isfinite(v) else None for v in minimum],
                passed=all(n>0 for n in samples) and all(v>=0 for v in minimum))
    (output/'geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

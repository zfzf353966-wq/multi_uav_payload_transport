"""Plot actual fleet flight, payload events and measured obstacle heights."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args();root=args.run_dir
    rows=list(csv.DictReader((root/'control_trace.csv').open()))
    summary=json.loads((root/'summary.json').read_text())
    config=yaml.safe_load((root/'city_config.yaml').read_text())
    specs=json.loads((root/'obstacles.json').read_text())['obstacles']
    events=json.loads((root/'package_events.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    clock=json.loads((root/'mission_clock.json').read_text())['monotonic_start']
    departure={s['name']:[] for s in specs if s['category']=='departure'}
    recorded={i:[] for i in (1,2,3)}
    with (root/'gazebo_poses.jsonl').open() as stream:
        for index,line in enumerate(stream):
            if index%5:continue
            try:f=json.loads(line)
            except json.JSONDecodeError:continue
            t=f['monotonic_s']-clock
            if t<0:continue
            for pose in f['poses']:
                if pose['name'] in departure:departure[pose['name']].append((t,pose['position'][2]))
                for i in (1,2,3):
                    if pose['name']==f'x500_{i}':recorded[i].append([t]+pose['position'])
    fig=plt.figure(figsize=(15,10))
    xy=fig.add_subplot(221);space=fig.add_subplot(222,projection='3d')
    height=fig.add_subplot(223);cargo=fig.add_subplot(224)
    for o in config['static_obstacles']:
        x,y,z=o['center'];sx,sy,sz=o['size']
        xy.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,color='gray',alpha=.25))
        space.bar3d(x-sx/2,y-sy/2,0,sx,sy,sz,color='gray',alpha=.1,shade=False)
    for i,color in enumerate(['#1565c0','#e65100','#2e7d32'],1):
        samples=[r for r in rows if int(r['vehicle'])==i]
        if not samples:continue
        points=np.array(recorded[i]) if recorded[i] else np.array([[float(r[k]) for k in ('elapsed_s','x','y','z')] for r in samples])
        xy.plot(points[:,1],points[:,2],color=color,label=f'UAV {i}')
        space.plot(points[:,1],points[:,2],points[:,3],color=color,label=f'UAV {i}')
        height.plot(points[:,0],points[:,3],color=color,label=f'UAV {i}')
    for s in specs:
        color='#d32f2f' if s['category']=='departure' else '#8e24aa'
        xy.scatter(s['center'][0],s['center'][1],c=color,s=20,marker='x')
        space.scatter(*s['center'],c=color,s=15,marker='x')
    for pickup in config['pickup_points']:
        x,y,z=pickup['package_position']
        xy.scatter(x,y,c='#ffca28',edgecolors='black',s=28,marker='s',zorder=8)
    for destination in config['delivery_points']:
        name=destination['id'];x,y,z=destination['marker_position']
        tasks={t['id'] for t in plan['tasks'] if t['delivery']==name}
        done=len(tasks.intersection(summary['delivered']))
        owners=sorted({e['vehicle'] for e in events if e['kind']=='delivery' and e['task'] in tasks})
        label=f"{name}: {done}/{len(tasks)}\nUAV {','.join(map(str,owners)) or '-'}"
        xy.scatter(x,y,c='#29b6f6',edgecolors='black',s=48,marker='D',zorder=9)
        xy.annotate(label,(x,y),xytext=(4,5),textcoords='offset points',fontsize=7,zorder=10)
        space.scatter(x,y,z,c='#29b6f6',edgecolors='black',s=35,marker='D')
    for name,points in departure.items():
        if points:
            a=np.array(points);height.plot(a[:,0],a[:,1],'--',alpha=.6,label=name)
    for kind,color in [('pickup','#757575'),('delivery','#1565c0')]:
        t=[0.]+[e['elapsed_s'] for e in events if e['kind']==kind]
        cargo.step(t,range(len(t)),where='post',label=kind,color=color)
    cargo.axhline(summary['expected_tasks'],color='black',ls=':',label='All tasks')
    xy.set(xlabel='East (m)',ylabel='North (m)',title='Actual flight / red: departure obstacles / purple: city obstacles',xlim=(-40,40),ylim=(-30,30))
    xy.set_aspect('equal');xy.legend()
    space.set(xlabel='East (m)',ylabel='North (m)',zlabel='Height (m)',title='Gazebo actual flight, including takeoff and landing')
    height.set(xlabel='Elapsed (s)',ylabel='Actual height (m)',title='Aircraft and measured 3D obstacle motion');height.legend(fontsize=8)
    cargo.set(xlabel='Elapsed (s)',ylabel='Task count',title='Committed pickup and delivery events',ylim=(0,summary['expected_tasks']+.5));cargo.legend()
    for ax in (xy,height,cargo):ax.grid(alpha=.2)
    fig.suptitle(f"A* + direct PPO delivery / {'PASS' if summary['success'] else 'FAIL'} / delivered {len(summary['delivered'])}/{summary['expected_tasks']} / no APF")
    fig.tight_layout();output=root/'delivery_overview.png';fig.savefig(output,dpi=160);print(output)


if __name__=='__main__':main()

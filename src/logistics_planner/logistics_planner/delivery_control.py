"""Static-map novelty and conservative speed limiting, without APF guidance."""
import math
import itertools
import copy
import numpy as np
from .direct_ppo_contract import sector_directional_clearance
from .direct_ppo_contract import project_directional_velocity


def plan_around_observed_points(grid,start,goal,points,clearance=1.3,peer_points=()):
    """A* snapshot of sensed surface hits; never read obstacle truth poses."""
    from .grid_planner import AStar3D
    temporary=copy.copy(grid)
    temporary.occupied=grid.occupied.copy()
    # Surface hit + body radius + reaction/motion reserve. Quantize to avoid
    # rebuilding the same occupied box for every lidar pixel.
    anchors=np.unique(np.round(np.asarray(points).reshape(-1,3)/.6)*.6,axis=0)
    for p in anchors:
        temporary._occupy_box(tuple(p-clearance),tuple(p+clearance))
    for point in peer_points:
        p=np.asarray(point)
        lo=temporary.point_to_index(tuple(p-4.),clamp=True)
        hi=temporary.point_to_index(tuple(p+4.),clamp=True)
        for cell in itertools.product(*(range(a,b+1) for a,b in zip(lo,hi))):
            if np.linalg.norm(np.asarray(temporary.index_to_point(cell))-p)<3.6:
                temporary.occupied.add(cell)
    # An already-close sensed surface can inflate over the start voxel.
    # Preserve the measured safe origin and a short escape connection only.
    origin=temporary.point_to_index(start)
    if origin not in grid.occupied:temporary.occupied.discard(origin)
    return AStar3D(temporary).plan(tuple(start),tuple(goal))


def fleet_radial_bounds(relative_positions,peer_velocities):
    """Both aircraft obey separation, including the right-of-way winner.

    FLU vectors are relative to the controlled aircraft. Never assume an
    aircraft moving away will keep moving; approaching peers count in full.
    """
    relative=np.asarray(relative_positions,dtype=float).reshape(-1,3)
    peers=np.asarray(peer_velocities,dtype=float).reshape(-1,3)
    distances=np.linalg.norm(relative,axis=1)
    valid=(distances>.01)&(distances<8.)
    normals=relative[valid]/distances[valid,None]
    approaching=np.minimum(0.,np.sum(normals*peers[valid],axis=1))
    return normals,(distances[valid]-3.2)/2.+approaching


def known_static_cloud_mask(world,boxes,crowns):
    """Match known physical geometry and its thin roof/pad visual surfaces."""
    world=np.asarray(world).reshape(-1,3)
    q=np.abs(world[:,None,:]-boxes.mean(axis=1))-(boxes[:,1]-boxes[:,0])/2
    distance=np.linalg.norm(np.maximum(q,0),axis=2)+np.minimum(q.max(axis=2),0)
    # Caps extend .16 m, service pads .24 m above the collision boxes.
    # Add .16 m for sensing/pose error. Moving orbits have >=.5 m clearance.
    static=(distance.min(axis=1)<.4)|(world[:,2]<.25)
    crowns=np.asarray(crowns)
    if len(crowns):static|=(np.linalg.norm(world[:,None,:]-crowns[None,:,:],axis=2)-1.55).min(axis=1)<.25
    return static


def cloud_radial_bounds(points,static_mask=None):
    """Safety-only FLU surface hits, translated from lidar to body origin.

    The wider cloud is separate from PPO's trained 36-value observation.
    Known static surfaces need no moving-object approach reserve.
    """
    points=np.asarray(points,dtype=float).reshape(-1,3)
    relative=points+np.array([0.,0.,.6])
    distance=np.linalg.norm(relative,axis=1)
    valid=np.isfinite(relative).all(axis=1)&(distance>.75)&(distance<6.)
    relative,distance=relative[valid],distance[valid]
    static=np.zeros(len(points),dtype=bool) if static_mask is None else np.asarray(static_mask,dtype=bool)
    static=static[valid]
    reserve=np.where(static,.85,1.5)
    approach=np.where(static,0.,.35)
    bounds=np.maximum(0.,(distance-reserve)/1.8-approach)
    return relative/distance[:,None],bounds


def limit_cloud_velocity(command,radial_bounds):
    """Reduce requested components continuously before blind vertical approach."""
    requested=np.asarray(command,dtype=np.float32)
    normals,bounds=radial_bounds
    if not len(bounds):return requested,False
    choices=[]
    for mask in itertools.product((0.,1.),repeat=3):
        candidate=requested*np.asarray(mask)
        closing=normals@candidate
        toward=closing>1e-6
        scale=min(1.,float(np.min(bounds[toward]/closing[toward]))) if toward.any() else 1.
        candidate=candidate*max(0.,scale)
        if np.any(normals@candidate>bounds+1e-5):continue
        choices.append((float(np.sum((candidate-requested)**2)),candidate))
    selected=min(choices,key=lambda x:x[0])[1].astype(np.float32) if choices else np.zeros(3,dtype=np.float32)
    return selected,bool(np.linalg.norm(selected-requested)>1e-5)


def physical_segment_free(first,last,boxes,crowns,minimum,maximum,clearance=.35):
    first,last=np.asarray(first),np.asarray(last)
    points=np.linspace(first,last,max(2,int(np.ceil(np.linalg.norm(last-first)/.15))+1))
    if np.any(points<np.asarray(minimum)) or np.any(points>np.asarray(maximum)):return False
    centre=boxes.mean(axis=1);half=(boxes[:,1]-boxes[:,0])/2
    q=np.abs(points[:,None,:]-centre[None,:,:])-half[None,:,:]
    signed=np.linalg.norm(np.maximum(q,0),axis=2)+np.minimum(q.max(axis=2),0)
    if signed.min()-.45<clearance:return False
    if len(crowns) and (np.linalg.norm(points[:,None,:]-np.array(crowns)[None,:,:],axis=2)-1.55-.45).min()<clearance:return False
    return True


def novel_obstacle_minimum(measured, expected, tolerance=.6):
    actual=np.asarray(measured,dtype=np.float32)
    static=np.asarray(expected,dtype=np.float32)
    if actual.shape!=(36,) or static.shape!=(36,):
        raise ValueError('Expected 36-sector scans')
    mask=np.isfinite(actual) & np.isfinite(static) & (actual<static-tolerance)
    return float(actual[mask].min()) if mask.any() else 20.


def relevant_dynamic_minimum(measured,expected,goal,nominal,previous=None,dt=.1,body_velocity=None):
    actual=np.asarray(measured,dtype=float);static=np.asarray(expected,dtype=float)
    novel=np.where(actual<static-.6,actual,20.)
    directional=sector_directional_clearance(novel,goal)
    normals,bounds=dynamic_radial_bounds(actual,static,previous,dt,body_velocity)
    active=(actual<static-.6)&(actual<4.5)
    if len(bounds):
        risk=normals@np.asarray(nominal)>bounds
        if risk.any():directional=min(directional,float(actual[active][risk].min()))
    return directional


def limit_tracking_speed(command, sectors):
    velocity=np.asarray(command,dtype=np.float32)
    speed=float(np.linalg.norm(velocity))
    if speed<.05:return velocity
    clearance=sector_directional_clearance(sectors,velocity)
    # Keep the unchanged braking inequality satisfied with 0.1 m reserve.
    maximum=max(0.,-1.+math.sqrt(max(0.,1.+4.*(clearance-.9))))
    return velocity*min(1.,maximum/speed)


def dynamic_radial_bounds(measured,expected,previous=None,dt=.1,body_velocity=None):
    scan=np.asarray(measured,dtype=float)
    static=np.asarray(expected,dtype=float)
    azimuth=np.tile(np.arange(12)*np.pi/6,3)
    elevation=np.repeat(np.radians([-12.,0.,12.]),12)
    normals=np.column_stack([np.cos(elevation)*np.cos(azimuth),np.cos(elevation)*np.sin(azimuth),np.sin(elevation)])
    active=(scan<static-.6)&(scan<4.5)
    closing=np.full(36,.25)
    if previous is not None and body_velocity is not None and .03<dt<.4:
        before=np.asarray(previous,dtype=float)
        valid=(before<8)&(scan<8)&(np.abs(scan-before)<.6)
        obstacle_radial=(scan-before)/dt+normals@np.asarray(body_velocity)
        closing[valid]=np.clip(-obstacle_radial[valid],0.,1.2)
    return normals[active],((scan-1.4)/1.2-closing)[active]


def project_requested_components(command, sectors, map_free=None, radial_bounds=None):
    """Keep the closest safe subset of requested components; never add motion.

    This safety projection avoids freezing a requested climb merely because
    its horizontal component is blocked. Zero remains the final fallback.
    """
    requested=np.array(command,dtype=np.float32)
    if requested.shape!=(3,) or not np.isfinite(requested).all():
        raise ValueError('Expected finite three-vector velocity')
    norm=float(np.linalg.norm(requested))
    if norm>2:requested*=2/norm
    requested[2]=np.clip(requested[2],-1.,1.)
    choices=[]
    for mask in itertools.product((0.,1.),repeat=3):
        candidate=requested*np.array(mask,dtype=np.float32)
        _,blocked=project_directional_velocity(candidate,sectors)
        valid=True
        if np.linalg.norm(candidate)>.01 and map_free is not None:
            try:valid=map_free(candidate)
            except ValueError:valid=False
        if blocked or not valid:
            continue
        if radial_bounds is not None:
            normals,bounds=radial_bounds
            if len(bounds) and np.any(normals@candidate>bounds+1e-5):continue
        choices.append((float(np.sum((candidate-requested)**2)),candidate))
    selected=min(choices,key=lambda item:item[0])[1] if choices else np.zeros(3,dtype=np.float32)
    return selected, bool(np.linalg.norm(selected-requested)>1e-5)

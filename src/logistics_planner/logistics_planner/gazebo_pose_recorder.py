"""Record actual simulator pose messages for offline sensor diagnostics."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import argparse
import json
import signal
import threading
import time
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node, SubscribeOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    node = Node()
    with open(args.output, 'w', buffering=1) as stream:
        def callback(message):
            poses = []
            for pose in message.pose:
                if pose.name in ('x500_1', 'x500_2', 'x500_3', 'uav_1_lidar', 'uav_2_lidar', 'uav_3_lidar', 'mupt_complex_moving_obstacles',
                                  'orange_sphere', 'red_sphere', 'purple_sphere','moving_obstacle_1','sphere_link') or pose.name.startswith(('city_dynamic_', 'package_')):
                    p, q = pose.position, pose.orientation
                    poses.append(dict(name=pose.name, id=pose.id,
                                      position=[p.x, p.y, p.z],
                                      orientation_xyzw=[q.x, q.y, q.z, q.w]))
            if poses:
                stream.write(json.dumps(dict(monotonic_s=time.monotonic(),
                    simulation_s=message.header.stamp.sec + message.header.stamp.nsec * 1e-9,
                    poses=poses)) + '\n')
        options = SubscribeOptions()
        options.msgs_per_sec = 30
        topic = f'/world/{args.world}/pose/info'
        if not node.subscribe(Pose_V, topic, callback, options):
            raise SystemExit(f'Could not subscribe to {topic}')
        print(f'Recording {topic}', flush=True)
        while not stopped.wait(.2):
            pass
        node.unsubscribe(topic)


if __name__ == '__main__':
    main()

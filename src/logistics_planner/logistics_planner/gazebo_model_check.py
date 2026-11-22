"""Confirm an EntityFactory request resulted in a real Gazebo model."""
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import argparse
import json
import time
from pathlib import Path
from gz.transport13 import Node
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.scene_pb2 import Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    node = Node()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        success, scene = node.request(f"/world/{args.world}/scene/info", Empty(), Empty, Scene, 1000)
        if success:
            for model in scene.model:
                if model.name == args.model:
                    pose = model.pose.position
                    result = dict(world=args.world, model=model.name,
                                  position=[pose.x, pose.y, pose.z], links=[link.name for link in model.link])
                    if args.output:
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps(result))
                    return
        time.sleep(.1)
    raise SystemExit(f"Gazebo model {args.model!r} was not created; inspect the SDF path and simulator log")


if __name__ == "__main__":
    main()

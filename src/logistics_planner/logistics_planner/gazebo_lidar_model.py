"""Ray geometry of the leveled virtual Gazebo lidar, in body FLU coordinates."""
import numpy as np

AZIMUTHS = np.linspace(-np.pi, np.pi, 120, dtype=np.float32)
ELEVATIONS = np.radians(np.array([-12., 0., 12.], dtype=np.float32))
RAYS = np.array([(np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e))
                 for e in ELEVATIONS for a in AZIMUTHS], dtype=np.float32)
SECTORS = (np.floor((AZIMUTHS + np.pi / 12) / (np.pi / 6)).astype(int) % 12)
DOWN_ANGLES = np.radians(np.linspace(-15, 15, 5))
DOWN_RAYS = np.array([(np.sin(e), np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a))
                      for e in DOWN_ANGLES for a in DOWN_ANGLES], dtype=np.float32)


def rotate_rays(rays, heading):
    c, s = np.cos(heading), np.sin(heading)
    result = rays.copy()
    result[:, 0] = c * rays[:, 0] - s * rays[:, 1]
    result[:, 1] = s * rays[:, 0] + c * rays[:, 1]
    return result


def cast_ranges(origin, rays, obstacles, maximum=20., minimum=.6):
    distances = np.full(len(rays), maximum, dtype=np.float32)
    for obstacle in obstacles:
        relative = obstacle.position - origin
        projection = rays @ relative
        discriminant = projection ** 2 - (float(relative @ relative) - obstacle.radius ** 2)
        hit = projection - np.sqrt(np.maximum(discriminant, 0))
        valid = (discriminant >= 0) & (hit >= minimum) & (hit <= maximum)
        distances = np.minimum(distances, np.where(valid, hit, maximum))
    below = rays[:, 2] < -1e-6
    ground = np.full(len(rays), maximum, dtype=np.float32)
    ground[below] = -origin[2] / rays[below, 2]
    ground = np.where((ground >= minimum) & (ground <= maximum), ground, maximum)
    return np.minimum(distances, ground).astype(np.float32)


def sector_scan(position, heading, obstacles):
    origin = position + np.array([0, 0, .6], dtype=np.float32)
    distances = cast_ranges(origin, rotate_rays(RAYS, heading), obstacles).reshape(3, 120)
    sectors = np.full((3, 12), 20., dtype=np.float32)
    for layer in range(3):
        np.minimum.at(sectors[layer], SECTORS, distances[layer])
    return sectors.reshape(-1)


def downward_clearance(position, heading, obstacles):
    origin = position + np.array([0, 0, -.4], dtype=np.float32)
    return float(cast_ranges(origin, rotate_rays(DOWN_RAYS, heading), obstacles, 15., .3).min())

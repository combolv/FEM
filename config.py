import numpy as np
TETRA_PARSE = (
    ((0,0,0),(0,0,1),(0,1,0),(1,0,0)),
    ((0,1,1),(0,0,1),(1,1,1),(0,1,0)),
    ((1,1,0),(1,0,0),(1,1,1),(0,1,0)),
    ((1,0,1),(0,0,1),(1,1,1),(1,0,0)),
    ((0,0,1),(1,1,1),(1,0,0),(0,1,0))
)

def generate_trangle_mesh(shape_config):
    x_start, y_start, x_len, y_len = shape_config["x0"], shape_config["y0"], shape_config["width"], shape_config["height"]
    force_config = shape_config["force"]
    f_len = force_config["duration"]
    x = np.linspace(x_start, x_start + x_len, x_len + 1)
    y = np.linspace(y_start, y_start + y_len, y_len + 1)
    xy = np.array(np.meshgrid(x, y)).transpose(2, 1, 0).reshape((-1, 2))
    idx = []
    force = np.zeros_like(xy)
    if force_config["type"] == "up-down":
        for i in range(x_len + 1):
            force[i * (y_len + 1)] = [0, - force_config["magnitude"]]
            force[i * (y_len + 1) + y_len] = [0, force_config["magnitude"]]
    else:
        raise NotImplementedError
    for i in range(x_len):
        for j in range(y_len):
            pt1 = i * (y_len + 1) + j
            pt2 = (i + 1) * (y_len + 1) + j
            pt3 = pt1 + 1
            pt4 = pt2 + 1
            idx.append([pt1, pt2, pt3])
            idx.append([pt2, pt3, pt4])
    batch_force = np.array([force] * f_len)
    return xy, idx, batch_force


def generate_tetra_cuboid_and_force(shape_config, force_config):
    size_scalar = shape_config["cell side length"]
    x_start, y_start, z_start = shape_config["x0"], shape_config["y0"], shape_config["z0"]
    x_len, y_len, z_len = shape_config["length"], shape_config["width"], shape_config["height"]
    f_len = force_config["duration"]
    x = np.linspace(x_start, x_start + x_len, x_len + 1)
    y = np.linspace(y_start, y_start + y_len, y_len + 1)
    z = np.linspace(z_start, z_start + z_len, z_len + 1)
    xyz = np.array(np.meshgrid(x, y, z)).transpose(2, 1, 3, 0).reshape((-1, 3))
    # [0, 0, 0~z] -> [0, 1, 0~z] -> ... -> [0, y, 0~z], -> [1, 0, 0~z] -> ...
    to_idx = lambda i, j, k : i * (y_len+1) * (z_len+1) + j * (z_len+1) + k
    boundary_ids = []
    # faces with both size visible
    for i in range(x_len):
        for j in range(y_len):
            for k in [0, z_len]:
                boundary_ids.append([to_idx(i, j, k), to_idx(i + 1, j, k), to_idx(i, j + 1, k)])
                boundary_ids.append([to_idx(i+1, j+1, k), to_idx(i, j + 1, k), to_idx(i + 1, j, k)])
                boundary_ids.append([to_idx(i, j, k), to_idx(i, j + 1, k), to_idx(i + 1, j, k)])
                boundary_ids.append([to_idx(i+1, j+1, k), to_idx(i + 1, j, k), to_idx(i, j + 1, k)])
    for i in range(x_len):
        for j in [0, y_len]:
            for k in range(z_len):
                boundary_ids.append([to_idx(i, j, k), to_idx(i + 1, j, k), to_idx(i, j, k + 1)])
                boundary_ids.append([to_idx(i + 1, j, k + 1), to_idx(i + 1, j, k), to_idx(i, j, k + 1)])
                boundary_ids.append([to_idx(i, j, k), to_idx(i, j, k + 1), to_idx(i + 1, j, k)])
                boundary_ids.append([to_idx(i + 1, j, k + 1), to_idx(i, j, k + 1), to_idx(i + 1, j, k)])
    for i in [0, x_len]:
        for j in range(y_len):
            for k in range(z_len):
                boundary_ids.append([to_idx(i, j, k), to_idx(i, j + 1, k), to_idx(i, j, k + 1)])
                boundary_ids.append([to_idx(i, j + 1, k + 1), to_idx(i, j + 1, k), to_idx(i, j, k + 1)])
                boundary_ids.append([to_idx(i, j, k), to_idx(i, j, k + 1), to_idx(i, j + 1, k)])
                boundary_ids.append([to_idx(i, j + 1, k + 1), to_idx(i, j, k + 1), to_idx(i, j + 1, k)])

    idx = []
    force = np.zeros_like(xyz)
    if force_config["type"] == "xy":
        for i in range(x_len + 1):
            for j in range(y_len + 1):
                force[to_idx(i, j, 0)] = [0, 0, - force_config["magnitude"]]
                force[to_idx(i, j, z_len)] = [0, 0, force_config["magnitude"]]
    elif force_config["type"] == "shear":
        for i in range(x_len + 1):
            for j in range(y_len + 1):
                force[to_idx(i, j, 0)] = [0, - force_config["magnitude"], 0]
                force[to_idx(i, j, z_len)] = [0, force_config["magnitude"], 0]
    for i in range(x_len):
        for j in range(y_len):
            for k in range(z_len):
                for tet in TETRA_PARSE:
                    idx.append([to_idx(i + pt[0], j + pt[1], k + pt[2]) for pt in tet])

    batch_force = np.array([force] * f_len)
    # print(batch_force)
    return np.array(xyz * size_scalar, dtype=np.float64), np.array(idx, dtype=int), \
           np.array(batch_force, dtype=np.float64), np.array(boundary_ids, dtype=int)

"""
config2d = {"shape": {
        "type": "rectangle grid",
        "x0": 20,
        "y0": 20,
        "width": 100, # 20
        "height": 200, # 40
        "force": {
            "type": "up-down",
            "magnitude": 1,
            "duration": 1000
        }
    },
    "material": {
        "mass": 2, # 10
        "constitutive_model": "linear",
        "mu": 1,
        "lambda": 2
    }}

config3d = {"shape": {
        "type": "cuboid",
        "x0": -5,
        "y0": -5,
        "z0": -10,
        "length": 10,
        "width": 10, # 20
        "height": 20, # 40
        "cell side length": 0.001
    },
    "force": {
            "type": "xy",
            "magnitude": 1,
            "duration": 1000
        },
    "material": {
        "mass": 2, # 10
        "constitutive_model": "linear",
        "mu": 1,
        "lambda": 2,
        "time_step": 0.5
    },
    }
"""
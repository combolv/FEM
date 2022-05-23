from scene_cpp import SceneC3d
from scene_py import Scene
from config import generate_tetra_cuboid_and_force
import numpy as np

CONFIG = {
        "shape": {
            "type": "cuboid",
            "x0": -5,
            "y0": -5,
            "z0": -10,
            "length": 20,
            "width": 20,
            "height": 40,
            "cell side length": 0.005
        },
        "force": {
            "type": "shear", # xy, shear, None
            "magnitude": 10 / 400, # 1000N / # grids
            "duration": 30
        },
        "material": {
            "mass": 1.25e-7 * 1e3,  # V * rho, rho = 1 g/cm^3
            "constitutive_model": "VK",
            "mu": 6e6 / 2.9, # k = 6MPa, nu = 0.45
            "lambda": 6e6 * 4.5 / 1.45,
            # "time_step": 0.05
        },
        "integration": {
            "method": "ImplicitEuler",
            "hyperparameters": {
                "type": "direct Newton",
                "Newton": {
                    "max iter": 20,
                    "eps": 10
                },
                "Conjugate Gradient": {
                    "max iter": 200,
                    "eps": 1,
                }
            },
            "time_step": 1e-5
        },
        "debug recording": {
            "mode": "all", #"displacement only",
            "step": 1
        }
    }

xyz, idx, test_force, boundary_ids = generate_tetra_cuboid_and_force(CONFIG["shape"], CONFIG["force"])
scene_numpy = Scene(xyz, idx, CONFIG["material"], dim=3)
scene_eigen = SceneC3d(xyz, idx, CONFIG["material"], dim=3)

# scene_numpy.solve(test_force, time_step=CONFIG["integration"]["time_step"])
# scene_eigen.solve(test_force, time_step=CONFIG["integration"]["time_step"])

scene_eigen.solve(test_force, time_step=0.01, method=CONFIG["integration"]["method"],
                  method_config=CONFIG["integration"]["hyperparameters"], debug_config=CONFIG["debug recording"])

scene_numpy.solve(test_force, time_step=0.01, method=CONFIG["integration"]["method"],
                  method_config=CONFIG["integration"]["hyperparameters"])

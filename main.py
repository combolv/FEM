import numpy as np
from scene_py import Scene
from config import generate_tetra_cuboid_and_force


if __name__ == "__main__":
    config = {
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
            "type": "xy", # xy, shear, None
            "magnitude": 2000 / 400, # 1000N / # grids
            "duration": 1000
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
                "max iter": 100,
                "eps": 1e-2
            },
            "time_step": 1e-4
        }
    }
    xyz, idx, test_force, boundary_ids = generate_tetra_cuboid_and_force(config["shape"], config["force"])
    scene = Scene(xyz, idx, config["material"], dim=3)
    
    # scene.solve(test_force, time_step=config["integration"]["time_step"], method=config["integration"]["method"],
    #             method_config=config["integration"]["hyperparameters"])

    # np.save("NRnpy.npy", np.array(scene.x_i))

    from visualization import generate_video3d
    generate_video3d(scene, 0, 1000, 1, boundary_ids, out=".\\output\\NR\\", given_xi="NRnpy.npy")

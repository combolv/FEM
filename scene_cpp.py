import numpy as np
from tqdm import tqdm
from cpp.fem_lib import Scene3d, StdDoubleVector, StdIntVector

class SceneC3d:
    def __init__(self, grids, idx, material_config, dim=2):
        self.dim = dim
        assert dim == 3
        self.XY = grids  # [# V x dim] coordinate of undeformed shape
        self.M = idx     # [# M x (dim + 1)] indices of vertices
        self.W = None    # [# M] volumn of tetrahedrals
        self.Bm = None   # [# M] inverse of deformed shape matrix
        self.material = material_config
        self.v_i = []
        self.a_i = []
        self.x_i = []
        self.core = Scene3d()


    def solve(self, external_force, time_step, method="ExplicitEuler", method_config={},
              debug_config={"mode": "displacement only", "step": 5}):
        frame_len, nodal_num, dim = external_force.shape
        print("external force shape:", external_force.shape)
        assert dim == self.dim
        assert nodal_num == len(self.XY)

        self.core.InitializeGridsAndVelocity(StdDoubleVector([float(j) for j in self.XY.ravel()]),
                                             StdIntVector([int(j) for j in self.M.ravel()]))
        self.x_i.append(np.asarray(self.core.displacement).reshape((-1, 3)))
        self.v_i.append(np.asarray(self.core.velocity).reshape((-1, 3)))
        support_model_list = ["Linear", "VK"]
        assert self.material["constitutive_model"] in support_model_list
        self.core.InitializeConstitutiveModel(float(self.material["mu"]),float(self.material["lambda"]),
                                              support_model_list.index(self.material["constitutive_model"]))
        self.core.PreComputation()
        if method == "ExplicitEuler":
            for t in range(frame_len):
                self.core.ComputeNodalForce(StdDoubleVector([float(j) for j in external_force[t].ravel()]))
                self.core.ForwardEuler(float(self.material["mass"]), float(time_step))
                if t % debug_config["step"] == 0:
                    if not debug_config["mode"] == "displacement only":
                        self.a_i.append(np.asarray(self.core.force).reshape((-1, 3)))
                        self.v_i.append(np.asarray(self.core.velocity).reshape((-1, 3)))
                    self.x_i.append(np.asarray(self.core.displacement).reshape((-1, 3)))

        elif method == "ImplicitEuler":
            assert "Newton" in method_config
            assert "Conjugate Gradient" in method_config
            assert debug_config["step"] == 1

            for t in tqdm(range(frame_len)):
                self.core.ImplicitIntegration(StdDoubleVector([float(j) for j in external_force[t].ravel()]),
                                                float(time_step),
                                                float(self.material["mass"]),
                                                float(method_config["Newton"]["eps"]),
                                                float(method_config["Conjugate Gradient"]["eps"]),
                                                int(method_config["Newton"]["max iter"]),
                                                int(method_config["Conjugate Gradient"]["max iter"])
                                                )
                if debug_config["mode"] != "displacement only":
                    self.a_i.append(np.asarray(self.core.force).reshape((-1, 3)))
                    self.v_i.append(np.asarray(self.core.velocity).reshape((-1, 3)))
                self.x_i.append(np.asarray(self.core.displacement).reshape((-1, 3)))
                print(len(self.x_i), self.x_i[-1].shape)
        else:
            raise NotImplementedError("Not supported yet: " + method)


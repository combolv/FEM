import numpy as np
from tqdm import tqdm


class npo:
    """
    Batch Operators in Numpy :
    X : [M x d x d]
    trace : X.transpose(1, 2, 0).trace() -> [M]
    transpose : X.transpose(0, 2, 1) -> [M x d x d]
    scalar mult (* [M]) : np.expand_dims(M, axis=(1,2)) * X -> [M x d x d]
    scalar I ([M] * I): ... -> [M x d x d]
    dot : [N x 3], [N x 3] -> scalar
    """
    trace = lambda x: x.transpose(1, 2, 0).trace()
    T = lambda x: x.transpose(0, 2, 1)
    scalarMult = lambda s, x: np.expand_dims(s, axis=(1,2)) * x
    scalarI = lambda s, d: np.expand_dims(s, axis=(1,2)) * np.expand_dims(np.eye(d), axis=0)
    dot = lambda a, b: np.dot(a.ravel(), b.ravel())

class Scene:
    def __init__(self, grids, idx, material_config, dim=2):
        self.dim = dim
        self.XY = grids  # [# V x dim] coordinate of undeformed shape
        self.M = idx     # [# M x (dim + 1)] indices of vertices
        self.W = None    # [# M] volumn of tetrahedrals
        self.Bm = None   # [# M] inverse of deformed shape matrix
        self.material = material_config
        self.v_i = [np.zeros_like(grids)]
        self.a_i = []
        self.x_i = [grids]

    def _constitutive_model(self, F_mat):
        # F -> P(F)
        mu, lamb = self.material["mu"], self.material["lambda"]
        d = self.dim
        I = np.eye(d)

        def _VK(F):
            E = 0.5 * (npo.T(F) @ F - I)
            return F @ (2 * mu * E + lamb * npo.scalarI(npo.trace(E), d))

        def _Linear(F):
            return mu * (F + npo.T(F) - 2 * I) + lamb * npo.scalarI(npo.trace(F - I), d)

        if "constitutive_model" in self.material:
            if self.material["constitutive_model"] == "VK":
                return _VK(F_mat)
            elif self.material["constitutive_model"] == "Linear":
                return _Linear(F_mat)
            else:
                raise NotImplementedError("No method named:" + self.material["constitutive_model"])
        return _Linear(F_mat)

    def _constitutive_model_differentials(self, F_mat, delta_F_mat):
        # F, delta_F -> delta_P(F; delta_F)
        mu, lamb = self.material["mu"], self.material["lambda"]
        d = self.dim
        I = np.eye(d)

        def _VK(F, dF):
            E = 0.5 * (npo.T(F) @ F - I)
            dE = 0.5 * (npo.T(dF) @ F + npo.T(F) @ dF)
            return dF @ (2 * mu * E + lamb * npo.scalarI(npo.trace(E), d)) + \
                    F @ (2 * mu *dE + lamb * npo.scalarI(npo.trace(dE),d))

        def _Linear(dF):
            return mu * (dF + npo.T(dF)) + lamb * npo.scalarI(npo.trace(dF), d)

        if "constitutive_model" in self.material:
            if self.material["constitutive_model"] == "VK":
                return _VK(F_mat, delta_F_mat)
            else:
                raise NotImplementedError("No method named:" + self.material["constitutive_model"])
        return _Linear(delta_F_mat)

    def _precomputation(self):
        """
        Python Loop (slow)
        self.W = []
        self.Bm = []
        for i, j, k in self.M:
            Xi, Yi = self.XY[i]
            Xj, Yj = self.XY[j]
            Xk, Yk = self.XY[k]
            Dm = np.array([
                [Xi - Xk, Xj - Xk],
                [Yi - Yk, Yj - Yk]
            ])
            self.W.append(np.abs(0.5 * np.linalg.det(Dm)))
            self.Bm.append(np.linalg.inv(Dm))
        """
        assert self.dim + 1 == len(self.M[0])
        d = self.dim
        volumn_scalar = 2 if d == 2 else 6
        triangles = self.XY[self.M]
        Dm = npo.T(triangles[:, :d, :] - triangles[:, d:, :])
        self.W = np.abs(np.linalg.det(Dm) / volumn_scalar)
        self.Bm = np.linalg.inv(Dm)

    def _compute_nodal_force_from_f_and_x(self, f, x):
        ret = f.copy()
        d = self.dim
        triangles = x[self.M]
        Ds = npo.T(triangles[:, :d, :] - triangles[:, d:, :])
        F = Ds @ self.Bm
        P = self._constitutive_model(F)
        H = - npo.scalarMult(self.W, P @ npo.T(self.Bm))
        minusHsum = - np.sum(H, axis=-1, keepdims=True)
        Hijk = np.c_[H, minusHsum]
        np.add.at(ret, self.M, npo.T(Hijk))
        return ret

    def _compute_nodal_force(self, f_ext):
        """
        Python Loop (slow)
        for i, idx in enumerate(self.M):
            a, b, c = idx
            ds = np.array([[self.x_i[-1][a][0] - self.x_i[-1][c][0], self.x_i[-1][b][0] - self.x_i[-1][c][0]],
                           [self.x_i[-1][a][1] - self.x_i[-1][c][1], self.x_i[-1][b][1] - self.x_i[-1][c][1]]])
            f = ds @ self.Bm[i]
            p = self.material["mu"] * (f + f.T - 2 * np.eye(2)) + self.material["lambda"] * (f - np.eye(2)).trace() * np.eye(2)
            h = - self.W[i] * p @ self.Bm[i].T
        """
        assert self.W is not None
        assert self.Bm is not None
        assert f_ext.shape == self.XY.shape
        f_i = self._compute_nodal_force_from_f_and_x(f_ext, self.x_i[-1])
        self.a_i.append(f_i / self.material["mass"])

    def _forward_euler(self, time_step):
        assert len(self.a_i) == len(self.x_i) == len(self.v_i)
        self.v_i.append(self.v_i[-1] + time_step * self.a_i[-1])
        self.x_i.append(self.x_i[-1] + time_step * self.v_i[-2])

    def _NRsolver(self, time_step, para_config, f_ext):
        assert len(self.a_i) == len(self.x_i) == len(self.v_i)
        newton_config = para_config["Newton"]
        cg_config = para_config["Conjugate Gradient"]

        v0 = self.v_i[-1]
        M = self.material["mass"]
        t2_m = time_step ** 2 / M
        def conjugate_gradient(r0, d, v0, max_iter, eps):
            for i in range(max_iter):
                Ad = d - self._compute_nodal_force_differentials(t2_m * d)
                r0_2 = npo.dot(r0, r0)
                if r0_2 < eps:
                    return v0, i + 1
                alpha = r0_2 / npo.dot(d, Ad)
                v0 = v0 + alpha * d
                r0 = r0 - alpha * Ad
                d = r0 + npo.dot(r0, r0) / r0_2 * d
            else:
                return v0, 0

        r0 = time_step * self.a_i[-1] + self._compute_nodal_force_differentials(t2_m * v0)
        v1, ret = conjugate_gradient(r0, r0, v0, cg_config["max iter"], cg_config["eps"])
        if not ret:
            print("Max iter reached in CG at: " + str(len(self.x_i) - 1))
        self.v_i.append(v1)
        self.x_i.append(self.x_i[-1] + time_step * v1)
        if newton_config["max iter"] > 1:
            # Here x, v already computed and stored as self.x_i[-1], v_i[-1]
            # only to refine them
            # i.e. self.x_i[-1] = x_{t+1}
            for i in range(newton_config["max iter"] - 1):
                # Note: you cannot reuse f_now since generally
                # f^{ext}_{t+1} != f^{ext}_{t}
                print("Init res:", time_step * np.linalg.norm(self.a_i) ** 2)
                # print(f_ext[:5, :])
                f_now = self._compute_nodal_force_from_f_and_x(f_ext, self.x_i[-1])
                true_residual = self.v_i[-1] - self.v_i[-2] - (time_step / M) * f_now
                true_norm = np.linalg.norm(true_residual) ** 2
                print("start N iter with res:", true_norm)
                if true_norm < newton_config["eps"]:
                    return
                v0 = self.v_i[-1]
                r0 = - true_residual
                v1, ret = conjugate_gradient(r0, r0, v0, cg_config["max iter"], cg_config["eps"])

                if not ret:
                    print("Max iter reached in CG at: " + str(len(self.x_i) - 1))
                if ret == 1 or ret == 2:
                    print("No progress could be made at " + str(len(self.x_i) - 1))
                    return
                self.v_i[-1] = v1
                self.x_i[-1] = self.x_i[-2] + time_step * v1
                print("x:", self.x_i[-1][:5,:], "v:", self.v_i[-1][:5,:])
                input()
            else:
                print("Max iter reached in NR at: " + str(len(self.x_i) - 1))
                return

    def _compute_nodal_force_differentials(self, delta_x):
        """
        :param delta_x:
        :return: delta_f
        """
        assert self.W is not None
        assert self.Bm is not None
        assert delta_x.shape == self.XY.shape == self.x_i[-1].shape
        d = self.dim
        delta_f_i = np.zeros_like(delta_x)
        triangles = self.x_i[-1][self.M]
        delta_triangles = delta_x[self.M]
        Ds = npo.T(triangles[:, :d, :] - triangles[:, d:, :])
        delta_Ds = npo.T(delta_triangles[:, :d, :] - delta_triangles[:, d:, :])

        F = Ds @ self.Bm
        delta_F = delta_Ds @ self.Bm
        delta_P = self._constitutive_model_differentials(F, delta_F)
        delta_H = - npo.scalarMult(self.W, delta_P @ npo.T(self.Bm))
        delta_Hijk = np.c_[delta_H, - np.sum(delta_H, axis=-1, keepdims=True)]
        np.add.at(delta_f_i, self.M, npo.T(delta_Hijk))
        return delta_f_i

    def solve(self, external_force, time_step, method="ExplicitEuler", method_config={}):
        frame_len, nodal_num, dim = external_force.shape
        print("external force shape:", external_force.shape)
        assert dim == self.dim
        assert nodal_num == len(self.XY)
        self._precomputation()
        if method == "ExplicitEuler":
            for t in tqdm(range(frame_len)):
            # for t in range(frame_len):
                self._compute_nodal_force(external_force[t])
                self._forward_euler(time_step)
                # if t % 50 == 0:
                #     print(self.x_i[-1] - self.XY)
        elif method == "ImplicitEuler":
            for t in tqdm(range(frame_len)):
                self._compute_nodal_force(external_force[t])
                self._NRsolver(time_step, method_config, external_force[t])
        else:
            raise NotImplementedError("Not supported yet: " + method)


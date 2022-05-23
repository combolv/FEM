#include "solver.h"

#define Diverge(x) (std::isnan(x) || std::isinf(x))
#define Std2Eigen(v) Eigen::Map<const Eigen::VectorXd, Eigen::Unaligned>(v.data(), v.size())
#define Split3(x) {x(Eigen::seq(0, Eigen::last, Eigen::fix<3>), 0)(scene_->tetra_idx, 0),\
                   x(Eigen::seq(1, Eigen::last, Eigen::fix<3>), 0)(scene_->tetra_idx, 0),\
                   x(Eigen::seq(2, Eigen::last, Eigen::fix<3>), 0)(scene_->tetra_idx, 0)}
#define DsFill(name, dis) double x40##name = dis[0](e*4+3), x41##name = dis[1](e*4+3), x42##name = dis[2](e*4+3);Eigen::Matrix3d name;\
                        name << dis[0](e*4) - x40##name, dis[0](e*4+1) - x40##name, dis[0](e*4+2) - x40##name,\
                                dis[1](e*4) - x41##name, dis[1](e*4+1) - x41##name, dis[1](e*4+2) - x41##name,\
                                dis[2](e*4) - x42##name, dis[2](e*4+1) - x42##name, dis[2](e*4+2) - x42##name
#define HmatApply(f, H) for (int e = 0; e < tetra_num; e++){\
                            for(register int i=0; i<3; i++)\
                                for(register int j=0; j<3; j++)\
                                    f(scene_->tetra_idx[e*4+i] * 3 + j) += H[e](j, i);\
                            for(register int i=0; i<3; i++)\
                                f(scene_->tetra_idx[e*4+3] * 3 + i) -= H[e](i,0) + H[e](i,1) + H[e](i,2);\
                            }

#define _compute_nodal_force_diff(x) if(scene_->constitutive_method_type_ == scene_->Linear){\
                                        _compute_nodal_force_diff_linear(x);\
                                    } else {\
                                        _compute_nodal_force_diff_nonlinear(x);\
                                    }

void ImplicitSolver::Initialize(Scene3d* scene, const std::vector<double>& external_force){
    scene_ = scene;
    ext_force = Std2Eigen(external_force);
    x0 = Std2Eigen(scene->displacement);
    v0 = Std2Eigen(scene->velocity);
    x1 = x0;
    v1 = v0;
    force0 = Std2Eigen(scene->force);
    dforce1 = ext_force;
}

inline Eigen::Matrix3d ImplicitSolver::_constitutive_model(const Eigen::Matrix3d& F_mat){
    Eigen::Matrix3d I, E;
    I.setIdentity(3, 3);
    switch (scene_->constitutive_method_type_){
        case scene_->StVK:
            E = 0.5 * (F_mat.transpose() * F_mat - I);
            return F_mat * ((2. * scene_->mu_) * E + (scene_->lambda_ * E.trace()) * I);
        default: // by default, we have linear model.
            return scene_->mu_ * (F_mat + F_mat.transpose() - 2. * I) +
               (scene_->lambda_ * (F_mat.trace() - 3.)) * I;
    }
}

inline Eigen::Matrix3d ImplicitSolver::_constitutive_model_differential_Linear(const Eigen::Matrix3d& dF_mat){
    Eigen::Matrix3d I;
    I.setIdentity(3, 3);
    return scene_->mu_ * (dF_mat + dF_mat.transpose()) + 
            (scene_->lambda_ * dF_mat.trace()) * I;
}

inline Eigen::Matrix3d ImplicitSolver::_constitutive_model_differential_Nonlinear(const Eigen::Matrix3d& F_mat, const Eigen::Matrix3d& dF_mat){
    Eigen::Matrix3d I;
    I.setIdentity(3, 3);
    switch (scene_->constitutive_method_type_){
        case scene_->StVK:
        default:
            Eigen::Matrix3d E = 0.5 * (F_mat.transpose() * F_mat - I);
            Eigen::Matrix3d dE = 0.5 * (dF_mat.transpose() * F_mat + 
                                        F_mat.transpose() * dF_mat);
            return dF_mat * ((scene_->mu_ * 2.) * E + (scene_->lambda_ * E.trace()) * I) + 
                   F_mat  * ((scene_->mu_ * 2.) *dE + (scene_->lambda_ *dE.trace()) * I);
    }
}

void ImplicitSolver::_compute_nodal_force(){
    // x1 -> force1
    force1 = ext_force;
    int tetra_num = scene_->tetra_idx.size() / 4;
    std::vector<Eigen::Matrix3d> H_mat(tetra_num);
    Eigen::VectorXd xyz[3] = Split3(x1);
    #pragma omp parallel for
    for (int e = 0; e < tetra_num; e++){
        DsFill(Ds_mat, xyz);
        Eigen::Matrix3d F_mat = Ds_mat * scene_->Bm[e];
        Eigen::Matrix3d P_mat = _constitutive_model(F_mat);
        H_mat[e] = - scene_->W[e] * (P_mat * scene_->Bm[e].transpose());
    }
    HmatApply(force1, H_mat);
}

void ImplicitSolver::_compute_nodal_force_diff_linear(const Eigen::VectorXd& w){
    dforce1.setZero();
    int tetra_num = scene_->tetra_idx.size() / 4;
    std::vector<Eigen::Matrix3d> dH_mat(tetra_num);
    Eigen::VectorXd dxyz[3] = Split3(w);
    #pragma omp parallel for
    for (int e = 0; e < tetra_num; e++){
        DsFill(dDs_mat, dxyz);
        Eigen::Matrix3d dF_mat = dDs_mat * scene_->Bm[e];
        Eigen::Matrix3d dP_mat = _constitutive_model_differential_Linear(dF_mat);
        dH_mat[e] = - scene_->W[e] * (dP_mat * scene_->Bm[e].transpose());
    }
    HmatApply(dforce1, dH_mat);
}

void ImplicitSolver::_compute_nodal_force_diff_nonlinear(const Eigen::VectorXd& w){
    dforce1.setZero();
    int tetra_num = scene_->tetra_idx.size() / 4;
    std::vector<Eigen::Matrix3d> dH_mat(tetra_num);
    Eigen::VectorXd xyz[3] = Split3(x1);
    Eigen::VectorXd dxyz[3] = Split3(w);
    #pragma omp parallel for
    for (int e = 0; e < tetra_num; e++){
        DsFill(Ds_mat, xyz);
        DsFill(dDs_mat, dxyz);
        Eigen::Matrix3d F_mat = Ds_mat * scene_->Bm[e];
        Eigen::Matrix3d dF_mat = dDs_mat * scene_->Bm[e];
        Eigen::Matrix3d dP_mat = _constitutive_model_differential_Nonlinear(F_mat, dF_mat);
        dH_mat[e] = - scene_->W[e] * (dP_mat * scene_->Bm[e].transpose());
    }
    HmatApply(dforce1, dH_mat);
}

int ImplicitSolver::_conjugate_gradient(Eigen::VectorXd& v, Eigen::VectorXd& r){
    /* Conjugate Gradient Algorithm
     * return +: steps used to converge
     *        0: cannot make progress in CG
     *       -1: exhaust max steps
     *       -2: Nan or Inf encountered
     */
    Eigen::VectorXd p(r);
    double r1_2, r0_2 = r.squaredNorm();
    for(int i = 0; i < max_cg_iter_; i++){
        if (r0_2 < cg_eps_) return i;
        if (Diverge(r0_2)){
            std::cout << "Divergence in CG." << std::endl;
            return -2;
        }
        _compute_nodal_force_diff(t2_m * p);
        Eigen::VectorXd Ap = p - dforce1;
        double alpha = r0_2 / p.dot(Ap);
        v += alpha * p;
        r -= alpha * Ap;
        r1_2 = r.squaredNorm();
        p = r + (r1_2 / r0_2) * p;
        r0_2 = r1_2;
    }
    // std::cout << "Max CG step " << max_cg_iter_ << " reached with residual error: " << r0_2 << std::endl;
    return -1;
}

double ImplicitSolver::Solve(){
    /* Implicit Integration Solver by Conjugate Gradient
     * given x0, v0, f0, solve x1, v1
     * return: +: residual error
     *             when 1. residual is below required
     *                  2. no progress could be made
     *                  3. exhaust given max iterations
     *                  4. max_newton_iter <= 1 (only linearize once)
     *        -1: somehow default return value unchanged
     *        -2: Nan or Inf encountered in the N-R method
     *        -3: Nan or Inf encountered in the conjugate gradient solver
     */
    double ret = -1.;
    _compute_nodal_force_diff(t2_m * v0);
    Eigen::VectorXd r0 = t_m * force0 + dforce1;
    int num_cg_steps = _conjugate_gradient(v1, r0);
    x1 = x0 + time_step_ * v1;
    if(num_cg_steps == -2) return -3.;
    // if(num_cg_steps == -1) num_cg_steps = max_cg_iter_;
    // std::cout << "Finish first round of CG with steps " << num_cg_steps << std::endl;
    if(max_newton_iter_ > 1){
        for(int i = 1; i < max_newton_iter_; i++){
            _compute_nodal_force();
            r0 = v0 - v1 + t_m * force1;
            ret = r0.squaredNorm();
            // std::cout << "Gap " << ret << "at NR step " << i << std::endl;
            if(ret < newton_eps_) {
                // std::cout << "The error " << ret << " is below eps at NR step " << i << std::endl;
                return ret;
            }
            if(Diverge(ret)){
                std::cout << "Divergence in NR." << std::endl;
                return -2.;
            }
            num_cg_steps = _conjugate_gradient(v1, r0);
            switch(num_cg_steps){
                case 0:
                    // std::cout << "No progress could be made at NR step " << i << std::endl;
                    return ret;
                case -1:
                    // std::cout << "Max CG step reached at NR step " << i << std::endl;
                    break; // the break is to break "switch", not "for"
                case -2:
                    return -3.;
                // default:
                    // std::cout << "CG solver returns within steps " << num_cg_steps << std::endl;
            }
            x1 = x0 + time_step_ * v1;
        }
        std::cout << "Max NR step " << max_newton_iter_ << " reached with residual error: " << ret << std::endl;
        return ret;
    } else return t_m * force0.squaredNorm();
}
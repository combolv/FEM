#include "scene.h"

#define I3(a,x) {(a)[((x)*3)], (a)[((x)*3 + 1)], (a)[((x)*3 + 2)]}
#define Eigen2Std(veigen, vstd) vstd.assign(&veigen[0], veigen.data()+veigen.cols()*veigen.rows())

void Scene3d::InitializeGridsAndVelocity(const std::vector<double>& grid, 
                        const std::vector<int>& idx){
    XY_grid.assign(grid.begin(), grid.end());
    displacement.assign(grid.begin(), grid.end());
    tetra_idx.assign(idx.begin(), idx.end());
    velocity.resize(grid.size());
}

void Scene3d::InitializeConstitutiveModel(const double mu, const double lambda, 
                                          const int method_type){
        mu_ = mu;
        lambda_ = lambda;
        constitutive_method_type_ = method_type;
    }

void Scene3d::PreComputation(){
    int tetra_num = tetra_idx.size() / 4;
    for (int e = 0; e < tetra_num; e++){
        Eigen::Matrix3d Dm_mat;
        double X1[3] = I3(XY_grid, tetra_idx[e*4]);
        double X2[3] = I3(XY_grid, tetra_idx[e*4+1]);
        double X3[3] = I3(XY_grid, tetra_idx[e*4+2]);
        double X4[3] = I3(XY_grid, tetra_idx[e*4+3]);
        Dm_mat << X1[0] - X4[0], X2[0] - X4[0], X3[0] - X4[0],
                  X1[1] - X4[1], X2[1] - X4[1], X3[1] - X4[1],
                  X1[2] - X4[2], X2[2] - X4[2], X3[2] - X4[2];
        Bm.push_back(Dm_mat.inverse());
        W.push_back(std::abs(Dm_mat.determinant() / 6.));
    }
}

inline Eigen::Matrix3d Scene3d::_constitutive_model(const Eigen::Matrix3d& F_mat){
    Eigen::Matrix3d I, E;
    I.setIdentity(3, 3);
    switch (constitutive_method_type_){
        case StVK:
            E = 0.5 * (F_mat.transpose() * F_mat - I);
            return F_mat * ((2. * mu_) * E + (lambda_ * E.trace()) * I);
        default: // by default, we have linear model.
            return mu_ * (F_mat + F_mat.transpose() - 2. * I) +
               (lambda_ * (F_mat.trace() - 3.)) * I;
    }
}

void Scene3d::ImplicitIntegration(const std::vector<double>& external_force, 
    const double time_step, const double cell_mass, const double newton_eps,
    const double cg_eps, const int max_newton_iter, const int max_cg_iter){
    ComputeNodalForce(external_force); // should NOT be called by Python, to reduce communication between Python and cpp
    ImplicitSolver solver(time_step, cell_mass, newton_eps, cg_eps, max_newton_iter, max_cg_iter);
    solver.Initialize(this, external_force);
    double ret = solver.Solve();
    Eigen2Std(solver.x1, displacement);
    Eigen2Std(solver.v1, velocity);
    // std::cout << "Solved within error: " << ret << std::endl;    
}

void Scene3d::ComputeNodalForce(const std::vector<double>& external_force){
    force.assign(external_force.begin(), external_force.end());
    int tetra_num = tetra_idx.size() / 4;
    std::vector<Eigen::Matrix3d> H_mat(tetra_num);
    #pragma omp parallel for
    for (int e = 0; e < tetra_num; e++){
        Eigen::Matrix3d Ds_mat;
        double X1[3] = I3(displacement, tetra_idx[e*4]);
        double X2[3] = I3(displacement, tetra_idx[e*4+1]);
        double X3[3] = I3(displacement, tetra_idx[e*4+2]);
        double X4[3] = I3(displacement, tetra_idx[e*4+3]);
        Ds_mat << X1[0] - X4[0], X2[0] - X4[0], X3[0] - X4[0],
                  X1[1] - X4[1], X2[1] - X4[1], X3[1] - X4[1],
                  X1[2] - X4[2], X2[2] - X4[2], X3[2] - X4[2];
        Eigen::Matrix3d F_mat = Ds_mat * Bm[e];
        Eigen::Matrix3d P_mat = Scene3d::_constitutive_model(F_mat);
        H_mat[e] = - W[e] * (P_mat * Bm[e].transpose());
    }
    for (int e = 0; e < tetra_num; e++){
        for(register int i=0; i<3; i++)
            for(register int j=0; j<3; j++)
                force[tetra_idx[e*4+i] * 3 + j ] += H_mat[e](j, i);
        for(register int i=0; i<3; i++)
            force[tetra_idx[e*4+3] * 3 + i ] -= H_mat[e](i,0) + H_mat[e](i,1) + H_mat[e](i,2);
    }
}

void Scene3d::ForwardEuler(const double cell_mass, const double time_step){
    int vertices_num = XY_grid.size();
    double velocity_step = time_step / cell_mass;
    #pragma omp parallel for
    for (int v = 0; v < vertices_num; v++){
        displacement[v] += velocity[v] * time_step;
        velocity[v] += force[v] * velocity_step;
    }
}

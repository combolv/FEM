#include "scene.h"

template<>
void Scene<3>::InitializeGridsAndVelocity(const std::vector<std::array<double, 3>>& grid,
        const std::vector<std::array<int, 4>>& idx){
            XY_grid.assign(grid.begin(), grid.end());
            displacement.assign(grid.begin(), grid.end());
            tetra_idx.assign(idx.begin(), idx.end());
            velocity.resize(grid.size());
        }

template<int dim>
void Scene<dim>::InitializeConstitutiveModel(const double mu, const double lambda,
                                 const int method_type){
         mu_ = mu;
         lambda_ = lambda;
         constitutive_method_type_ = method_type;
     }

template<>
void Scene<3>::PreComputation(){
    for (auto a : tetra_idx){
        Eigen::Matrix3d Dm_mat;
        std::array<double, 3> X1 = XY_grid[a[0]];
        std::array<double, 3> X2 = XY_grid[a[1]];
        std::array<double, 3> X3 = XY_grid[a[2]];
        std::array<double, 3> X4 = XY_grid[a[3]];
        Dm_mat << X1[0] - X4[0], X2[0] - X4[0], X3[0] - X4[0],
                  X1[1] - X4[1], X2[1] - X4[1], X3[1] - X4[1],
                  X1[2] - X4[2], X2[2] - X4[2], X3[2] - X4[2];
        Bm.push_back(Dm_mat.inverse());
        W.push_back(Dm_mat.determinant() / 6.);
    }
}

template<>
Eigen::Matrix3d Scene<3>::_constitutive_model(Eigen::Matrix3d F_mat){
    if(constitutive_method_type_ == Linear){
        Eigen::Matrix3d I;
        I.setIdentity(3, 3);
        return mu_ * (F_mat + F_mat.transpose() - 2. * I) +
               (lambda_ * (F_mat(0,0)+F_mat(1,1)+F_mat(2,2)-3.)) * I;
    } else {
        // assert constitutive_method_type_ == StVK
        Eigen::Matrix3d I;
        I.setIdentity(3, 3);
        Eigen::Matrix3d E = 0.5 * (F_mat.transpose() * F_mat - I);
        return F_mat * ((2. * mu_) * E + (lambda_ * (E(0,0)+E(1,1)+E(2,2))) * I);
    }
}

template<>
void Scene<3>::ComputeNodalForce(const std::vector<std::array<double, 3>>& external_force){
    force.assign(external_force.begin(), external_force.end());
    int tetra_num = tetra_idx.size();
    for (int e = 0; e < tetra_num; e++){
        std::array<int, 4> a = tetra_idx[e];
        Eigen::Matrix3d Ds_mat;
        std::array<double, 3> X1 = displacement[a[0]];
        std::array<double, 3> X2 = displacement[a[1]];
        std::array<double, 3> X3 = displacement[a[2]];
        std::array<double, 3> X4 = displacement[a[3]];
        Ds_mat << X1[0] - X4[0], X2[0] - X4[0], X3[0] - X4[0],
                  X1[1] - X4[1], X2[1] - X4[1], X3[1] - X4[1],
                  X1[2] - X4[2], X2[2] - X4[2], X3[2] - X4[2];
        Eigen::Matrix3d F_mat = Ds_mat * Bm[e];
        Eigen::Matrix3d P_mat = Scene<3>::_constitutive_model(F_mat);
        Eigen::Matrix3d H_mat = - W[e] * (P_mat * Bm[e].transpose());
        for(register int i=0; i<3; i++)
            for(register int j=0; j<3; j++)
                force[a[i]][j] += H_mat(j, i);
        for(register int i=0; i<3; i++)
            force[a[3]][i] -= H_mat(i,0) + H_mat(i,1) + H_mat(i,2);
    }
}

template<>
void Scene<3>::ForwardEuler(const double cell_mass, const double time_step){
    int vertices_num = XY_grid.size();
    double velocity_step = time_step / cell_mass;
    for (int v = 0; v < vertices_num; v++){
        for (register int i=0; i<3; i++){
            displacement[v][i] += velocity[v][i] * time_step;
            velocity[v][i] += force[v][i] * velocity_step;
        }
    }
}

// template class Scene<2>; // 2D scene is not available yet
template class Scene<3>;
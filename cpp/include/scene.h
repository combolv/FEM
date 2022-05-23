#ifndef SCENE_H
#define SCENE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include "omp.h"
#include "solver.h"

// to be deleted
// #include <cstdlib>
// end

class Scene3d {
    friend class ImplicitSolver;

public:
    Scene3d() {};
    void InitializeGridsAndVelocity(const std::vector<double>& grid,
        const std::vector<int>& idx);

    void InitializeConstitutiveModel(const double mu, const double lambda,
                                    const int method_type);
    // void Scene3d::InitializeConstitutiveModelFromKR(const double k, const double r, 
    //                                       const int method_type){
    // }
    void PreComputation();
    void ComputeNodalForce(const std::vector<double>& external_force);
    void ForwardEuler(const double cell_mass, const double time_step);

    std::vector<double> displacement;
    std::vector<double> velocity;
    std::vector<double> force;

    void ImplicitIntegration(const std::vector<double>& external_force, 
    const double time_step, const double cell_mass, const double newton_eps,
    const double cg_eps, const int max_newton_iter, const int max_cg_iter);

private:
    double mu_;
    double lambda_;
    int constitutive_method_type_;
    enum ConstitutiveModelType { Linear, StVK };
    std::vector<double> XY_grid;
    std::vector<int> tetra_idx;
    std::vector<Eigen::Matrix3d> Bm;
    std::vector<double> W;
    
    Eigen::Matrix3d _constitutive_model(const Eigen::Matrix3d& F_mat);
};

#endif
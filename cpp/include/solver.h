#ifndef SOLVER_H
#define SOLVER_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include "omp.h"
#include "scene.h"

class Scene3d;

class ImplicitSolver {

public:
    ImplicitSolver(const double time_step, const double cell_mass, 
    const double newton_eps, const double cg_eps, 
    const int max_newton_iter, const int max_cg_iter) : scene_(nullptr),
    time_step_(time_step), cell_mass_(cell_mass), newton_eps_(newton_eps),
    cg_eps_(cg_eps), max_newton_iter_(max_newton_iter), max_cg_iter_(max_cg_iter) {
        t_m = time_step / cell_mass;
        t2_m = t_m * time_step;
    };

    void Initialize(Scene3d* scene, const std::vector<double>& external_force);

    double Solve(); // return squared norm error in double

    // Result of x_{t+1}, v_{t+1}, f(x_{t+1}) stored here and improved iteratively
    Eigen::VectorXd v1;
    Eigen::VectorXd x1;
    Eigen::VectorXd force1;

private:
    Scene3d* scene_; // DON'T delete this when calling ~ImplicitSolver()
    
    // Input of x_{t}, v_{t}, f(x_{t}) stored here and remains constant
    Eigen::VectorXd x0;
    Eigen::VectorXd v0;
    Eigen::VectorXd force0;

    Eigen::VectorXd dforce1;
    Eigen::VectorXd ext_force;

    // some constants
    const double time_step_;
    const double cell_mass_;
    const double newton_eps_;
    const double cg_eps_;
    const int max_newton_iter_;
    const int max_cg_iter_;

    double t_m;
    double t2_m;

    int _conjugate_gradient(Eigen::VectorXd& v, Eigen::VectorXd& r);

    void _compute_nodal_force();
    void _compute_nodal_force_diff_linear(const Eigen::VectorXd& w);
    void _compute_nodal_force_diff_nonlinear(const Eigen::VectorXd& w);

    // Inline Functions of P(F), delta-P(F; delta-F)
    Eigen::Matrix3d _constitutive_model(const Eigen::Matrix3d& F_mat);
    Eigen::Matrix3d _constitutive_model_differential_Linear(const Eigen::Matrix3d& dF_mat);
    Eigen::Matrix3d _constitutive_model_differential_Nonlinear(const Eigen::Matrix3d& F_mat, const Eigen::Matrix3d& dF_mat);
    
    
};

#endif
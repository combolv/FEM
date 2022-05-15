#ifndef SCENE_H
#define SCENE_H

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <array>

template<int dim>
class Scene {
public:
    Scene();
    void InitializeGridsAndVelocity(const std::vector<std::array<double, dim>>& grid,
        const std::vector<std::array<int, 4>>& idx);

    void InitializeConstitutiveModel(const double mu, const double lambda,
                                    const int method_type);

    void PreComputation();
    void ComputeNodalForce(const std::vector<std::array<double, 3>>& external_force);
    void ForwardEuler(const double cell_mass, const double time_step);

    std::vector<std::array<double, dim>> displacement;
    std::vector<std::array<double, dim>> velocity;
    std::vector<std::array<double, dim>> force;

private:
    Eigen::Matrix3d _constitutive_model(Eigen::Matrix3d F_mat);
    double mu_;
    double lambda_;
    int constitutive_method_type_;
    enum ConstitutiveModelType { Linear, StVK };
    std::vector<std::array<double, dim>> XY_grid;
    std::vector<std::array<int, 4>> tetra_idx;
    std::vector<Eigen::Matrix3d> Bm; // 2d not support yet
    std::vector<double> W;
};

#endif
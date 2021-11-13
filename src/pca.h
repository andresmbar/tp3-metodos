#pragma once
#include "types.h"

class PCA {
public:
    PCA(unsigned int n_components);
    void fit(Matrix X);
    Eigen::MatrixXd transform(Matrix X);
    Matrix componentesPrincipales();

private:
    unsigned int _n_components;
    Matrix _X;
    Matrix _componentesPrincipales;
};

#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) {
    this->_n_components = n_components;
}

void PCA::fit(Matrix X) {
    _X = X;
    vector<double> promedios;
    double suma;
    for (int i=0; i<X.cols(); i++) {
        suma = 0;
        for (int j=0; j<X.rows(); j++) {
            suma = suma + X(j,i);
        }
        promedios.push_back(suma/X.rows());
    }

    for (int i=0; i<X.cols(); i++) {
        for (int j=0; j<X.rows(); j++) {
            _X(j,i) = (_X(j,i) - promedios[i])/(sqrt(X.rows() - 1));
        }
    }

}


MatrixXd PCA::transform(Matrix X) {
  fit(X);
  X = _X;
  Matrix Xt = X.transpose();
  Matrix Cov = Xt * X;

  pair<Vector, Matrix> eigen = get_first_eigenvalues(Cov, _n_components, 5000);

  Matrix V = eigen.second;
  _componentesPrincipales = V;

  return X * V;
}

Matrix PCA::componentesPrincipales() {
    return _componentesPrincipales;
}
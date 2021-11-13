#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps) {
    Vector b = Vector::Random(X.cols());
    double eigenvalue;

    for (int i=0; i<num_iter; i++) {
        Vector b_anterior = b;
        b = X*b;
        b = b/b.norm();
        if(b_anterior == b) break;
    }

    eigenvalue = (b.transpose() * (X*b));
    eigenvalue = eigenvalue/(b.transpose() * b);

    return make_pair(eigenvalue, b);
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon) {
    Matrix A(X);
    Vector eigvalues(num);
    Matrix eigvectors(A.rows(), num);

    for (int i=0; i<num; i++) {
        pair<double, Vector> c = power_iteration(A, num_iter, epsilon);
        eigvalues(i) = c.first;
        for(unsigned j=0; j<eigvectors.rows(); j++) {
            eigvectors(j,i) = c.second(j);
        }
        A = A - c.first * c.second * c.second.transpose();
    }

    return make_pair(eigvalues, eigvectors);
}

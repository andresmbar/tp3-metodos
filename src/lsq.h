#pragma once

#include "types.h"
#include <fstream>
using namespace std;


class LeastSquareMethod {
public:
    LeastSquareMethod();

    void fit(Matrix &A, Vector &b);
    void retrieve_matrix_from_file(string file);
    Vector ajustar(Matrix &A, Vector &b);

private:

    //Vector solveUpperTriangular(Matrix &A, Vector &b);
    //pair<Matrix, Vector> GaussianElimination(Matrix &A, Vector &b);
    std::vector<tuple<Eigen::VectorXd, int>> imagenes;



};

#include <algorithm>
//#include <chrono>
#include <iostream>
#include "lsq.h"
#include <fstream>

using namespace std;

#include <iostream>
#include <fstream>

LeastSquareMethod::LeastSquareMethod()  {

}

void LeastSquareMethod::fit(Matrix &A, Vector &b) {
    std::ofstream outfile("imagenes.txt");
    Matrix A_prima = A.transpose() * A;
    Vector b_prima = A.transpose() * b;
    outfile << A_prima.rows() << ' ' << A_prima.cols() << std::endl;
    for (unsigned k = 0; k < A_prima.rows(); ++k) {
        //auto image_pixels = Eigen::VectorXd(X.cols());
        for (unsigned j = 0; j < A_prima.cols(); ++j) {
            outfile << A_prima(k, j) << ' ';
        }
        outfile << b_prima(k) << std::endl;
    }
    outfile.close();
    this->imagenes.clear();
}


// (A.transpose() * A).ldlt().solve(A.transpose() * b)
Vector LeastSquareMethod::ajustar(Matrix &A, Vector &b) {
    auto prediccion = Vector(A.rows());
    prediccion = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    return prediccion;
}

void LeastSquareMethod::retrieve_matrix_from_file(string file) {
    std::ifstream ifs(file);
    std::vector<tuple<Eigen::VectorXd, int>> imagenes;
    double number;
    string line;
    std::getline(ifs, line);
    stringstream iss(line);
    uint cols;
    uint rows;
    iss >> rows;
    iss >> cols;

    for (unsigned i = 0; i < rows; ++i) {
        std::getline(ifs, line);
        stringstream iss(line);
        auto image_pixels = Eigen::VectorXd(cols);
        for (unsigned k = 0; k < cols; ++k) {
            iss >> number;
            image_pixels[k] = number;
        }
        int category;
        iss >> category;
        this->imagenes.push_back(make_tuple(image_pixels, category));
    }
    ifs.close();
}
/*
Vector solveUpperTriangular(Matrix &A, Vector &b){
    Vector res = Vector(A.rows());
    for (int i = 0; i < A.rows(); ++i) {
        double sumatoria = 0;

        for (int j = i-1; j >=0 ; --j) {
            sumatoria +=
            
        }
        
    }
}
*/



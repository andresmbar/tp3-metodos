#pragma once

#include "types.h"
#include <fstream>
using namespace std;


class KNNClassifier {
public:
    KNNClassifier(unsigned int k_neighbors);

    void fit(Matrix &X, Matrix &y);
    std::vector<tuple<Eigen::VectorXd, int>> retrieve_matrix_from_file(string file);
    vector<tuple<double, int>> neighbours_sorted_by_distance(Matrix &X,std::vector<tuple<Eigen::VectorXd, int>> &imagenes,int indice_imagen);
    Vector predict(Matrix &X);
    int majority_category(vector<tuple<double, int>> &vecinos,uint cant_vecinos);
    vector<tuple<double, int>> neighbours_sorted_by_distance_2(Matrix &X,int indice_imagen);
private:
    unsigned int _k_neighbors;
    Matrix* x_train;
    Matrix* y_train;



};
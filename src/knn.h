#pragma once

#include "types.h"
#include <fstream>
using namespace std;


class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix &X, Matrix &y);
    void retrieve_matrix_from_file(string file);
    vector<tuple<double, int>>* neighbours_sorted_by_distance(Matrix &X,std::vector<tuple<Eigen::VectorXd, int>> &imagenes,int indice_imagen);
    Vector predict(Matrix &X);
    int majority_category(vector<tuple<double, int>> &vecinos,uint cant_vecinos);
    void change_k(unsigned int n_neighbors);
private:
    unsigned int _n_neighbors;
    std::vector<tuple<Eigen::VectorXd, int>> imagenes;



};
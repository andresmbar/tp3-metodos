#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"
#include <fstream>

using namespace std;

#include <iostream>
#include <fstream>

KNNClassifier::KNNClassifier(unsigned int k_neighbors) {
    this->_k_neighbors = k_neighbors;
}

void KNNClassifier::fit(Matrix &X, Matrix &y) {
    this->x_train=&X;
    this->y_train=&y;
}

template<typename KeyType, typename ValueType>
std::pair<KeyType, ValueType> get_max(const std::map<KeyType, ValueType> &x) {
    using pairtype = std::pair<KeyType, ValueType>;
    return *std::max_element(x.begin(), x.end(), [](const pairtype &p1, const pairtype &p2) {
        return p1.second < p2.second;
    });
}

Vector KNNClassifier::predict(Matrix &X) {
    auto prediccion_categoria = Vector(X.rows());
    for (unsigned k = 0; k < X.rows(); ++k) {
        vector<tuple<double, int>> vecinos_ordenados = this->neighbours_sorted_by_distance_2(X,k);
        uint categoria_imagen = this->majority_category(vecinos_ordenados,this->_k_neighbors);
        prediccion_categoria(k) = categoria_imagen;
    }
    return prediccion_categoria;
}

vector<tuple<double, int>>
KNNClassifier::neighbours_sorted_by_distance_2(Matrix &X,int indice_imagen) {

    auto imagen = X.row(indice_imagen);
    Matrix imagen_to_matrix = imagen.replicate(this->x_train->rows(),1);
    Matrix diferencia_entre_imagenes = *(this->x_train) - imagen_to_matrix;

    vector<tuple<double, int>> images_norm;
    for (unsigned i = 0; i < this->x_train->rows(); ++i) {
        double norma = diferencia_entre_imagenes.row(i).norm();
        int categoria =(*this->y_train)(i,0);
        images_norm.push_back(make_tuple(norma, categoria));
    }
    sort(images_norm.begin(), images_norm.end());
    return images_norm;


}

int KNNClassifier::majority_category(vector<tuple<double, int>> &vecinos, uint cant_vecinos) {
    map<int, int> occurrences;
    occurrences[0] = 0;
    occurrences[1] = 0;
    occurrences[2] = 0;
    occurrences[3] = 0;
    occurrences[4] = 0;
    occurrences[5] = 0;
    occurrences[6] = 0;
    occurrences[7] = 0;
    occurrences[8] = 0;
    occurrences[9] = 0;

    for (int i = 0; i < cant_vecinos; i++) {
        occurrences[get<1>(vecinos[i])] = occurrences[get<1>(vecinos[i])] + 1;
    }
    auto max = get_max(occurrences);
    return max.first;
}
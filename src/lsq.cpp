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




// (A.transpose() * A).ldlt().solve(A.transpose() * b)
Vector LeastSquareMethod::ajustar(Matrix &A, Vector &b) {
    auto prediccion = Vector(A.rows());
    prediccion = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    return prediccion;
}





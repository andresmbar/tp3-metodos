#pragma once

#include "types.h"
#include <fstream>
using namespace std;


class LeastSquareMethod {
public:
    LeastSquareMethod();
  
    Vector ajustar(Matrix &A, Vector &b);


};

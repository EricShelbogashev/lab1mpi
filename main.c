#include <stdio.h>
#include "malloc.h"
#include <math.h>
#include "time.h"

#define N 12000 // matrix size

double calculateNorm(const double *vector, size_t size) {
    double res = 0;
    for (int i = 0; i < size; ++i) {
        res += vector[i] * vector[i];
    }
    return sqrt(res);
}

double *newDoubleArray(size_t len) {
    return (double *) calloc(len, sizeof(double));
}

struct Matrix {
    size_t height;
    size_t width;
    double *data;
};

struct Matrix *initMatrix(size_t height, size_t width) {
    struct Matrix *matrix = (struct Matrix *) malloc(sizeof(struct Matrix));
    matrix->data = newDoubleArray(height * width);
    matrix->height = height;
    matrix->width = width;
    return matrix;
}


int precision(const double *xNext, const double *receiverArray, const double *b, double epsilon,
              double bNorm) {

    double *yFragment = newDoubleArray(N);
    for (int i = 0; i < N; ++i) {
        yFragment[i] = 0;
    }

    //  Ax^n
    for (int i = 0; i < N; ++i) {
        yFragment[i / N] += receiverArray[i] * xNext[i % N];
    }

    //  Ax^n - b
    for (int i = 0; i < N; ++i) {
        yFragment[i] -= b[i];
    }


    double firstNorm = calculateNorm(yFragment, N);
    printf("%f norm precision\n", firstNorm);
    free(yFragment);
    if (firstNorm/bNorm < epsilon){
        return 1;
    }
    else {
        return 0;
    }
}

double *iterativeAlgorithm(const double *xPrevious, const double *matrixFragment, const double *b) {
    double *yFragment = newDoubleArray(N);
    for (int i = 0; i < N; ++i) {
        yFragment[i] = 0;
    }

    //  Ax^n
    for (int i = 0; i < N; ++i) {
        yFragment[i / N] += matrixFragment[i] * xPrevious[i % N];
    }

    //  Ax^n - b
    for (int i = 0; i < N; ++i) {
        yFragment[i] -= b[i];
    }

    double *matrixYFragment = newDoubleArray(N);
    for (int i = 0; i < N; ++i) {
        matrixYFragment[i] += matrixFragment[i] * yFragment[i % N];
    }

    double matrixYScalarMultiFragment = 0;
    for (int i = 0; i < N; ++i) {
        matrixYScalarMultiFragment += matrixYFragment[i] * matrixYFragment[i];
    }

    double yMatrixYScalarMultiFragment = 0;
    for (int i = 0; i < N; ++i) {
        yMatrixYScalarMultiFragment += yFragment[i] * matrixYFragment[i];
    }

    double *xNext = newDoubleArray(N);

    {
        free(yFragment);
        free(matrixYFragment);
    }
    return xNext;
}

void diagonalFillMatrix(struct Matrix *matrix) {
    for (int i = 0; i < matrix->height * matrix->width; ++i) {
        if (i % (N + 1) == 0) {
            matrix->data[i] = 2.0;
        } else {
            matrix->data[i] = 1.0;
        }
    }
}

void nPlusOneFillMatrix(struct Matrix *matrix) {
    for (int i = 0; i < N; ++i) {
        matrix->data[i] = N  + 1;  //i+200
    }
}

int main(int argc, char **argv) {
    struct Matrix *A = initMatrix(N, N);
    struct Matrix *x = initMatrix(1, N);
    struct Matrix *b = initMatrix(1, N);
    diagonalFillMatrix(A);
    nPlusOneFillMatrix(b);
    double bNorm = calculateNorm(b->data, N);
    time_t startTime = time(NULL);

    int flag = 0;
    int count = 0;
    while (flag == 0) {
        double *xNext = iterativeAlgorithm(x->data, A->data, b->data);
        flag = precision(xNext, A->data, b->data, 0.00000000001, bNorm);
        count++;
        if (flag == 1) {
            for (int i = 0; i < N; ++i) {
                printf("%f\n", xNext[i]);
            }
        }
    }

    free(x->data);
    free(x);
    free(b->data);
    free(b);
    time_t endTime = time(0);
    printf("Time taken: %ld sec.\n", endTime - startTime);
    return 0;
}

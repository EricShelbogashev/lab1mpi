#include <stdio.h>
#include <mpi.h>
#include "malloc.h"
#include "math.h"

#define N 100 // matrix size
#define DATA_ARRAYS_NUMBER 2

double calculateNorm(const double *vector, size_t size) {
    double res = 0;
    for (int i = 0; i < size; ++i) {
        res += vector[i] * vector[i];
    }
    return sqrt(res);
}

int **newIntPtrArray(size_t len) {
    return (int **) calloc(len, sizeof(int *));
}

int *newIntArray(size_t len) {
    return (int *) calloc(len, sizeof(int));
}

double *newDoubleArray(size_t len) {
    return (double *) calloc(len, sizeof(double));
}

double **newDoublePtrArray(size_t len) {
    return (double **) calloc(len, sizeof(double *));
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

int **splitMatrices(size_t procNumber, size_t capacity, int **previousSeparationParameters) {
    int **separationParameters = newIntPtrArray(DATA_ARRAYS_NUMBER);

    if (capacity == N) {
        // Vector separation
        int *lenArray = newIntArray(procNumber);
        for (int i = 0; i < procNumber; ++i) {
            lenArray[i] = previousSeparationParameters[0][i] / N;
        }

        int *indentArray = newIntArray(procNumber);
        for (int i = 0; i < procNumber; ++i) {
            if (i == 0) {
                indentArray[i] = 0;
            } else {
                indentArray[i] = indentArray[i - 1] + lenArray[i - 1];
            }
        }
        separationParameters[0] = lenArray;
        separationParameters[1] = indentArray;
    } else {
        // Matrix separation
        int *lenArray = newIntArray(procNumber);
        {
            size_t capacityCopy = capacity;
            int i = 0;
            while (i < procNumber - 1) {
                capacityCopy -= N * (N / procNumber);
                lenArray[i] = N * (N / procNumber);
                ++i;
            }
            lenArray[i] = capacityCopy;
        }

        int *indentArray = newIntArray(procNumber);

        for (int i = 0; i < procNumber; ++i) {
            if (i == 0) {
                indentArray[i] = 0;
            } else {
                indentArray[i] = indentArray[i - 1] + lenArray[i - 1];
            }
        }
        separationParameters[0] = lenArray;
        separationParameters[1] = indentArray;
    }

    return separationParameters;
}

void separator_free(int **separator) {
    for (int i = 0; i < 2; ++i) {
        free(separator[i]);
    }
    free(separator);
}

void accuracy(const double *xNext, const double *receiverArray, const double *b, double epsilon,
             int **matrixSeparationParameters, int **ySeparationParameters, double bNorm) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);


    double *yFragment = newDoubleArray(matrixSeparationParameters[0][rank] / N);
    for (int i = 0; i < (matrixSeparationParameters[0][rank] / N); ++i) {
        yFragment[i] = 0;
    }

    //  Ax^n
    for (int i = 0; i < matrixSeparationParameters[0][rank]; ++i) {
        yFragment[i / N] += receiverArray[i] * xNext[i % N];
    }

    //  Ax^n - b
    for (int i = 0; i < (matrixSeparationParameters[0][rank] / N); ++i) {
        yFragment[i] -= b[i + rank];
    }


    double *yN = newDoubleArray(N);
    MPI_Allgatherv(yFragment, ySeparationParameters[0][rank], MPI_DOUBLE, yN, ySeparationParameters[0],
                   ySeparationParameters[1], MPI_DOUBLE, MPI_COMM_WORLD);

    double firstNorm = calculateNorm(yN, N);
    printf("%f accuracy calculateNorm\n", firstNorm);
    free(yFragment);
    free(yN);
}


double *iterativeAlgorithm(const double *xPrevious, const double *matrixFragment, const double *b,
                           int **matrixSeparationParameters, int **ySeparationParameters,
                           int **xSeparationParameters, double *xNextFragment) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    double *yFragment = newDoubleArray((matrixSeparationParameters[0][rank] / N));
    for (int i = 0; i < (matrixSeparationParameters[0][rank] / N); ++i) {
        yFragment[i] = 0;
    }

    //  Ax^n
    for (int i = 0; i < matrixSeparationParameters[0][rank]; ++i) {
        yFragment[i / N] += matrixFragment[i] * xPrevious[i % N];
    }

    //  Ax^n - b
    for (int i = 0; i < (matrixSeparationParameters[0][rank] / N); ++i) {
        yFragment[i] -= b[i + rank];
    }

    // Cuts the vector into parts depending on the number of threads
    double *yN = newDoubleArray(N);
    MPI_Allgatherv(yFragment, ySeparationParameters[0][rank], MPI_DOUBLE, yN, ySeparationParameters[0],
                   ySeparationParameters[1], MPI_DOUBLE, MPI_COMM_WORLD);

    double *matrixYFragment = newDoubleArray(matrixSeparationParameters[0][rank] / N);
    for (int i = 0; i < matrixSeparationParameters[0][rank]; ++i) {
        matrixYFragment[i / N] += matrixFragment[i] * yN[i % N];
    }

    double matrixYScalarMultiFragment = 0;
    for (int i = 0; i < matrixSeparationParameters[0][rank] / N; ++i) {
        matrixYScalarMultiFragment += matrixYFragment[i] * matrixYFragment[i];
    }

    double yMatrixYScalarMultiFragment = 0;
    for (int i = 0; i < matrixSeparationParameters[0][rank] / N; ++i) {
        yMatrixYScalarMultiFragment += yN[i + rank] * matrixYFragment[i];
    }

    double *dualMatrixYScalarMulti = newDoubleArray(commSize);
    MPI_Gather(&matrixYScalarMultiFragment, 1, MPI_DOUBLE, dualMatrixYScalarMulti, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    double *yMatrixYScalarMulti = newDoubleArray(commSize);
    MPI_Gather(&yMatrixYScalarMultiFragment, 1, MPI_DOUBLE, yMatrixYScalarMulti, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

    double t;
    if (rank == 0) {
        double yMatrixYSum = 0;
        double matrixYSum = 0;
        for (int i = 0; i < commSize; ++i) {
            yMatrixYSum += yMatrixYScalarMulti[i];
        }
        for (int i = 0; i < commSize; ++i) {
            matrixYSum += dualMatrixYScalarMulti[i];
        }
        t = (double) (yMatrixYSum / matrixYSum);
        MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    if (rank != 0) {
        MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }


    for (int i = 0; i < matrixSeparationParameters[0][rank] / N; ++i) {
        xNextFragment[i] = 0;
    }

    for (int i = 0; i < matrixSeparationParameters[0][rank] / N; ++i) {
        xNextFragment[i] += xPrevious[i + rank] - (t * yN[i + rank]);
    }

    double *xNext = newDoubleArray(N);
    MPI_Allgatherv(xNextFragment, matrixSeparationParameters[0][rank] / N, MPI_DOUBLE, xNext,
                   xSeparationParameters[0], xSeparationParameters[1], MPI_DOUBLE, MPI_COMM_WORLD);

    {
        free(yFragment);
        free(yN);
        free(matrixYFragment);
        free(dualMatrixYScalarMulti);
        free(yMatrixYScalarMulti);
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
        matrix->data[i] = i;  //i+200
    }
}

int main() {
    MPI_Init(NULL, NULL);

    double startTime = MPI_Wtime();
    struct Matrix *matrixA = initMatrix(N, N);

    int commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int **matrixSeparationParameters = splitMatrices(commSize, matrixA->height * matrixA->width, NULL);
    int **xSeparationParameters = splitMatrices(commSize, N, matrixSeparationParameters);
    int **ySeparationParameters = splitMatrices(commSize, N, matrixSeparationParameters);
    double *xNextFragment = newDoubleArray((matrixSeparationParameters[0][rank] / N));
    double *receiverArray = newDoubleArray(matrixSeparationParameters[0][rank]);

    MPI_Barrier(MPI_COMM_WORLD);

    struct Matrix *x = initMatrix(1, N);
    double *tmp = x->data;
    struct Matrix *b = initMatrix(1, N);
    if (rank == 0) {
        diagonalFillMatrix(matrixA);
        MPI_Scatterv(matrixA, matrixSeparationParameters[0], matrixSeparationParameters[1], MPI_DOUBLE,
                     receiverArray, matrixSeparationParameters[0][rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        nPlusOneFillMatrix(b);
    }

    if (rank != 0) {
        MPI_Scatterv(matrixA, matrixSeparationParameters[0], matrixSeparationParameters[1], MPI_DOUBLE,
                     receiverArray, matrixSeparationParameters[0][rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }


    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double bNorm = calculateNorm(b->data, N);

    double *xNext = iterativeAlgorithm(x->data, receiverArray, b->data, matrixSeparationParameters,
                                       ySeparationParameters, xSeparationParameters, xNextFragment);
    accuracy(xNext, receiverArray, b->data, 0.00000000001, matrixSeparationParameters, ySeparationParameters,
             bNorm);
    x->data = xNext;
    free(x);
    free(tmp);
    free(b);
    separator_free(matrixSeparationParameters);
    separator_free(xSeparationParameters);
    separator_free(ySeparationParameters);
    free(xNextFragment);
    free(receiverArray);
    double end_time = MPI_Wtime();
    printf("Time taken: %lf sec.\n", end_time - startTime);

    MPI_Finalize();
    return 0;
}
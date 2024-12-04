#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
using namespace std;

// CUDA kernel для вычисления произведения в каждом блоке
__global__ void blockProductKernel(double* matrix, double* blockProducts, int rows, int cols) {
    __shared__ double partialProduct[1024];  // Shared память для частичных произведений
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;  // Глобальный индекс потока
    int tid = threadIdx.x;  // Локальный индекс потока в блоке

    // Проверяем, что поток не выходит за границы
    if (threadId < rows* cols) {
        partialProduct[tid] = matrix[threadId];  // Загружаем элемент в shared память
    }
    else {
        partialProduct[tid] = 1.0;  // Для потоков вне границ - нейтральное значение
    }
    __syncthreads();

    // Редукция произведения внутри блока
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partialProduct[tid] *= partialProduct[tid + stride];
        }
        __syncthreads();
    }

    // Первый поток блока записывает результат редукции в массив блоков
    if (tid == 0) {
        blockProducts[blockIdx.x] = partialProduct[0];
    }
}


// Хелпер для проверки ошибок CUDA
void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    setlocale(LC_ALL, "rus");
    srand(static_cast<unsigned>(time(0)));

    const int rows = 10000;
    const int cols = rows;
    const int size = rows * cols;

    // Создание матрицы
    double* h_matrix = new double[size];

    // Заполнение матрицы случайными целыми числами от 1 до 10
    for (int i = 0; i < size; ++i) {
        h_matrix[i] = rand() % 10 + 1;  // Генерация целых чисел от 1 до 10
    }

    double* d_matrix;
    double* d_blockProducts;
    double h_result = 1.0;

    int threadsPerBlock = 256;

    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    double* h_blockProducts = new double[blocksPerGrid];

    // Выделение памяти на устройстве
    checkCuda(cudaMalloc(&d_matrix, size * sizeof(double)), "Ошибка выделения памяти для d_matrix");
    checkCuda(cudaMalloc(&d_blockProducts, blocksPerGrid * sizeof(double)), "Ошибка выделения памяти для d_blockProducts");
    checkCuda(cudaMemcpy(d_matrix, h_matrix, size * sizeof(double), cudaMemcpyHostToDevice), "Ошибка копирования матрицы на устройство");

    double min_time = 1e9;
    double max_time = 0.0;
    double total_time = 0.0;
    int checksCount = 3;

    for (int k = 0; k < checksCount; ++k) {
        double start_time = clock();

        // Запуск ядра для вычисления произведений в блоках
        blockProductKernel <<<blocksPerGrid, threadsPerBlock>>> (d_matrix, d_blockProducts, rows, cols);
        checkCuda(cudaGetLastError(), "Ошибка запуска ядра");
        checkCuda(cudaMemcpy(h_blockProducts, d_blockProducts, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost), "Ошибка копирования результатов блоков на хост");

        // Вычисление итогового произведения на хосте
        h_result = 1.0;  // Сброс результата перед новым измерением
        for (int i = 0; i < blocksPerGrid; ++i) {
            h_result *= h_blockProducts[i];
        }

        double run_time = (clock() - start_time) / CLOCKS_PER_SEC;

        if (run_time > max_time) max_time = run_time;
        if (run_time < min_time) min_time = run_time;
        total_time += run_time;
    }

    double average_time = total_time / checksCount;

    /*
    cout << "Матрица:" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << h_matrix[i * cols + j] << " ";
        }
        cout << endl;
    }*/

    // Вывод результатов
    cout << "Произведение всех элементов матрицы: " << h_result << endl;
    cout << "Минимальное время: " << min_time << " секунд" << endl;
    cout << "Максимальное время: " << max_time << " секунд" << endl;
    cout << "Среднее время: " << average_time << " секунд\n" << endl;
        


    // Освобождение памяти
    cudaFree(d_matrix);
    cudaFree(d_blockProducts);
    delete[] h_matrix;
    delete[] h_blockProducts;


    return 0;

}




//int main() {
//    setlocale(LC_ALL, "rus");
//    srand(static_cast<unsigned>(time(0)));
//
//    const int rows = 32 * 192;
//    const int cols = rows;
//    const int size = rows * cols;
//
//    // Создание матрицы
//    double* h_matrix = new double[size];
//
//    // Заполнение матрицы случайными целыми числами от 1 до 10
//    for (int i = 0; i < size; ++i) {
//        h_matrix[i] = rand() % 10 + 1;  // Генерация целых чисел от 1 до 10
//    }
//
//    double* d_matrix;
//    double* d_blockProducts;
//    double h_result = 1.0;
//
//    //int threadsPerBlock = 160;
//    for (int threadsPerBlock = 32; threadsPerBlock <= 160; threadsPerBlock += 16) {
//
//        cout << threadsPerBlock << endl;
//
//        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
//
//        double* h_blockProducts = new double[blocksPerGrid];
//
//        // Выделение памяти на устройстве
//        checkCuda(cudaMalloc(&d_matrix, size * sizeof(double)), "Ошибка выделения памяти для d_matrix");
//        checkCuda(cudaMalloc(&d_blockProducts, blocksPerGrid * sizeof(double)), "Ошибка выделения памяти для d_blockProducts");
//        checkCuda(cudaMemcpy(d_matrix, h_matrix, size * sizeof(double), cudaMemcpyHostToDevice), "Ошибка копирования матрицы на устройство");
//
//        double min_time = 1e9;
//        double max_time = 0.0;
//        double total_time = 0.0;
//        int checksCount = 3;
//
//        for (int k = 0; k < checksCount; ++k) {
//            double start_time = clock();
//
//            // Запуск ядра для вычисления произведений в блоках
//            blockProductKernel << <blocksPerGrid, threadsPerBlock >> > (d_matrix, d_blockProducts, rows, cols);
//            checkCuda(cudaGetLastError(), "Ошибка запуска ядра");
//            checkCuda(cudaMemcpy(h_blockProducts, d_blockProducts, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost), "Ошибка копирования результатов блоков на хост");
//
//            // Вычисление итогового произведения на хосте
//            h_result = 1.0;  // Сброс результата перед новым измерением
//            for (int i = 0; i < blocksPerGrid; ++i) {
//                h_result *= h_blockProducts[i];
//            }
//
//            double run_time = (clock() - start_time) / CLOCKS_PER_SEC;
//
//            if (run_time > max_time) max_time = run_time;
//            if (run_time < min_time) min_time = run_time;
//            total_time += run_time;
//        }
//
//        double average_time = total_time / checksCount;
//
//        /*
//        cout << "Матрица:" << endl;
//        for (int i = 0; i < rows; ++i) {
//            for (int j = 0; j < cols; ++j) {
//                cout << h_matrix[i * cols + j] << " ";
//            }
//            cout << endl;
//        }*/
//
//        // Вывод результатов
//        cout << "Произведение всех элементов матрицы: " << h_result << endl;
//        cout << "Минимальное время: " << min_time << " секунд" << endl;
//        cout << "Максимальное время: " << max_time << " секунд" << endl;
//        cout << "Среднее время: " << average_time << " секунд\n" << endl;
//        // Освобождение памяти
//
//        cudaFree(d_blockProducts);
//
//        delete[] h_blockProducts;
//    }
//
//    cudaFree(d_matrix);
//    delete[] h_matrix;
//
//
//    return 0;
//
//}



//
//int main() {
//    setlocale(LC_ALL, "rus");
//    srand(static_cast<unsigned>(time(0)));
//
//    const int rows = 32 * 192;
//    const int cols = rows;
//    const int size = rows * cols;
//
//    // Создание матрицы
//    double* h_matrix = new double[size];
//
//    // Заполнение матрицы случайными целыми числами от 1 до 10
//    for (int i = 0; i < size; ++i) {
//        h_matrix[i] = rand() % 10 + 1;  // Генерация целых чисел от 1 до 10
//    }
//
//    double* d_matrix;
//    double* d_blockProducts;
//    double h_result = 1.0;
//
//    // Массив для хранения времени для разных значений потоков
//    vector<double> times;
//
//    for (int threadsPerBlock = 32; threadsPerBlock <= 160; threadsPerBlock += 16) {
//
//        cout << "Тест с " << threadsPerBlock << " потоками на блок:\n";
//
//        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
//
//        double* h_blockProducts = new double[blocksPerGrid];
//
//        // Выделение памяти на устройстве
//        checkCuda(cudaMalloc(&d_matrix, size * sizeof(double)), "Ошибка выделения памяти для d_matrix");
//        checkCuda(cudaMalloc(&d_blockProducts, blocksPerGrid * sizeof(double)), "Ошибка выделения памяти для d_blockProducts");
//        checkCuda(cudaMemcpy(d_matrix, h_matrix, size * sizeof(double), cudaMemcpyHostToDevice), "Ошибка копирования матрицы на устройство");
//
//        double min_time = 1e9;
//        double max_time = 0.0;
//        double total_time = 0.0;
//        int checksCount = 3;
//
//        for (int k = 0; k < checksCount; ++k) {
//            double start_time = clock();
//
//            // Запуск ядра для вычисления произведений в блоках
//            blockProductKernel << <blocksPerGrid, threadsPerBlock >> > (d_matrix, d_blockProducts, rows, cols);
//            checkCuda(cudaGetLastError(), "Ошибка запуска ядра");
//            checkCuda(cudaMemcpy(h_blockProducts, d_blockProducts, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost), "Ошибка копирования результатов блоков на хост");
//
//            // Вычисление итогового произведения на хосте
//            h_result = 1.0;  // Сброс результата перед новым измерением
//            for (int i = 0; i < blocksPerGrid; ++i) {
//                h_result *= h_blockProducts[i];
//            }
//
//            double run_time = (clock() - start_time) / CLOCKS_PER_SEC;
//
//            if (run_time > max_time) max_time = run_time;
//            if (run_time < min_time) min_time = run_time;
//            total_time += run_time;
//        }
//
//        double average_time = total_time / checksCount;
//        times.push_back(average_time);  // Добавляем среднее время в массив
//
//        cout << "Произведение всех элементов матрицы: " << h_result << endl;
//        cout << "Минимальное время: " << min_time << " секунд" << endl;
//        cout << "Максимальное время: " << max_time << " секунд" << endl;
//        cout << "Среднее время: " << average_time << " секунд\n" << endl;
//
//        delete[] h_blockProducts;
//    }
//
//    // Освобождение памяти
//    cudaFree(d_matrix);
//    cudaFree(d_blockProducts);
//    delete[] h_matrix;
//
//    // Построение графика зависимости времени выполнения от количества потоков
//    cout << "\nГрафик зависимости времени выполнения от количества потоков:\n";
//
//    // Определяем масштаб для вывода графика
//    double max_time_for_graph = *std::max_element(times.begin(), times.end());
//    int graph_width = 50;
//
//    for (int i = 0; i < times.size(); ++i) {
//        int bar_length = static_cast<int>((times[i] / max_time_for_graph) * graph_width);
//        cout << (32 + i * 32) << " поток(ов): " << string(bar_length, '*') << " " << times[i] << " секунд\n";
//    }
//
//    return 0;
//}
//


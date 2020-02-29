#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <time.h>
#include "../common/book.h"

//Elabora un número aleatorio
__global__ void make_rand(int seed, char* m, int size) {
    float myrandf;
    int num;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state); //Se prepara la ejecución del random de CUDA
    myrandf = curand_uniform(&state);
    myrandf *= (size - 0 + 0.999999);
    num = myrandf;
    if (m[num] == 'O')
    {
        m[num] = 'X';
    }
}

__global__ void prepare_matrix(char* p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    p[idx] = 'O';
}

//Se genera una matriz de manera que los elementos bajan una fila
__global__ void matrix_operation(char* m, char* p, int width, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int counter = 0; 
    if ((idx % width != 0) && (idx - width >= 0) && (m[idx - width - 1] == 'X')) //Esquina superior izquierda
    {
        counter++;
    }
    if ((idx % width != 0) && (m[idx - 1] == 'X')) //Lateral izquierdo
    {
        counter++;
    }
    if ((idx - width >= 0) && (m[idx - width] == 'X')) //Lateral superior
    {
        counter++;
    }
    if ((idx % width != width-1) && (idx - width >= 0) && (m[idx - width + 1] == 'X')) //Esquina superior derecha
    {
        counter++;
    }
    if ((idx % width != width - 1) && (m[idx + 1] == 'X')) //Lateral izquierdo
    {
        counter++;
    }
    if ((idx % width != 0) && (idx + width < size) && (m[idx + width - 1] == 'X')) //Esquina inferior izquierda
    {
        counter++;
    }
    if ((idx + width < size) && (m[idx + width] == 'X')) //Lateral inferior
    {
        counter++;
    }
    if ((idx % width != width - 1) && (idx + width < size) && (m[idx + width + 1] == 'X')) //Esquina inferior derecha
    {
        counter++;
    }
    if ((counter == 3) && (m[idx] == 'O')) // Una célula muerte se convierte en viva si tiene 3 células vivas alrededor de ella
    {
        p[idx] = 'X';
    }
    else if (((counter < 2) || (counter > 3)) && (m[idx] == 'X')) // Una célula viva se convierte en muerte si alrededor de ella hay un número de células distinto de 2 o 3
    {
        p[idx] = 'O';
    }
    else //La célula mantiene su estado
    {
        p[idx] = m[idx];
    }
}


void generate_matrix(char* m, int size, int nBlocks, int nThreads);
int generate_random(int min, int max);
void step_life(char* m, char* p, int width, int size, int nBlocks, int nThreads);
void show_info_gpu_card();
int main(int argc, char* argv[])
{
    show_info_gpu_card();
    printf("Comienza el juego de la vida:\n");
    int number_blocks = 1;
    int number_rows = 32;
    int number_columns = 32;
    char execution_mode = 'a';
    if (argc == 2)
    {
        execution_mode = argv[1][0];
    }
    else if (argc == 3)
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
    }
    else if (argc >= 4)
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
        number_columns = atoi(argv[3]);
        
    }
    int size = number_rows * number_columns;
    int width = number_columns;
    if (size <= 1024)
    {
        int counter = 1;
        char* a = (char*)malloc(size * sizeof(char));
        char* b = (char*)malloc(size * sizeof(char));
        generate_matrix(a, size, number_blocks, size);
        printf("Situacion Inicial:\n");
        for (int i = 0; i < size; i++)//Representación matriz inicial
        {
            if (i % width == width - 1)
            {
                printf("%c\n", a[i]);
            }
            else
            {
                printf("%c ", a[i]);
            }
        }
        while (true)
        {
            if (counter % 2 == 1)
            {
                step_life(a, b, width, size, number_blocks, size);
                printf("Matriz paso %d:\n", counter);
                for (int i = 0; i < size; i++)//Representación matriz inicial
                {
                    if (i % width == width - 1)
                    {
                        printf("%c\n", b[i]);
                    }
                    else
                    {
                        printf("%c ", b[i]);
                    }
                }
            }
            else
            {
                step_life(b, a, width, size, number_blocks, size);
                printf("Matriz paso %d:\n", counter);
                for (int i = 0; i < size; i++)//Representación matriz inicial
                {
                    if (i % width == width - 1)
                    {
                        printf("%c\n", a[i]);
                    }
                    else
                    {
                        printf("%c ", a[i]);
                    }
                }
            }
            counter++;
            if (execution_mode == 'm') //Si el modo seleccionado no es automático para hasta que el usuario pulse una tecla
            {
                getchar();
            }
        }

        free(a);
        free(b);

    }
    else 
    {
        printf("Las dimensiones de la matriz introducidas no son válidas.\n");
    }
    getchar();
    getchar();
    return 0;
}

void generate_matrix(char* m, int size, int nBlocks, int nThreads)
{
    srand(time(NULL));
    int seed = rand() % 50000;
    char* m_d;
    int numElem = generate_random(1, size*0.15);
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    prepare_matrix <<<nBlocks, nThreads >>> (m_d);
    make_rand <<<nBlocks, numElem >>> (seed, m_d, size);
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
}

void step_life(char* m, char* p, int width, int size, int nBlocks, int nThreads)
{
    char* m_d;
    char* p_d;
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMalloc((void**)&p_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, size * sizeof(char), cudaMemcpyHostToDevice);
    matrix_operation <<<nBlocks, nThreads >>> (m_d, p_d, width, size);
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, p_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
    cudaFree(p_d);
}

int generate_random(int min, int max)
{
    srand(time(NULL));
    int randNumber = rand() % (max - min) + min;
    return randNumber;
}

void show_info_gpu_card()
{
    cudaDeviceProp prop;

    int count;
    //Obtención número de dispositivos compatibles con CUDA
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Numero de dispositivos compatibles con CUDA: %d.\n", count);

    //Obtención de características relativas a cada dispositivo
    for (int i = 0; i < count; i++)
    {
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
        printf("Informacion general del dispositivo %d compatible con CUDA:\n", i + 1);
        printf("Nombre GPU: %s.\n", prop.name);
        printf("Capacidad de computo: %d,%d.\n", prop.major, prop.minor);
        printf("Velocidad de reloj: %d kHz.\n", prop.clockRate);
        printf("Copia solapada dispositivo: ");
        if (prop.deviceOverlap)
        {
            printf("Activo.\n");
        }
        else
        {
            printf("Inactivo.\n");
        }
        printf("Timeout de ejecucion del Kernel: ");
        if (prop.kernelExecTimeoutEnabled)
        {
            printf("Activo.\n");
        }
        else
        {
            printf("Inactivo.\n");
        }

        printf("\nInformacion de memoria para el dispositivo %d:\n", i + 1);
        printf("Memoria global total: %zu GB.\n", prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("Memoria constante total: %zu Bytes.\n", prop.totalConstMem);
        printf("Memoria compartida por bloque: %zu Bytes.\n", prop.sharedMemPerBlock);
        printf("Ancho del bus de memoria global: &d.\n", prop.memoryBusWidth);
        printf("Numero registros compartidos por bloque: %d.\n", prop.regsPerBlock);
        printf("Numero hilos maximos por bloque: %d.\n", prop.maxThreadsPerBlock);
        printf("Memoria compartida por multiprocesador: %zu Bytes.\n", prop.sharedMemPerMultiprocessor);
        printf("Numero registros compartidos por multiprocesador: %d.\n", prop.regsPerMultiprocessor);
        printf("Numero hilos maximos por multiprocesador: %d.\n", prop.maxThreadsPerMultiProcessor);
        printf("Numero de hilos en warp: %d.\n", prop.warpSize);
        printf("Alineamiento maximo de memoria: %zu.\n", prop.memPitch);
        printf("Textura de alineamiento: %zd.\n", prop.textureAlignment);
        printf("Total de multiprocesadores: %d.\n", prop.multiProcessorCount);
        printf("Maximas dimensiones de un hilo: (%d, %d, %d).\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximas dimensiones de grid: (%d, %d, %d).\n\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    }
    getchar();
}
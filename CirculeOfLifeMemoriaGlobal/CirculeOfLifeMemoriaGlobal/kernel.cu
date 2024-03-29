#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <time.h>
#include "../common/book.h"

//Elabora un n�mero aleatorio
__global__ void make_rand(int seed, char* m, int size) {
    float myrandf;
    int num;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, idx, 0, &state); //Se prepara la ejecuci�n del random de CUDA
    myrandf = curand_uniform(&state);
    myrandf *= (size - 0 + 0.999999);
    num = myrandf;
    if (m[num] == 'O')
    {
        m[num] = 'X';
    }
}
//Se da el valor inicial de las distintas casillas de la matriz
__global__ void prepare_matrix(char* p)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    p[idx] = 'O';
}

//Se genera una matriz de manera que los elementos bajan una fila
__global__ void matrix_operation(char* m, char* p, int width, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Id del hilo seg�n su bloque y su posici�n en el mismo e �ndice de la posici�n trabaja de las distintas matrices
    int counter = 0; // Contador del n�mero de posiciones con c�lulas vivas
    if ((idx % width != 0) && (idx - width >= 0) && (m[idx - width - 1] == 'X')) // Estudia si existe esquina superior izquierda y si tiene una c�lula viva
    {
        counter++;
    }
    if ((idx % width != 0) && (m[idx - 1] == 'X')) //Estudia si existe el casilla en el lateral izquierdo y si tiene una c�lula viva
    {
        counter++;
    }
    if ((idx - width >= 0) && (m[idx - width] == 'X')) //Estudia si existe el casilla en el lateral superior y si tiene una c�lula viva
    {
        counter++;
    }
    if ((idx % width != width-1) && (idx - width >= 0) && (m[idx - width + 1] == 'X')) // Estudia si existe esquina superior derecha y si tiene una c�lula viva
    {
        counter++;
    }
    if ((idx % width != width - 1) && (m[idx + 1] == 'X')) //Estudia si existe el casilla en el lateral derecho y si tiene una c�lula viva
    {
        counter++;
    }
    if ((idx % width != 0) && (idx + width < size) && (m[idx + width - 1] == 'X')) // Estudia si existe esquina inferior izquierda y si tiene una c�lula viva
    {
        counter++;
    }
    if ((idx + width < size) && (m[idx + width] == 'X')) //Estudia si existe el casilla en el lateral inferior y si tiene una c�lula viva
    {
        counter++;
    }
    if ((idx % width != width - 1) && (idx + width < size) && (m[idx + width + 1] == 'X')) // Estudia si existe esquina inferior derecha y si tiene una c�lula viva
    {
        counter++;
    }
    if ((counter == 3) && (m[idx] == 'O')) // Una c�lula muerte se convierte en viva si tiene 3 c�lulas vivas alrededor de ella
    {
        p[idx] = 'X';
    }
    else if (((counter < 2) || (counter > 3)) && (m[idx] == 'X')) // Una c�lula viva se convierte en muerte si alrededor de ella hay un n�mero de c�lulas distinto de 2 o 3
    {
        p[idx] = 'O';
    }
    else //La c�lula mantiene su estado
    {
        p[idx] = m[idx];
    }
}


void generate_matrix(char* m, int size, int nBlocks, int nThreads);
int generate_random(int min, int max);
void step_life(char* m, char* p, int width, int size, int nBlocks, int nThreads);
void show_info_gpu_card();
int get_max_number_threads_block();
int main(int argc, char* argv[])
{
    show_info_gpu_card(); // Muestra la informaci�n de la tarjeta gr�fica
    int maxThreads = get_max_number_threads_block(); // Devuelve el n�mero m�ximo de hilos que se pueden ejecutar por bloque
    printf("Comienza el juego de la vida:\n");
    int number_blocks = 1; //N�mero de bloques
    int number_rows = 32; // N�mero de elementos por fila
    int number_columns = 32; // N�mero de elementos por columna
    char execution_mode = 'a'; // Modo de ejecuci�n del programa
    // Condiciones para los casos en los que se est� pasando por terminal una serie de par�metros
    if (argc == 2)//Consideraci�n si solo se pasan dos par�metros por consola
    {
        execution_mode = argv[1][0];
    }
    else if (argc == 3)//Consideraci�n si solo se pasan tres par�metros por consola
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
    }
    else if (argc >= 4)//Consideraci�n si solo se pasan cuatro o m�s par�metros por consola
    {
        execution_mode = argv[1][0];
        number_rows = atoi(argv[2]);
        number_columns = atoi(argv[3]);
        
    }
    int size = number_rows * number_columns; //Tama�o de la matriz
    int width = number_columns; //Ancho del bloque
    if (size <= maxThreads)
        // Si el tama�o de la matriz es inferior o igual al m�ximo n�mero de hilos por bloque que admite la GPU 
    {
        int counter = 1;//Contador de n�mero de paso de matriz del juego
        char* a = (char*)malloc(size * sizeof(char));
        char* b = (char*)malloc(size * sizeof(char));
        generate_matrix(a, size, number_blocks, size);
        printf("Situacion Inicial:\n");
        for (int i = 0; i < size; i++)//Representaci�n matriz inicial
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
        while (execution_mode == 'm' || execution_mode == 'a')
        {
            if (counter % 2 == 1)
            {
                step_life(a, b, width, size, number_blocks, size);
                printf("Matriz paso %d:\n", counter);
                for (int i = 0; i < size; i++)//Representaci�n matriz inicial
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
                for (int i = 0; i < size; i++)//Representaci�n matriz inicial
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
            if (execution_mode == 'm') //Si el modo seleccionado no es autom�tico para hasta que el usuario pulse una tecla
            {
                getchar();
            }
        }
        if (execution_mode != 'm' && execution_mode != 'a')//Si el modo ejecuci�n es distinto a los propuestos parar el programa
        {
            printf("El modo de ejecucion del programa es incorrecto.\n");
        }

        free(a);
        free(b);

    }
    else // Si el tama�o de la matriz es mayor al m�ximo n�mero de hilos por bloque que admite la GPU 
    {
        printf("Las dimensiones de la matriz introducidas no son validas.\n");
    }
    getchar();
    getchar();
    return 0;
}

void generate_matrix(char* m, int size, int nBlocks, int nThreads)
// Genera la matriz con su estado inicial
{
    srand(time(NULL));
    int seed = rand() % 50000;
    char* m_d;
    int numElem = generate_random(1, size*0.15);// Genera un n�mero aleatorio de m�xima n�mero de c�lulas vivas en la etapa inicial siendo el m�ximo un 15% del m�ximo n�mero de casillas
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    prepare_matrix <<<nBlocks, nThreads >>> (m_d);//Prepara la matriz con todas las casillas con c�lulas muertas
    make_rand <<<nBlocks, numElem >>> (seed, m_d, size);// Va colocando de forma aleatoria c�lulas vivas en las casillas de la matriz
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
}

void step_life(char* m, char* p, int width, int size, int nBlocks, int nThreads)
// Genera la matriz resultado a partir de una matriz inicial con las restricciones marcadas para cada casilla
{
    char* m_d;
    char* p_d;
    cudaMalloc((void**)&m_d, size * sizeof(char));
    cudaMalloc((void**)&p_d, size * sizeof(char));
    cudaMemcpy(m_d, m, size * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, size * sizeof(char), cudaMemcpyHostToDevice);
    matrix_operation <<<nBlocks, nThreads >>> (m_d, p_d, width, size);// Estudia el cambio o no de valor de las distintas casillas de la matriz 
    cudaDeviceSynchronize();
    cudaMemcpy(m, m_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(p, p_d, size * sizeof(char), cudaMemcpyDeviceToHost);
    cudaFree(m_d);
    cudaFree(p_d);
}

int generate_random(int min, int max)// Genera un n�mero aleatorio entre un m�nimo y un m�ximo
{
    srand(time(NULL));
    int randNumber = rand() % (max - min) + min;
    return randNumber;
}

int get_max_number_threads_block()// Devuelve el n�mero m�ximo de hilos que se pueden ejecutar por bloque
{
    cudaDeviceProp prop;

    int count;
    //Obtenci�n n�mero de dispositivos compatibles con CUDA
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
    return prop.maxThreadsPerBlock;

}


void show_info_gpu_card()// Muestra las caracter�sticas de la tarjeta gr�fica usada
{
    cudaDeviceProp prop;

    int count;
    //Obtenci�n n�mero de dispositivos compatibles con CUDA
    HANDLE_ERROR(cudaGetDeviceCount(&count));
    printf("Numero de dispositivos compatibles con CUDA: %d.\n", count);

    //Obtenci�n de caracter�sticas relativas a cada dispositivo
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
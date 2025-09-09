// #include <buddy/Core/Container.h>

// using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef MATMUL

#if MATMUL == 1
#define I 32
#define K 32
#define J 32

#elif MATMUL == 2
#define I 64
#define J 64
#define K 64

#elif MATMUL == 3
#define I 128
#define J 128
#define K 128

#elif MATMUL == 4
#define I 256
#define J 256
#define K 256

#elif MATMUL == 5
#define I 512
#define J 512 
#define K 512 

#elif MATMUL == 6
#define I 1024
#define J 1024 
#define K 1024 
#endif
#endif

#ifndef MATMUL
#define MATMUL 0
#define I 1
#define K 1
#define J 1
#endif

#ifdef CONV
#define BATCH_SIZE 1
#define IN_CHANNELS 1
#define OUT_CHANNELS 1
#define IN_DIM 256

#if CONV == 1
#define KERNEL_DIM 3
#define OUT_DIM 254

#elif CONV == 2
#define KERNEL_DIM 5
#define OUT_DIM 252

#elif CONV == 3
#define KERNEL_DIM 7
#define OUT_DIM 250

#elif CONV == 4
#define KERNEL_DIM 9
#define OUT_DIM 248

#elif CONV == 5
#define KERNEL_DIM 11
#define OUT_DIM 246

#elif CONV == 6
#define KERNEL_DIM 13
#define OUT_DIM 244

#endif
#endif

#ifndef CONV
#define CONV 0
#define BATCH_SIZE 0
#define IN_CHANNELS 0
#define OUT_CHANNELS 0
#define KERNEL_DIM 0
#define IN_DIM 0
#define OUT_DIM 0
#endif

// If DIALECT is 1,we use linalg dialect,otherwise we use gemmini.
#ifndef DIALECT 
#define DIALECT 0
#endif

// Simple descriptor for MemRef data
// Based on MLIR's typical MemRef descriptor format
struct MemRefDescriptor {
  void *allocated;     // Base pointer to allocated memory (or nullptr if not allocated)
  void *aligned;       // Aligned pointer to the data
  int64_t offset;      // Offset from aligned to the start of the data
  int64_t sizes[4];    // Size of each dimension (for up to 4D tensors)
  int64_t strides[4];  // Stride for each dimension
};

// Helper function to setup a 2D MemRef descriptor for a pre-allocated array
static void setupMemRef2D(MemRefDescriptor *desc, void *data, int64_t dim0, int64_t dim1) {
  desc->allocated = data;
  desc->aligned = data;
  desc->offset = 0;
  desc->sizes[0] = dim0;
  desc->sizes[1] = dim1;
  desc->strides[0] = dim1;
  desc->strides[1] = 1;
}

// Helper function to setup a 4D MemRef descriptor for a pre-allocated array
static void setupMemRef4D(MemRefDescriptor *desc, void *data, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) {
  desc->allocated = data;
  desc->aligned = data;
  desc->offset = 0;
  desc->sizes[0] = dim0;
  desc->sizes[1] = dim1;
  desc->sizes[2] = dim2;
  desc->sizes[3] = dim3;
  desc->strides[0] = dim1 * dim2 * dim3;
  desc->strides[1] = dim2 * dim3;
  desc->strides[2] = dim3;
  desc->strides[3] = 1;
}

extern "C" {
// Declare MLIR C interface functions that expect MemRef descriptors
void _mlir_ciface_linalg_matmul1(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output);
void _mlir_ciface_linalg_matmul2(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output);
void _mlir_ciface_linalg_matmul3(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output);
void _mlir_ciface_linalg_matmul4(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output);
void _mlir_ciface_linalg_matmul5(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output);
void _mlir_ciface_linalg_matmul6(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output);

void _mlir_ciface_gemmini_matmul1(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output, MemRefDescriptor *bias);
void _mlir_ciface_gemmini_matmul2(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output, MemRefDescriptor *bias);
void _mlir_ciface_gemmini_matmul3(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output, MemRefDescriptor *bias);
void _mlir_ciface_gemmini_matmul4(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output, MemRefDescriptor *bias);
void _mlir_ciface_gemmini_matmul5(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output, MemRefDescriptor *bias);
void _mlir_ciface_gemmini_matmul6(MemRefDescriptor *input0, MemRefDescriptor *input1, MemRefDescriptor *output, MemRefDescriptor *bias);

void _mlir_ciface_linalg_conv1(MemRefDescriptor *input, MemRefDescriptor *kernel, MemRefDescriptor *output);
void _mlir_ciface_linalg_conv2(MemRefDescriptor *input, MemRefDescriptor *kernel, MemRefDescriptor *output);
void _mlir_ciface_linalg_conv3(MemRefDescriptor *input, MemRefDescriptor *kernel, MemRefDescriptor *output);
void _mlir_ciface_linalg_conv4(MemRefDescriptor *input, MemRefDescriptor *kernel, MemRefDescriptor *output);
void _mlir_ciface_linalg_conv5(MemRefDescriptor *input, MemRefDescriptor *kernel, MemRefDescriptor *output);
void _mlir_ciface_linalg_conv6(MemRefDescriptor *input, MemRefDescriptor *kernel, MemRefDescriptor *output);

void _mlir_ciface_gemmini_conv1(MemRefDescriptor *input, MemRefDescriptor *weights, MemRefDescriptor *bias, MemRefDescriptor *output);
void _mlir_ciface_gemmini_conv2(MemRefDescriptor *input, MemRefDescriptor *weights, MemRefDescriptor *bias, MemRefDescriptor *output);
void _mlir_ciface_gemmini_conv3(MemRefDescriptor *input, MemRefDescriptor *weights, MemRefDescriptor *bias, MemRefDescriptor *output);
void _mlir_ciface_gemmini_conv4(MemRefDescriptor *input, MemRefDescriptor *weights, MemRefDescriptor *bias, MemRefDescriptor *output);
void _mlir_ciface_gemmini_conv5(MemRefDescriptor *input, MemRefDescriptor *weights, MemRefDescriptor *bias, MemRefDescriptor *output);
void _mlir_ciface_gemmini_conv6(MemRefDescriptor *input, MemRefDescriptor *weights, MemRefDescriptor *bias, MemRefDescriptor *output);

static uint64_t readCycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}
}

int main() {
  if (MATMUL) {
    if (DIALECT == 1) {
      // Allocate and initialize input arrays
      int8_t input0_data[I][K];
      int8_t input1_data[K][J];
      int8_t output_data[I][J];
      
      // Initialize input arrays
      for (int i = 0; i < I; i++) {
        for (int k = 0; k < K; k++) {
          input0_data[i][k] = 1;
        }
      }
      
      for (int k = 0; k < K; k++) {
        for (int j = 0; j < J; j++) {
          input1_data[k][j] = 2;
        }
      }
      
      // Clear output array
      memset(output_data, 0, sizeof(output_data));
      
      // Setup MemRef descriptors
      MemRefDescriptor input0_desc, input1_desc, output_desc;
      setupMemRef2D(&input0_desc, input0_data, I, K);
      setupMemRef2D(&input1_desc, input1_data, K, J);
      setupMemRef2D(&output_desc, output_data, I, J);
      
      uint64_t start, end;
      switch (MATMUL) {
      case 1:
        start = readCycles();
        _mlir_ciface_linalg_matmul1(&input0_desc, &input1_desc, &output_desc);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_linalg_matmul2(&input0_desc, &input1_desc, &output_desc);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_linalg_matmul3(&input0_desc, &input1_desc, &output_desc);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_linalg_matmul4(&input0_desc, &input1_desc, &output_desc);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_linalg_matmul5(&input0_desc, &input1_desc, &output_desc);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_linalg_matmul6(&input0_desc, &input1_desc, &output_desc);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong matmul test case.\n");
        return 0;
      }
      printf("The linalg.matmul test case is %d\n", MATMUL);
      printf("I = %d K = %d J = %d\n", I, K, J);
      printf("Cycles taken %lu\n", end - start); 
      return 0;
    } else if (DIALECT == 2) {
      // Allocate and initialize input arrays
      int8_t input0_data[I][K];
      int8_t input1_data[K][J];
      int8_t output_data[I][J];
      int32_t bias_data[I][J];
      
      // Initialize input arrays
      for (int i = 0; i < I; i++) {
        for (int k = 0; k < K; k++) {
          input0_data[i][k] = 1;
        }
      }
      
      for (int k = 0; k < K; k++) {
        for (int j = 0; j < J; j++) {
          input1_data[k][j] = 2;
        }
      }
      
      // Clear output and bias arrays
      memset(output_data, 0, sizeof(output_data));
      memset(bias_data, 0, sizeof(bias_data));
      
      // Setup MemRef descriptors
      MemRefDescriptor input0_desc, input1_desc, output_desc, bias_desc;
      setupMemRef2D(&input0_desc, input0_data, I, K);
      setupMemRef2D(&input1_desc, input1_data, K, J);
      setupMemRef2D(&output_desc, output_data, I, J);
      setupMemRef2D(&bias_desc, bias_data, I, J);
      
      uint64_t start, end;
      switch (MATMUL) {
      case 1:
        start = readCycles();
        _mlir_ciface_gemmini_matmul1(&input0_desc, &input1_desc, &output_desc, &bias_desc);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_gemmini_matmul2(&input0_desc, &input1_desc, &output_desc, &bias_desc);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_gemmini_matmul3(&input0_desc, &input1_desc, &output_desc, &bias_desc);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_gemmini_matmul4(&input0_desc, &input1_desc, &output_desc, &bias_desc);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_gemmini_matmul5(&input0_desc, &input1_desc, &output_desc, &bias_desc);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_gemmini_matmul6(&input0_desc, &input1_desc, &output_desc, &bias_desc);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong matmul test case.\n");
        return 0;
      }
      printf("The gemmini.matmul test case is %d\n", MATMUL);
      printf("I = %d K = %d J = %d\n", I, K, J);
      printf("Cycles taken %lu\n", end - start);
      return 0;
    }
  }

  if (CONV) {
    if (DIALECT == 1) {
      // Allocate and initialize input arrays for convolution
      int8_t input_data[BATCH_SIZE][IN_CHANNELS][IN_DIM][IN_DIM];
      int8_t weights_data[OUT_CHANNELS][IN_CHANNELS][KERNEL_DIM][KERNEL_DIM];
      int8_t output_data[BATCH_SIZE][OUT_CHANNELS][OUT_DIM][OUT_DIM];
      
      // Initialize input arrays
      for (int n = 0; n < BATCH_SIZE; n++) {
        for (int c = 0; c < IN_CHANNELS; c++) {
          for (int h = 0; h < IN_DIM; h++) {
            for (int w = 0; w < IN_DIM; w++) {
              input_data[n][c][h][w] = 1;
            }
          }
        }
      }
      
      for (int oc = 0; oc < OUT_CHANNELS; oc++) {
        for (int ic = 0; ic < IN_CHANNELS; ic++) {
          for (int kh = 0; kh < KERNEL_DIM; kh++) {
            for (int kw = 0; kw < KERNEL_DIM; kw++) {
              weights_data[oc][ic][kh][kw] = 1;
            }
          }
        }
      }
      
      // Clear output array
      memset(output_data, 0, sizeof(output_data));
      
      // Setup MemRef descriptors
      MemRefDescriptor input_desc, weights_desc, output_desc;
      setupMemRef4D(&input_desc, input_data, BATCH_SIZE, IN_CHANNELS, IN_DIM, IN_DIM);
      setupMemRef4D(&weights_desc, weights_data, OUT_CHANNELS, IN_CHANNELS, KERNEL_DIM, KERNEL_DIM);
      setupMemRef4D(&output_desc, output_data, BATCH_SIZE, OUT_CHANNELS, OUT_DIM, OUT_DIM);
      
      uint64_t start, end;
      switch (CONV) {
      case 1:
        start = readCycles();
        _mlir_ciface_linalg_conv1(&input_desc, &weights_desc, &output_desc);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_linalg_conv2(&input_desc, &weights_desc, &output_desc);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_linalg_conv3(&input_desc, &weights_desc, &output_desc);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_linalg_conv4(&input_desc, &weights_desc, &output_desc);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_linalg_conv5(&input_desc, &weights_desc, &output_desc);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_linalg_conv6(&input_desc, &weights_desc, &output_desc);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong conv test case.\n");
        return 0;
      }
      printf("The linalg.conv test case is %d\n", CONV);
      printf("BATCH_SIZE = %d IN_CHANNELS = %d OUT_CHANNELS = %d IN_DIM = %d "
            "KERNEL_DIM = %d OUT_DIM = %d\n",
            BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS, IN_DIM, KERNEL_DIM, OUT_DIM);
      printf("Cycles taken = %lu\n", end - start);
      return 0;
    } else if(DIALECT == 2) {
      // Allocate and initialize input arrays
      int8_t input_data[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS];
      int8_t weights_data[KERNEL_DIM * KERNEL_DIM][1];
      int32_t bias_data[OUT_CHANNELS];
      int8_t output_data[OUT_DIM * OUT_DIM][1];
      
      // Initialize input arrays
      for (int n = 0; n < BATCH_SIZE; n++) {
        for (int h = 0; h < IN_DIM; h++) {
          for (int w = 0; w < IN_DIM; w++) {
            for (int c = 0; c < IN_CHANNELS; c++) {
              input_data[n][h][w][c] = 1;
            }
          }
        }
      }
      
      for (int i = 0; i < KERNEL_DIM * KERNEL_DIM; i++) {
        weights_data[i][0] = 1;
      }
      
      // Clear output and bias arrays
      memset(output_data, 0, sizeof(output_data));
      memset(bias_data, 0, sizeof(bias_data));
      
      // Setup MemRef descriptors
      MemRefDescriptor input_desc, weights_desc, bias_desc, output_desc;
      setupMemRef4D(&input_desc, input_data, BATCH_SIZE, IN_DIM, IN_DIM, IN_CHANNELS);
      setupMemRef2D(&weights_desc, weights_data, KERNEL_DIM * KERNEL_DIM, 1);
      // For 1D bias, manually adjust the descriptor
      bias_desc.allocated = bias_data;
      bias_desc.aligned = bias_data;
      bias_desc.offset = 0;
      bias_desc.sizes[0] = OUT_CHANNELS;
      bias_desc.strides[0] = 1;
      setupMemRef2D(&output_desc, output_data, OUT_DIM * OUT_DIM, 1);
      
      uint64_t start, end;
      switch (CONV) {
      case 1:
        start = readCycles();
        _mlir_ciface_gemmini_conv1(&input_desc, &weights_desc, &bias_desc, &output_desc);
        end = readCycles();
        break;
      case 2:
        start = readCycles();
        _mlir_ciface_gemmini_conv2(&input_desc, &weights_desc, &bias_desc, &output_desc);
        end = readCycles();
        break;
      case 3:
        start = readCycles();
        _mlir_ciface_gemmini_conv3(&input_desc, &weights_desc, &bias_desc, &output_desc);
        end = readCycles();
        break;
      case 4:
        start = readCycles();
        _mlir_ciface_gemmini_conv4(&input_desc, &weights_desc, &bias_desc, &output_desc);
        end = readCycles();
        break;
      case 5:
        start = readCycles();
        _mlir_ciface_gemmini_conv5(&input_desc, &weights_desc, &bias_desc, &output_desc);
        end = readCycles();
        break;
      case 6:
        start = readCycles();
        _mlir_ciface_gemmini_conv6(&input_desc, &weights_desc, &bias_desc, &output_desc);
        end = readCycles();
        break;
      default:
        printf("You specify the wrong conv test case.\n");
        return 0;
      }
      printf("The gemmini.conv test case is %d\n", CONV);
      printf("BATCH_SIZE = %d IN_CHANNELS = %d OUT_CHANNELS = %d IN_DIM = %d "
            "KERNEL_DIM = %d OUT_DIM = %d\n",
            BATCH_SIZE, IN_CHANNELS, OUT_CHANNELS, IN_DIM, KERNEL_DIM, OUT_DIM);
      printf("Cycles taken = %lu\n", end - start);
      return 0;
    }
  }
  return 0;
}

#include "gemmini.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// -----------------------------------------------------------------------------
// 参数化的配置
// -----------------------------------------------------------------------------

// 这些参数应该在编译时定义或通过函数参数传入
#ifndef MATMUL_NN
#define MATMUL_NN 12544
#endif

#ifndef MATMUL_MM  
#define MATMUL_MM 256
#endif

#ifndef MATMUL_KK
#define MATMUL_KK 64
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

// 计算分块参数
#define N_TILES (MATMUL_NN / TILE_SIZE)
#define M_TILES (MATMUL_MM / TILE_SIZE)
#define K_TILES (MATMUL_KK / TILE_SIZE)

// 访存步长
#define A_STRIDE MATMUL_KK
#define B_STRIDE MATMUL_MM
#define C_STRIDE MATMUL_MM

// -----------------------------------------------------------------------------
// 工具函数
// -----------------------------------------------------------------------------

static uint64_t read_cycles() {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// 计算矩阵访存偏移的辅助函数
static inline uint64_t get_matrix_offset(int row, int col, int stride) {
    return (uint64_t)(row * stride + col);
}

// 计算tile地址的辅助函数
static inline uint64_t get_tile_address(uint64_t base_addr, int tile_row, int tile_col, 
                                        int tiles_per_row, int tile_size) {
    return base_addr + (tile_row * tiles_per_row + tile_col) * tile_size * tile_size;
}

// -----------------------------------------------------------------------------
// 参数化的内存管理
// -----------------------------------------------------------------------------

#define GEMM_HEAP_SIZE 100000
#define GEMM_DIM 16

typedef struct __attribute__((__packed__)) NewBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} NewBlock;

NewBlock BLOCKS[GEMM_HEAP_SIZE / sizeof(NewBlock)];
uint32_t gemm_last_ptr;

void gemm_init_mem() {
  for (uint32_t i = 0; i < sizeof(BLOCKS); i++)
    ((uint8_t *)BLOCKS)[i] = 0;
  gemm_last_ptr = 0;
}

uint32_t gemm_malloc(long unsigned int size) {
  if (size == 0)
    return -1;
  size = (size + GEMM_DIM - 1) / GEMM_DIM;
  int i;
  for (i = 0; i < GEMM_HEAP_SIZE / sizeof(NewBlock) && BLOCKS[i].size > 0; i++) {
    if (BLOCKS[i].is_used)
      continue;
    if (BLOCKS[i].size < size)
      continue;
    break;
  }
  if (BLOCKS[i].size == 0) {
    BLOCKS[i].loc = gemm_last_ptr;
    BLOCKS[i].size = size;
    BLOCKS[i].is_used = 1;
    gemm_last_ptr += size;
    return BLOCKS[i].loc;
  }
  BLOCKS[i].is_used = 1;
  return BLOCKS[i].loc;
}

void gemm_free(uint32_t addr) {
  for (int i = 0; BLOCKS[i].size > 0; i++) {
    if (BLOCKS[i].is_used && BLOCKS[i].loc == addr) {
      BLOCKS[i].is_used = 0;
      return;
    }
  }
}

// -----------------------------------------------------------------------------
// 累加器内存管理
// -----------------------------------------------------------------------------

#define GEMM_ACC_HEAP_SIZE 100000
#define GEMM_ACC_DIM 16

typedef struct __attribute__((__packed__)) AccBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} AccBlock;

#define N_ACC_BLOCKS (GEMM_ACC_HEAP_SIZE / sizeof(AccBlock))
AccBlock ACC_BLOCKS[N_ACC_BLOCKS];
uint32_t gemm_acc_free_block;

void gemm_acc_init_mem() {
  uint8_t *buf = (uint8_t *)ACC_BLOCKS;
  for (uint32_t i = 0; i < sizeof(ACC_BLOCKS); i++)
    buf[i] = 0;
  gemm_acc_free_block = 0;
}

uint32_t gemm_acc_malloc(long unsigned int size) {
  if (size == 0)
    return -1;
  if (gemm_acc_free_block >= N_ACC_BLOCKS)
    return -1;

  size = (size + GEMM_ACC_DIM - 1) / GEMM_ACC_DIM;
  uint32_t i = gemm_acc_free_block;

  uint32_t loc = 0;
  if (i > 0) {
    loc = ACC_BLOCKS[i - 1].loc + ACC_BLOCKS[i - 1].size;
  }

  ACC_BLOCKS[i].size = size;
  ACC_BLOCKS[i].loc = loc;
  ACC_BLOCKS[i].is_used = 1;
  gemm_acc_free_block = i + 1;

  return (ACC_BLOCKS[i].loc | ((uint32_t)0x80000000));
}

void gemm_acc_free(uint32_t addr) {
  if (gemm_acc_free_block == 0)
    return;
  addr = addr & (uint32_t)(0x7FFFFFFF);
  
  if (ACC_BLOCKS[gemm_acc_free_block - 1].loc == addr) {
    ACC_BLOCKS[gemm_acc_free_block - 1].is_used = 0;
    for (int i = gemm_acc_free_block - 1; i >= 0; i--) {
      if (ACC_BLOCKS[i].is_used)
        break;
      gemm_acc_free_block = i;
    }
  } else {
    for (int i = gemm_acc_free_block - 1; i >= 0; i--) {
      if (ACC_BLOCKS[i].loc == addr) {
        ACC_BLOCKS[i].is_used = 0;
        break;
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 参数化的matmul kernel
// -----------------------------------------------------------------------------

void matmul_flexible(int NN, int MM, int KK, const float* scale, bool act, 
                    const int8_t* A, const int8_t* B, int8_t* C) {
  // 验证输入参数
  assert(NN > 0 && MM > 0 && KK > 0);
  assert(NN % TILE_SIZE == 0 && MM % TILE_SIZE == 0 && KK % TILE_SIZE == 0);
  
  // 计算分块参数
  int n_tiles = NN / TILE_SIZE;
  int m_tiles = MM / TILE_SIZE;
  int k_tiles = KK / TILE_SIZE;
  
  // 配置gemmini
  gemmini_extended_config_st((MM), (act), (scale)[0]);
  gemmini_extended_config_ex(WS, 0, 0, 1, 0, 0);
  gemmini_extended3_config_ld((MM), 1.0f, 0, 2);
  gemmini_extended3_config_ld((KK), 1.0f, 0, 1);
  gemmini_extended3_config_ld(0, 1.0f, 0, 0);

  // 分配内存
  int8_t *a = (int8_t*) ((uint64_t)gemm_malloc(TILE_SIZE * TILE_SIZE * k_tiles * n_tiles * sizeof(int8_t)));
  int8_t *b = (int8_t*) ((uint64_t)gemm_malloc(TILE_SIZE * TILE_SIZE * k_tiles * m_tiles * sizeof(int8_t)));
  int32_t *res = (int32_t*) ((uint32_t)gemm_acc_malloc(TILE_SIZE * TILE_SIZE * m_tiles * sizeof(int32_t)));

  // 计算外层循环参数
  int outer_tiles = (n_tiles + 3) / 4;  // 向上取整
  int inner_tiles = 4;
  
  for (int outer_i = 0; outer_i < outer_tiles; outer_i++) {
    int start_i = outer_i * inner_tiles;
    int end_i = (start_i + inner_tiles > n_tiles) ? n_tiles : start_i + inner_tiles;
    
    for (int i = start_i; i < end_i; i++) {
      for (int j = 0; j < m_tiles; j++) {
        
        // 初始化结果tiles
        for (int k_sub = 0; k_sub < k_tiles; k_sub++) {
          uint64_t res_addr = ((uint32_t)((uint64_t)res)) + 
                             ((j * k_tiles + k_sub) * TILE_SIZE * TILE_SIZE) / 16;
          gemmini_extended_mvin(0, res_addr, TILE_SIZE, TILE_SIZE);
        }
        
        // 加载A tiles (只在j==0时加载)
        if (j == 0) {
          for (int k_sub = 0; k_sub < k_tiles; k_sub++) {
            uint64_t a_src = (uint64_t)&A[get_matrix_offset(i * TILE_SIZE, k_sub * TILE_SIZE, KK)];
            uint64_t a_dst = ((uint32_t)((uint64_t)a)) + 
                           ((i * k_tiles + k_sub) * TILE_SIZE * TILE_SIZE) / 16;
            gemmini_extended_mvin2(a_src, a_dst, TILE_SIZE * k_tiles, TILE_SIZE);
          }
        }
        
        // 加载B tiles (只在第一次迭代时加载)
        if (outer_i == 0 && i == start_i) {
          for (int k_sub = 0; k_sub < k_tiles; k_sub++) {
            uint64_t b_src = (uint64_t)&B[get_matrix_offset(k_sub * TILE_SIZE, j * TILE_SIZE, MM)];
            uint64_t b_dst = ((uint32_t)((uint64_t)b)) + 
                           ((j * k_tiles + k_sub) * TILE_SIZE * TILE_SIZE) / 16;
            gemmini_extended_mvin3(b_src, b_dst, TILE_SIZE * k_tiles, TILE_SIZE);
          }
        }
        
        // 执行计算
        for (int k_sub = 0; k_sub < k_tiles; k_sub++) {
          uint32_t a_addr = (uint32_t)((uint64_t)a) + 
                           ((i * k_tiles + k_sub) * TILE_SIZE * TILE_SIZE) / 16;
          uint32_t b_addr = (uint32_t)((uint64_t)b) + 
                           ((j * k_tiles + k_sub) * TILE_SIZE * TILE_SIZE) / 16;
          uint32_t res_addr = (uint32_t)((uint64_t)res) + 
                             ((j * k_tiles + k_sub) * TILE_SIZE * TILE_SIZE) / 16;
          
          gemmini_extended_preload(b_addr, res_addr | 0x40000000, 
                                  TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE);
          gemmini_extended_compute_preloaded(a_addr, ~((uint32_t)0), 
                                           TILE_SIZE, TILE_SIZE, TILE_SIZE, TILE_SIZE);
        }
        
        // 写回结果
        for (int k_sub = 0; k_sub < k_tiles; k_sub++) {
          uint64_t c_dst = (uint64_t)&C[get_matrix_offset(i * TILE_SIZE, 
                                                          j * TILE_SIZE + k_sub * TILE_SIZE, MM)];
          uint32_t res_src = (uint32_t)((uint64_t)res) + 
                            ((j * k_tiles + k_sub) * TILE_SIZE * TILE_SIZE) / 16;
          gemmini_extended_mvout(c_dst, res_src, TILE_SIZE, TILE_SIZE);
        }
      }
    }
  }
  
  // 释放内存
  gemm_acc_free((uint32_t)(res));
  gemm_free((uint64_t)(b));
  gemm_free((uint64_t)(a));
}

// 为了保持向后兼容，保留原始函数签名
void matmul_4(const float* scale, bool act, const int8_t* A, const int8_t* B, int8_t* C) {
  matmul_flexible(MATMUL_NN, MATMUL_MM, MATMUL_KK, scale, act, A, B, C);
}

// -----------------------------------------------------------------------------
// 测试代码
// -----------------------------------------------------------------------------

int test_matmul_flexible(int NN, int MM, int KK) {
  printf("Testing matmul_flexible with shape (%d, %d, %d)\n", NN, MM, KK);
  
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  // 分配测试数据
  int8_t* x = (int8_t*)malloc(NN * KK * sizeof(int8_t));
  int8_t* y = (int8_t*)malloc(KK * MM * sizeof(int8_t));
  int8_t* z = (int8_t*)malloc(NN * MM * sizeof(int8_t));
  
  if (!x || !y || !z) {
    printf("Memory allocation failed\n");
    return -1;
  }

  // 初始化测试数据
  for (int i = 0; i < NN * KK; i++) {
    x[i] = (i % 127) + 1;
  }
  for (int i = 0; i < KK * MM; i++) {
    y[i] = (i % 127) + 1;
  }
  for (int i = 0; i < NN * MM; i++) {
    z[i] = 0;
  }

  float scale[1] = {1.0f};
  
  unsigned long start = read_cycles();
  matmul_flexible(NN, MM, KK, scale, false, x, y, z);
  gemmini_fence();
  unsigned long end = read_cycles();
  
  printf("Cycles for shape (%d, %d, %d): %ld\n", NN, MM, KK, end - start);
  
  // 释放内存
  free(x);
  free(y);
  free(z);
  
  return 0;
}

// -----------------------------------------------------------------------------
// 主函数
// -----------------------------------------------------------------------------

int main() {
  printf("Testing flexible matmul kernel\n");
  
  // 测试不同的shapes
  int test_shapes[][3] = {
    {64, 64, 64},
    {128, 128, 64},
    {256, 256, 64},
    {512, 256, 64},
    {1024, 256, 64},
    {12544, 256, 64},  // 原始shape
  };
  
  int num_tests = sizeof(test_shapes) / sizeof(test_shapes[0]);
  
  for (int i = 0; i < num_tests; i++) {
    int NN = test_shapes[i][0];
    int MM = test_shapes[i][1];
    int KK = test_shapes[i][2];
    
    // 检查是否能被16整除
    if (NN % 16 == 0 && MM % 16 == 0 && KK % 16 == 0) {
      test_matmul_flexible(NN, MM, KK);
    } else {
      printf("Skipping shape (%d, %d, %d) - not divisible by 16\n", NN, MM, KK);
    }
  }
  
  printf("\nAll tests completed\n");
  return 0;
} 
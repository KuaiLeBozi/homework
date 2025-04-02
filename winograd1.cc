#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>

#include "utils.h"

#define NUM_THREADS1 64
#define NUM_THREADS2 64
void image_transform(float *__restrict__ packed_image,
                     float *__restrict__ V,
                     const V_shape_t vs,
                     const tiling_info_t ti,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_image_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  typedef float(*V_tensor_t)[ti.tile_in_w][collapsed_dim_size];
  packed_image_tensor_t packed_image_tensor = (packed_image_tensor_t)packed_image;
  V_tensor_t V_tensor = (V_tensor_t)V;

  float z0, z1, z2, z3, z4, z5, z6;

  #pragma omp parallel for schedule(static) num_threads(NUM_THREADS1) collapse(2)
  for (int64_t w = 0; w < ti.tile_in_w; ++w) {
    for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
      // 将z6预先读取到寄存器中以减少内存访问
      float z6_0 = packed_image_tensor[0][w][idx];
      float z6_1 = packed_image_tensor[1][w][idx];
      float z6_2 = packed_image_tensor[2][w][idx];
      float z6_3 = packed_image_tensor[3][w][idx];
      float z6_4 = packed_image_tensor[4][w][idx];
      float z6_5 = packed_image_tensor[5][w][idx];

      // 计算中间结果
      float z0 = 4.0f * z6_0;
      float z1 = -4.0f * z6_1;
      float z2 = 4.0f * z6_1;
      float z3 = -2.0f * z6_1;
      float z4 = 2.0f * z6_1;
      float z5 = 4.0f * z6_1;

      // 累积计算
      z0 += -5.0f * z6_2 + z6_4;
      z1 += -4.0f * z6_2 + z6_3 + z6_4;
      z2 += -4.0f * z6_2 - z6_3 + z6_4;
      z3 += -z6_2 + 2.0f * z6_3 + z6_4;
      z4 += -z6_2 - 2.0f * z6_3 + z6_4;
      z5 += -5.0f * z6_3 + z6_5;

      // 一次性写回结果
      V_tensor[0][w][idx] = z0;
      V_tensor[1][w][idx] = z1;
      V_tensor[2][w][idx] = z2;
      V_tensor[3][w][idx] = z3;
      V_tensor[4][w][idx] = z4;
      V_tensor[5][w][idx] = z5;
    }
  }

  // 针对第二个循环做类似优化
  #pragma omp parallel for schedule(static) num_threads(NUM_THREADS1) collapse(2)
  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
      float z6_0 = V_tensor[h][0][idx];
      float z6_1 = V_tensor[h][1][idx];
      float z6_2 = V_tensor[h][2][idx];
      float z6_3 = V_tensor[h][3][idx];
      float z6_4 = V_tensor[h][4][idx];
      float z6_5 = V_tensor[h][5][idx];

      float z0 = 4.0f * z6_0;
      float z1 = -4.0f * z6_1;
      float z2 = 4.0f * z6_1;
      float z3 = -2.0f * z6_1;
      float z4 = 2.0f * z6_1;
      float z5 = 4.0f * z6_1;

      z0 += -5.0f * z6_2 + z6_4;
      z1 += -4.0f * z6_2 + z6_3 + z6_4;
      z2 += -4.0f * z6_2 - z6_3 + z6_4;
      z3 += -z6_2 + 2.0f * z6_3 + z6_4;
      z4 += -z6_2 - 2.0f * z6_3 + z6_4;
      z5 += -5.0f * z6_3 + z6_5;

      V_tensor[h][0][idx] = z0;
      V_tensor[h][1][idx] = z1;
      V_tensor[h][2][idx] = z2;
      V_tensor[h][3][idx] = z3;
      V_tensor[h][4][idx] = z4;
      V_tensor[h][5][idx] = z5;
    }
  }
}

void filter_transform(float *__restrict__ packed_filter,
                     float *__restrict__ U,
                     const filter_shape_t fs,
                     const U_shape_t us,
                     const int64_t collapsed_dim_size) {
  typedef float(*packed_filter_tensor_t)[fs.w][collapsed_dim_size];
  typedef float(*U_tensor_t)[us.w][collapsed_dim_size];
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;
  U_tensor_t U_tensor = (U_tensor_t)U;

  float z0, z1, z2, z3, z4, z5, z6;
  
  // 修改并行化策略，在最外层合并两个维度
  #pragma omp parallel for collapse(2) schedule(static) num_threads(NUM_THREADS1)
  for (int64_t w = 0; w < fs.w; ++w) {
    for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
      z6 = packed_filter_tensor[0][w][idx];
      
      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = packed_filter_tensor[1][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = packed_filter_tensor[2][w][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      // 单独处理每个线程的U_tensor写入，避免竞争
      U_tensor[0][w][idx] = z0;
      U_tensor[1][w][idx] = z1;
      U_tensor[2][w][idx] = z2;
      U_tensor[3][w][idx] = z3;
      U_tensor[4][w][idx] = z4;
      U_tensor[5][w][idx] = z5;
    }
  }

  // 第二次变换也采用类似的并行化策略
  #pragma omp parallel for collapse(2) schedule(static) num_threads(NUM_THREADS1)
  for (int64_t h = 0; h < us.h; ++h) {
    for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
      z6 = U_tensor[h][0][idx];
      
      z0 = (1.0f / 4.0f) * z6;
      z1 = (-1.0f / 6.0f) * z6;
      z2 = (-1.0f / 6.0f) * z6;
      z3 = (1.0f / 24.0f) * z6;
      z4 = (1.0f / 24.0f) * z6;

      z6 = U_tensor[h][1][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (1.0f / 6.0f) * z6;
      z3 += (1.0f / 12.0f) * z6;
      z4 += (-1.0f / 12.0f) * z6;

      z6 = U_tensor[h][2][idx];

      z1 += (-1.0f / 6.0f) * z6;
      z2 += (-1.0f / 6.0f) * z6;
      z3 += (1.0f / 6.0f) * z6;
      z4 += (1.0f / 6.0f) * z6;
      z5 = z6;

      U_tensor[h][0][idx] = z0;
      U_tensor[h][1][idx] = z1;
      U_tensor[h][2][idx] = z2;
      U_tensor[h][3][idx] = z3;
      U_tensor[h][4][idx] = z4;
      U_tensor[h][5][idx] = z5;
    }
  }
}

void output_transform(float *__restrict__ M,
  float *__restrict__ Y,
  const tiling_info_t ti,
  const int64_t collapsed_dim_size) {
typedef float(*M_tensor_t)[ti.tile_in_w][collapsed_dim_size];
typedef float(*Y_tensor_t)[ti.tile_in_w][collapsed_dim_size];
M_tensor_t M_tensor = (M_tensor_t)M;
Y_tensor_t Y_tensor = (Y_tensor_t)Y;
float z0, z1, z2, z3, z4;

#pragma omp parallel for private(z0, z1, z2, z3, z4) schedule(static) num_threads(NUM_THREADS1)
for (int64_t idx = 0; idx < collapsed_dim_size; idx++) {
#pragma omp simd aligned(M_tensor, Y_tensor: 64)
for (int64_t w = 0; w < ti.tile_in_w; ++w) {
z4 = M_tensor[0][w][idx];
z0 = z4;

z4 = M_tensor[1][w][idx];
z0 = z0 + z4;
z1 = z4;
z2 = z4;
z3 = z4;

z4 = M_tensor[2][w][idx];
z0 += z4;
z1 += -z4;
z2 += z4;
z3 += -z4;

z4 = M_tensor[3][w][idx];
z0 += z4;
z1 += 2.0f * z4;
z2 += 4.0f * z4;
z3 += 8.0f * z4;

z4 = M_tensor[4][w][idx];
z0 += z4;
z1 += -2.0f * z4;
z2 += 4.0f * z4;
z3 += -8.0f * z4;

z4 = M_tensor[5][w][idx];
z3 += z4;

Y_tensor[0][w][idx] = z0;
Y_tensor[1][w][idx] = z1;
Y_tensor[2][w][idx] = z2;
Y_tensor[3][w][idx] = z3;
}
#pragma omp simd aligned(M_tensor, Y_tensor: 64) 
for (int64_t h = 0; h < ti.tile_out_h; ++h) {
z4 = Y_tensor[h][0][idx];

z0 = z4;

z4 = Y_tensor[h][1][idx];
z0 += z4;
z1 = z4;
z2 = z4;
z3 = z4;

z4 = Y_tensor[h][2][idx];
z0 += z4;
z1 += -z4;
z2 += z4;
z3 += -z4;

z4 = Y_tensor[h][3][idx];
z0 += z4;
z1 += 2.0f * z4;
z2 += 4.0f * z4;
z3 += 8.0f * z4;

z4 = Y_tensor[h][4][idx];
z0 += z4;
z1 += -2.0f * z4;
z2 += 4.0f * z4;
z3 += -8.0f * z4;

z4 = Y_tensor[h][5][idx];

z3 += z4;

Y_tensor[h][0][idx] = z0;
Y_tensor[h][1][idx] = z1;
Y_tensor[h][2][idx] = z2;
Y_tensor[h][3][idx] = z3;
}
}
}

// 对 filter_packing 采用缓存分块优化，减小内存访存延迟
void filter_packing(float *__restrict__ filter, float *__restrict__ packed_filter, const filter_shape_t fs) {
  typedef float(*filter_tensor_t)[fs.ic][fs.h][fs.w];
  typedef float(*packed_filter_tensor_t)[fs.w][fs.oc][fs.ic];
  filter_tensor_t filter_tensor = (filter_tensor_t)filter;
  packed_filter_tensor_t packed_filter_tensor = (packed_filter_tensor_t)packed_filter;

  const int TILE_SIZE = 128; // 可根据实际情况调整块大小
  
  #pragma omp parallel for schedule(static) num_threads(NUM_THREADS1) collapse(4)
  for (int64_t h0 = 0; h0 < fs.h; h0 += TILE_SIZE) {
    for (int64_t w0 = 0; w0 < fs.w; w0 += TILE_SIZE) {
      for (int64_t oc0 = 0; oc0 < fs.oc; oc0 += TILE_SIZE) {
        for (int64_t ic0 = 0; ic0 < fs.ic; ic0 += TILE_SIZE) {
          for (int64_t h = h0; h < std::min(fs.h, h0 + TILE_SIZE); h++) {
            for (int64_t w = w0; w < std::min(fs.w, w0 + TILE_SIZE); w++) {
              for (int64_t oc = oc0; oc < std::min(fs.oc, oc0 + TILE_SIZE); oc++) {
                for (int64_t ic = ic0; ic < std::min(fs.ic, ic0 + TILE_SIZE); ic++) {
                  packed_filter_tensor[h][w][oc][ic] = filter_tensor[oc][ic][h][w];
                }//地址连续，减少内存访问延迟
              }
            }
          }
        }
      }
    }
  }
}

void image_packing(float *__restrict__ image,
  float *__restrict__ packed_image,
  const image_shape_t is,
  const tiling_info_t ti) {
typedef float(*packedImage_tensor_t)[ti.tile_in_w][ti.num_tiles][is.ic];
typedef float(*image_tensor_t)[is.ic][is.h][is.w];
packedImage_tensor_t packed_image_tensor = (packedImage_tensor_t)packed_image;
image_tensor_t image_tensor = (image_tensor_t)image;

#pragma omp parallel for schedule(static) num_threads(NUM_THREADS1) collapse(4) 
for (int64_t tile = 0; tile < ti.num_tiles; tile++) {
  for (int64_t ic = 0; ic < is.ic; ic++) {
    for (int64_t h = 0; h < ti.tile_in_h; ++h) {
      for (int64_t w = 0; w < ti.tile_in_w; ++w) {
        tile_index_t tidx = get_tile_index(tile, ti);
        int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
        if (hh * 4 + h < is.h && ww * 4 + w < is.w)
          packed_image_tensor[h][w][tile][ic] = image_tensor[batch][ic][(hh * 4 + h)][(ww * 4 + w)];
        else
          packed_image_tensor[h][w][tile][ic] = 0;
        }
      }
    }
  }
}

void output_unpacking_store(float *__restrict__ Y,
              float *__restrict__ out,
              const out_shape_t os,
              const tiling_info_t ti) {
  typedef float(*Y_tensor_t)[ti.tile_in_w][os.oc][ti.num_tiles];
  typedef float(*out_tensor_t)[os.oc][os.h][os.w];
  Y_tensor_t Y_tensor = (Y_tensor_t)Y;
  out_tensor_t out_tensor = (out_tensor_t)out;

  #pragma omp parallel for schedule(static) num_threads(NUM_THREADS2) collapse(3)
  for (int oc = 0; oc < os.oc; ++oc) {
    for (int h = 0; h < ti.tile_out_h; ++h) {
        for (int w = 0; w < ti.tile_out_w; ++w) {
        #pragma omp simd aligned(Y_tensor, out_tensor: 64)
        for (int64_t tile = 0; tile < ti.num_tiles; ++tile) {
          tile_index_t tidx = get_tile_index(tile, ti);
          int64_t batch = tidx.b, ww = tidx.tw, hh = tidx.th;
          if (hh * 4 + h < os.h && ww * 4 + w < os.w) {
          out_tensor[batch][oc][(hh * 4 + h)][(ww * 4 + w)] = Y_tensor[h][w][oc][tile];
          }
        }
    }
  }
  }
}




void winograd_convolution(
    float *__restrict__ image, /**< float [batch_num][input_channel_num][image_height][image_width] */
    const int image_height,
    const int image_width,
    const int input_channel_num,
    float *__restrict__ filter, /**< float [output_channel_num][input_channel_num][FLT_H][FLT_W] */
    const int output_channel_num,
    const int batch_num,
    float *__restrict__ out) {
  /* new vars of shape */
  const image_shape_t is = {.bs = batch_num, .ic = input_channel_num, .h = image_height, .w = image_width};
  const filter_shape_t fs = {.oc = output_channel_num, .ic = input_channel_num, .h = FLT_H, .w = FLT_W};
  const out_shape_t os = get_output_shape(is, fs);
  const tiling_info_t ti = get_tiling_info(is, os);
  const U_shape_t us = get_U_shape(fs, ti);
  const V_shape_t vs = get_V_shape(is, ti);

  float *packed_filter = (float *)aligned_alloc(64,sizeof(float) * fs.h * fs.w * fs.oc * fs.ic);
  float *packed_image = (float *)aligned_alloc(64,sizeof(float) * ti.tile_in_h * ti.tile_in_w * ti.num_tiles * is.ic);
  float *U = (float *)aligned_alloc(64,sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * us.ic);
  float *V = (float *)aligned_alloc(64,sizeof(float) * ti.tile_in_h * ti.tile_in_w * vs.num_tiles * vs.ic);
  float *M = (float *)aligned_alloc(64,sizeof(float) * ti.tile_in_h * ti.tile_in_w * us.oc * vs.num_tiles);
  float *Y = (float *)aligned_alloc(64,sizeof(float) * ti.tile_out_h * ti.tile_in_w * os.oc * ti.num_tiles);

  filter_packing(filter, packed_filter, fs);
  filter_transform(packed_filter, U, fs, us, us.oc * us.ic);

  image_packing(image, packed_image, is, ti);
  image_transform(packed_image, V, vs, ti, vs.ic * vs.num_tiles);

  // 定义分块大小
  const int BLOCK_SIZE = 256;  // 根据实际缓存大小调整
  
  #pragma omp parallel for schedule(dynamic, 4) collapse(4) 
  for (int64_t h = 0; h < ti.tile_in_h; ++h) {
    for (int64_t w = 0; w < ti.tile_in_w; ++w) {
      for (int64_t tile_block = 0; tile_block < vs.num_tiles; tile_block += BLOCK_SIZE) {
        for (int64_t oc_block = 0; oc_block < us.oc; oc_block += BLOCK_SIZE) {

          

          typedef float(*U_tensor_t)[ti.tile_in_w][us.oc][us.ic];
          typedef float(*V_tensor_t)[ti.tile_in_w][vs.num_tiles][vs.ic];
          typedef float(*M_tensor_t)[ti.tile_in_w][us.oc][vs.num_tiles];
          U_tensor_t U_tensor = (U_tensor_t)U;
          V_tensor_t V_tensor = (V_tensor_t)V;
          M_tensor_t M_tensor = (M_tensor_t)M;

          // 计算当前块的实际大小
          const int tile_block_size = std::min(BLOCK_SIZE, (int)(vs.num_tiles - tile_block));
          const int oc_block_size = std::min(BLOCK_SIZE, (int)(us.oc - oc_block));

          // 使用局部缓存来存储块计算结果
          float local_sum[BLOCK_SIZE][BLOCK_SIZE] = {0};
          

          // 对输入通道进行分块计算
          
          for (int64_t ic = 0; ic < us.ic; ic += BLOCK_SIZE) {
            const int ic_block_size = std::min(BLOCK_SIZE, (int)(us.ic - ic));
            
            #pragma omp simd collapse(2) 
            for (int64_t tile_i = 0; tile_i < tile_block_size; ++tile_i) {
              for (int64_t oc_i = 0; oc_i < oc_block_size; ++oc_i) {
                float sum = 0.0f;

                #pragma omp simd reduction(+:sum) aligned(U_tensor, V_tensor: 64) 
                for (int64_t ic_i = 0; ic_i < ic_block_size; ++ic_i) {
                  sum += V_tensor[h][w][tile_block + tile_i][ic + ic_i] * 
                         U_tensor[h][w][oc_block + oc_i][ic + ic_i];
                }

                local_sum[tile_i][oc_i] += sum;
              }
            }
          }

           // 原始写回循环被优化为先转置 local_sum 后利用 memcpy 批量拷贝
           {
            float local_sum_t[BLOCK_SIZE * BLOCK_SIZE];
            
            // 转置 local_sum，使每个 oc_i 对应的 tile 值连续存储
            #pragma omp simd aligned(local_sum: 64)
            for (int64_t oc_i = 0; oc_i < oc_block_size; ++oc_i) {
                for (int64_t tile_i = 0; tile_i < tile_block_size; ++tile_i) {
                    local_sum_t[oc_i * tile_block_size + tile_i] = local_sum[tile_i][oc_i];
                }
            }
            // 对于每个 oc_i ，直接 memcpy 拷贝连续内存块
            for (int64_t oc_i = 0; oc_i < oc_block_size; ++oc_i) {
                memcpy(&M_tensor[h][w][oc_block + oc_i][tile_block],
                       &local_sum_t[oc_i * tile_block_size],
                       tile_block_size * sizeof(float));}
           
          
        }
      }
    }
  }
}

  output_transform(M, Y, ti, us.oc * vs.num_tiles);
  output_unpacking_store(Y, out, os, ti);

  free(packed_filter);
  free(packed_image);
  free(U);
  free(V);
  free(M);
  free(Y);
}

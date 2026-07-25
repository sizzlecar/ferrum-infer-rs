#pragma once

#include <stddef.h>
#include <stdint.h>

#define FERRUM_MARLIN_ABI_VERSION 1u

typedef enum FerrumMarlinScalarType {
  FERRUM_MARLIN_SCALAR_F16 = 1,
  FERRUM_MARLIN_SCALAR_BF16 = 2,
  FERRUM_MARLIN_SCALAR_S8 = 3,
  FERRUM_MARLIN_SCALAR_U4 = 4,
  FERRUM_MARLIN_SCALAR_U4B8 = 5,
  FERRUM_MARLIN_SCALAR_U8B128 = 6,
  FERRUM_MARLIN_SCALAR_FE2M1F = 7,
  FERRUM_MARLIN_SCALAR_FE4M3FN = 8,
} FerrumMarlinScalarType;

enum FerrumMarlinLaunchFlags {
  FERRUM_MARLIN_HAS_BIAS = 1u << 0,
  FERRUM_MARLIN_HAS_ACT_ORDER = 1u << 1,
  FERRUM_MARLIN_IS_K_FULL = 1u << 2,
  FERRUM_MARLIN_HAS_ZERO_POINTS = 1u << 3,
  FERRUM_MARLIN_USE_ATOMIC_ADD = 1u << 4,
  FERRUM_MARLIN_USE_FP32_REDUCE = 1u << 5,
  FERRUM_MARLIN_ZERO_POINTS_ARE_FLOAT = 1u << 6,
};

typedef struct FerrumMarlinLaunch {
  uint32_t abi_version;
  uint32_t struct_size;

  const void* a;
  const void* b;
  void* c;
  void* c_tmp;
  void* b_bias;
  void* a_scales;
  void* b_scales;
  void* global_scale;
  void* zero_points;
  void* group_index;
  void* permutation;
  void* a_tmp;
  void* workspace;
  void* stream;

  int32_t prob_m;
  int32_t prob_n;
  int32_t prob_k;
  int32_t lda;

  int32_t a_type;
  int32_t b_type;
  int32_t c_type;
  int32_t scale_type;

  int32_t num_groups;
  int32_t group_size;
  int32_t device;
  int32_t thread_k_init;
  int32_t thread_n_init;
  int32_t sms;

  uint32_t flags;
  uint32_t reserved;
} FerrumMarlinLaunch;

#ifdef __cplusplus
static_assert(sizeof(void*) == 8, "Ferrum Marlin requires a 64-bit ABI");
static_assert(sizeof(FerrumMarlinLaunch) == 184,
              "FerrumMarlinLaunch ABI size changed");
static_assert(offsetof(FerrumMarlinLaunch, a) == 8,
              "FerrumMarlinLaunch pointer layout changed");
static_assert(offsetof(FerrumMarlinLaunch, prob_m) == 120,
              "FerrumMarlinLaunch shape layout changed");
static_assert(offsetof(FerrumMarlinLaunch, flags) == 176,
              "FerrumMarlinLaunch flag layout changed");

extern "C" {
#endif

void ferrum_marlin_mm(const FerrumMarlinLaunch* launch);

#ifdef __cplusplus
}
#endif

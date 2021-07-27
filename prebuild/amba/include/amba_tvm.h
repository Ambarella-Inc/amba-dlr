/*******************************************************************************
 * amba_tvm.h
 *
 * History:
 *    2020/05/06  - [Monica Yang] created
 *
 * Copyright [2020] Ambarella International LP.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
******************************************************************************/

#ifndef _AMBA_TVM_H_
#define _AMBA_TVM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define AMBA_TVM_LIB_MAJOR 0
#define AMBA_TVM_LIB_MINOR 0
#define AMBA_TVM_LIB_PATCH 1
#define AMBA_TVM_LIB_VERSION ((AMBA_TVM_LIB_MAJOR << 16) | \
			(AMBA_TVM_LIB_MINOR << 8)  | \
			AMBA_TVM_LIB_PATCH)

#ifndef AMBA_API
#define AMBA_API __attribute__((visibility("default")))
#endif

#ifndef TVM_NET_NAME_MAX
#define TVM_NET_NAME_MAX		(64)
#endif

#define IN
#define OUT
#define INOUT

typedef struct {
	IN const char* engine_name;
	IN const char* engine_filepath;
	INOUT unsigned long engine_id;
	uint32_t reserve[11];
} amba_engine_cfg_t;

typedef struct {
	void* data_virt;
	uint32_t device_type;
	int32_t device_id;
	int32_t ndim;
	uint8_t dtype_code;
	uint8_t dtype_bits;
	uint16_t dtype_lanes;
	int64_t* shape;
	int64_t* strides;
	uint64_t byte_offset;
	uint32_t size;		// tensor size without padding
	uint32_t reserve[7];
} AmbaDLTensor;

typedef struct {
	AmbaDLTensor *tensors;
	const char** names;
	uint32_t num;
	uint32_t reserve[11];
} amba_engine_io_t;

typedef struct {
	uint32_t cvflow_time_us;
	uint32_t reserve[7];
} amba_perf_t;

AMBA_API int GetAmbaTVMLibVersion(void);

AMBA_API int InitAmbaTVM(void);
AMBA_API int InitAmbaEngine(amba_engine_cfg_t * engine_cfg,
	amba_engine_io_t * engine_input,amba_engine_io_t * engine_output);
AMBA_API int SetAmbaEngineInput(amba_engine_cfg_t *engine_cfg,
	const char *input_name, AmbaDLTensor *input);
AMBA_API int RunAmbaEngine(amba_engine_cfg_t * engine_cfg,
	amba_perf_t *perf);
AMBA_API int GetAmbaEngineOutput(amba_engine_cfg_t *engine_cfg,
	const char *output_name, AmbaDLTensor *output);
AMBA_API int DeleteAmbaTVM(amba_engine_cfg_t *engine_cfgs, uint32_t num);

AMBA_API int CheckAmbaEngineInputName(amba_engine_cfg_t * engine_cfg,
	const char * input_name);
AMBA_API int CheckAmbaEngineOutputName(amba_engine_cfg_t * engine_cfg,
	const char * output_name);

AMBA_API int ConfigAmbaEngineLocation(const char *dirpath);

AMBA_API void* AmbaDeviceAlloc(unsigned long nbytes, unsigned long alignment);
AMBA_API int AmbaDeviceFree(void* ptr);


#ifdef __cplusplus
}
#endif

#endif



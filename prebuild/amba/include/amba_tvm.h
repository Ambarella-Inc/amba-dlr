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
	INOUT uint32_t engine_id;
	IN const char* engine_name;
	IN const char* engine_filepath;
} amba_engine_cfg_t;

typedef struct {
	void *data_virt;	// should be float*
	uint32_t data_phys;
	uint8_t ndim;
	uint8_t bits;		// should always be float (bits==32)
	uint8_t reserve[2];
	int64_t *shape;
	uint32_t size;		// tensor size
} AmbaDLTensor;

typedef struct {
	uint32_t cvflow_time_us;
} amba_perf_t;

AMBA_API int GetAmbaTVMLibVersion(void);

AMBA_API int InitAmbaTVM(void);
AMBA_API int InitAmbaEngine(amba_engine_cfg_t * engine_cfg);
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


#ifdef __cplusplus
}
#endif

#endif



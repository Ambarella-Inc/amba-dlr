/*******************************************************************************
 * test_amba_dlr.cpp
 *
 * History:
 *    2022/05/20  - [Monica Yang] created
 *
 * Copyright [2022] Ambarella International LP.
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

// dlr
#include "dlr.h"
#include "dlr_common.h"

// system
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <string>
#include <signal.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <pthread.h>

// Amba SDK
#include "amba_tvm.h"

#define DLR_APP_MAJOR	(1)
#define DLR_APP_MINOR	(11)
#define DLR_APP_PATCH	(0)

#define FILENAME_LENGTH	(2048)
#define NAME_LENGTH	(32)
#define MAX_NET_NUM	(8)
#define MAX_IO_NUM		(16)

#ifndef ALIGN_32_BYTE
#define ALIGN_32_BYTE(x) ((((x) + 31) >> 5) << 5)
#endif

typedef struct {
	int type;
	int id;
}dlr_dev_t;

typedef struct dlr_io_cfg_s {
	char io_name[NAME_LENGTH];
	char io_fn[FILENAME_LENGTH];
} dlr_io_cfg_t;

typedef struct dlr_net_cfg_s {
	char model_dir[FILENAME_LENGTH];
	uint32_t input_num;
	dlr_io_cfg_t input_node[MAX_IO_NUM];
} dlr_net_cfg_t;

typedef struct {
	uint32_t net_num;
	dlr_net_cfg_t net_cfg[MAX_NET_NUM];

	dlr_dev_t dev;
	uint32_t show_io;

	uint32_t print_time;
}dlr_ctx_t;

#ifndef NO_ARG
#define NO_ARG (0)
#endif

#ifndef HAS_ARG
#define HAS_ARG (1)
#endif

typedef enum {
	SHOW_IO = 0,
} tvm_option_t;

static struct option long_options[] = {
	{"mod-dir",	HAS_ARG, 0, 'b'},
	{"in",	HAS_ARG, 0, 'i'},
	{"ifile",	HAS_ARG, 0, 'f'},

	{"print-time",	NO_ARG, 0, 'e'},
	{"show-io",	NO_ARG, 0, SHOW_IO},

	{"help",	NO_ARG, 0, 'h'},
	{0, 0, 0, 0},
};

static const char *short_options = "b:i:f:eh";

struct hint_s {
	const char *arg;
	const char *str;
};

static const struct hint_s hint[] = {
	{"", "\tFolder path that contains compiled files;"
		"Basename of all compiled files should be the same; One folder for each model."},
	{"", "\t\tName of input node. Use multiple -i if there are more than one input nodes."
		"Order of names should be the same as those in compiled.json file."},
	{"", "\tBinary file for network input with float format. Only for file mode and should be preprocessed."},

	{"", "\tEnable time print. Default is disable."},
	{"", "\tShow primary i/o info of compiled artifacts."},

	{"", "\tprint help info"},
};

static void usage(void)
{
	const char *itself = "test_amba_dlr";
	uint32_t i = 0;

	printf("%s usage:\n", itself);
	for (i = 0; i < sizeof(long_options) / sizeof(long_options[0]) - 1; i++) {
		if (isalpha(long_options[i].val))
			printf("-%c ", long_options[i].val);
		else
			printf("   ");
		printf("--%s", long_options[i].name);
		if (hint[i].arg[0] != 0)
			printf(" [%s]", hint[i].arg);
		printf("\t%s\n", hint[i].str);
	}
	printf("\nExamples:\n");

	printf("1. Show model i/o info.\n"
		"\t# %s -b model_folder -i input_name --show-io\n", itself);
	printf("2. Run one model in file mode.\n"
		"\t# %s -b model_folder -i data -f in_img.bin\n", itself);
}

static int init_param(int argc, char **argv, dlr_ctx_t *p_ctx)
{
	int ch = 0, value = 0, net_idx = 0, net_num = 0, in_idx = 0;
	int option_index = 0;
	opterr = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "-mod-dir")) {
			++p_ctx->net_num;
		}
	}
	if (p_ctx->net_num > 8) {
		printf("Error: only support %d net modes at most.\n", MAX_NET_NUM);
		return -1;
	}
	for (uint32_t i = 0; i < p_ctx->net_num; ++ i) {
		memset(p_ctx->net_cfg[i].model_dir, 0, FILENAME_LENGTH);
		memset(p_ctx->net_cfg[i].input_node, 0, MAX_IO_NUM * sizeof(dlr_io_cfg_t));
	}

	p_ctx->dev.type = kDLAmba;	// default device type is Amba Device
	p_ctx->dev.id = 0;			// default device id is 0
	while ((ch = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
		switch (ch) {
		case 'b':
			value = strlen(optarg);
			if (value >= FILENAME_LENGTH) {
				printf("Error: Filename [%s] is too long [%u] (>%u).\n", optarg,
					value, FILENAME_LENGTH);
				return -1;
			}
			net_idx = net_num;
			snprintf(p_ctx->net_cfg[net_idx].model_dir, sizeof(p_ctx->net_cfg[net_idx].model_dir), "%s", optarg);
			++ net_num;
			break;
		case 'i':
			value = strlen(optarg);
			if (value >= NAME_LENGTH) {
				printf("Error: Filename [%s] is too long [%u] (>%u).\n", optarg,
					value, NAME_LENGTH);
				return -1;
			}
			in_idx = p_ctx->net_cfg[net_idx].input_num;
			if (in_idx >= MAX_IO_NUM) {
				printf("IO pair number is too much: %u > %u.\n", in_idx, MAX_IO_NUM);
				return -1;
			}
			snprintf(p_ctx->net_cfg[net_idx].input_node[in_idx].io_name, sizeof(p_ctx->net_cfg[net_idx].input_node[in_idx].io_name), "%s", optarg);
			++ p_ctx->net_cfg[net_idx].input_num;
			break;
		case 'f':
			value = strlen(optarg);
			if (value >= FILENAME_LENGTH) {
				printf("Filename [%s] is too long [%d] (>%d).\n", optarg,
					value, FILENAME_LENGTH);
				return -1;
			}
			snprintf(p_ctx->net_cfg[net_idx].input_node[in_idx].io_fn, sizeof(p_ctx->net_cfg[net_idx].input_node[in_idx].io_fn), "%s", optarg);
			break;
		case 'e':
			p_ctx->print_time = 1;
			break;
		case SHOW_IO:
			p_ctx->show_io = 1;
			break;
		case 'h':
			usage();
			return -1;
		default:
			printf("Error: unknown option found: %c\n", ch);
			return -1;
		}
	}

	if (p_ctx->net_num == 0) {
		printf("Error: please select at least one model by -b\n");
		return -1;
	}
	for (uint32_t i = 0; i < p_ctx->net_num; ++i) {
		if (p_ctx->net_cfg[i].input_num == 0) {
			printf("Error: pleaes provide input name by -i\n");
			return -1;
		}
	}

	return 0;
}

static int check_dlr_version(void)
{
	int rval = 0;

	if (DLR_MAKE_VERSION(DLR_APP_MAJOR, DLR_APP_MINOR, DLR_APP_PATCH) < DLR_VERSION) {
		printf("Error: DLR version should not be less than (%d, %d, %d)\n", DLR_APP_MAJOR,
			DLR_APP_MINOR, DLR_APP_PATCH);
		rval = -1;
	}

	return rval;
}

static int dlr_DLTensor_string2datatype(DLTensor *t, const char *type)
{
	int rval = 0;

	/*
	* dtype.code: 0: int; 1: uint; 2: float
	* dtype.bits: 8, 16, 32, 64
	*/
	if (strcmp(type, "int8") == 0) {
		t->dtype.code = 0;
		t->dtype.bits = 8;
	} else if (strcmp(type, "int16") == 0) {
		t->dtype.code = 0;
		t->dtype.bits = 16;
	} else if (strcmp(type, "int32") == 0) {
		t->dtype.code = 0;
		t->dtype.bits = 32;
	} else if (strcmp(type, "int64") == 0) {
		t->dtype.code = 0;
		t->dtype.bits = 64;
	} else if (strcmp(type, "uint8") == 0) {
		t->dtype.code = 1;
		t->dtype.bits = 8;
	} else if (strcmp(type, "uint16") == 0) {
		t->dtype.code = 1;
		t->dtype.bits = 16;
	} else if (strcmp(type, "uint32") == 0) {
		t->dtype.code = 1;
		t->dtype.bits = 32;
	} else if (strcmp(type, "uint64") == 0) {
		t->dtype.code = 1;
		t->dtype.bits = 64;
	} else if (strcmp(type, "float32") == 0) {
		t->dtype.code = 2;
		t->dtype.bits = 32;
	} else if (strcmp(type, "bool") == 0) {
		t->dtype.code = 1;
		t->dtype.bits = 1;
	}else {
		printf("Error: unknown DLTensor data type code %s\n", type);
		rval = -1;
	}

	return rval;
}

static int dlr_DLTensor_datatype2string(DLTensor *t, std::string &type)
{
	int rval = 0;

	/*
	* dtype.code: 0: int; 1: uint; 2: float
	* dtype.bits: 8, 16, 32, 64
	*/
	if (t->dtype.code == 0) {
		if (t->dtype.bits == 8) {
			type = "int8";
		} else if (t->dtype.bits == 16) {
			type = "int16";
		} else if (t->dtype.bits == 32) {
			type = "int32";
		} else if (t->dtype.bits == 64) {
			type = "int64";
		} else {
			printf("Error: unknown DLTensor data type code %d bits %d\n", \
				t->dtype.code, t->dtype.bits);
			rval = -1;
		}
	} else if (t->dtype.code == 1) {
		if (t->dtype.bits == 1) {
			type = "bool";
		} else if (t->dtype.bits == 8) {
			type = "uint8";
		} else if (t->dtype.bits == 16) {
			type = "uint16";
		} else if (t->dtype.bits == 32) {
			type = "uint32";
		} else if (t->dtype.bits == 64) {
			type = "uint64";
		} else {
			printf("Error: unknown DLTensor data type code %d bits %d\n", \
				t->dtype.code, t->dtype.bits);
			rval = -1;
		}
	} else if (t->dtype.code == 2) {
		if (t->dtype.bits == 32) {
			type = "float32";
		} else {
			printf("Error: unknown DLTensor data type code %d bits %d\n", \
				t->dtype.code, t->dtype.bits);
			rval = -1;
		}
	} else {
		printf("Error: unknown DLTensor data type code %d\n", t->dtype.code);
		rval = -1;
	}

	return rval;
}

static void show_DLTensor_io(DLTensor **t, int num, const std::string &prefix)
{
	for (int i = 0; i < num; ++i) {
		std::string info(prefix), datatype("");
		info += " " + std::to_string(i) + " shape: ";
		for (int j = 0; j < t[i]->ndim; ++j) {
			if (j) info+= ", ";
			info += std::to_string(t[i]->shape[j]);
		}
		dlr_DLTensor_datatype2string(t[i], datatype);
		info += " datatype: " + datatype;
		printf("%s\n", info.c_str());
	}
}

static int dlr_get_DLTensor_size(const DLTensor *t)
{
	int size = 1;
	for(int i = 0; i < t->ndim; ++i) {
		size *= t->shape[i];
	}
	size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;

	return size;
}

static int dlr_read_binary(const char* filename, DLTensor *t)
{
	int rval = 0;
	int size = dlr_get_DLTensor_size(t);
	std::ifstream data_fin(filename, std::ios::binary);

	do {
		data_fin.seekg (0, data_fin.end);

		int file_size = data_fin.tellg();
		if (file_size != size) {
			printf("Error: input file size (%d) should be %d.\n", file_size, size);
			rval = -1;
			break;
		}
		data_fin.seekg (0, data_fin.beg);
		data_fin.read(static_cast<char*>(t->data), size);
	} while(0);
	data_fin.close();

	return rval;
}

static int dlr_process_classification(dlr_ctx_t *p_ctx, DLTensor* out, int num_cls)
{
	const float* out_iter = static_cast<const float*>(out->data);
	auto max_iter = std::max_element(out_iter, out_iter + num_cls);
	auto max_index = std::distance(out_iter, max_iter);

	float prob_sum = 0.0;
	float prob_softmax[num_cls] = {0.0};
	int prob_id[num_cls] = {0};

	for(int i = 0; i < num_cls; ++i) {
		prob_softmax[i] = exp(out_iter[i] - out_iter[max_index]);
		prob_sum += prob_softmax[i];
	}
	for (int i = 0; i < num_cls; ++i) {
		prob_softmax[i] /= prob_sum;
		prob_id[i] = i;
	}
	for (int i = 0; i < 5; ++ i) {
		float max_conf = prob_softmax[i];
		int max_id = i;
		for (int j = i + 1; j < num_cls; ++ j) {
			float conf = prob_softmax[j];
			if (conf > max_conf) {
				max_id = j;
				max_conf = conf;
			}
		}
		if (max_id != i) {
			prob_id[max_id] = i;
			prob_softmax[max_id] = prob_softmax[i];
			prob_id[i] = max_id;
			prob_softmax[i] = max_conf;
		}
	}

	printf("Top 5 categories: %d, %d, %d, %d, %d\n", prob_id[0], prob_id[1], prob_id[2],
		prob_id[3], prob_id[4]);
	printf("Top 5 scores: %.4f, %.4f, %.4f, %.4f, %.4f\n", prob_softmax[0], prob_softmax[1],
		prob_softmax[2], prob_softmax[3], prob_softmax[4]);

	return 0;
}

static int dlr_prepare_file_mode(dlr_net_cfg_t *p_net, DLTensor **in_t)
{
	int rval = 0;

	do {
		for (uint32_t i = 0; i < p_net->input_num; ++i) {
			if (dlr_read_binary(p_net->input_node[i].io_fn, in_t[i]) < 0) {
				printf("Error: dlr_read_binary.\n");
				rval = -1;
				break;
			}
		}
		if (rval) break;
	} while(0);

	return rval;
}

static int dlr_dump_outputs(DLTensor **out_t, int num)
{
	int rval = 0;

	for (int i = 0; i < num; ++i) {
		int out_size = dlr_get_DLTensor_size(out_t[i]);
		std::string out_fn = "out_" + std::to_string(i) + ".bin";
		std::ofstream out_file(out_fn, std::ios::binary);
		out_file.write(static_cast<char*>(out_t[i]->data), out_size);
		out_file.close();
	}

	return rval;
}

static int dlr_process_outputs(dlr_ctx_t *p_ctx,
		DLTensor** out_t, int num_outputs)
{
	int rval = 0;

	do {
		if (dlr_dump_outputs(out_t, num_outputs) < 0) {
			printf("Error: dlr_dump_outputs.\n");
			rval = -1;
			break;
		}
		if (num_outputs == 1) {
			dlr_process_classification(p_ctx, out_t[0],
				out_t[0]->shape[out_t[0]->ndim - 1]);
		}
	} while(0);

	return rval;
}

static int dlr_alloc_input_DLTensor(DLRModelHandle* handle, dlr_ctx_t *p_ctx,
		DLTensor **in_t, int num_inputs)
{
	int rval = 0;

	for (int i = 0; i < num_inputs; ++i) {
		int64_t size = 0;
		int dim = 0;
		GetDLRInputSizeDim(handle, i, &size, &dim);
		int64_t shape[dim];
		GetDLRInputShape(handle, i, shape);
		const char* type = nullptr;
		GetDLRInputType(handle, i,  &type);
		DLTensor t;
		dlr_DLTensor_string2datatype(&t, type);
		TVMArrayAlloc(shape, dim, t.dtype.code, t.dtype.bits, 1,
			p_ctx->dev.type, p_ctx->dev.id, &in_t[i]);
	}

	return rval;
}

static int dlr_free_input_DLTensor(DLTensor **in_t, int num_inputs)
{
	int rval = 0;

	for (int i = 0; i < num_inputs; ++i) {
		if (in_t[i]) {
			TVMArrayFree(in_t[i]);
		}
	}

	return rval;
}

static int dlr_alloc_output_DLTensor(DLRModelHandle* handle,
	DLManagedTensor **out_mt, DLTensor **out_t, int num_outputs)
{
	int rval = 0;

	for (int i = 0; i < num_outputs; ++i) {
		GetDLROutputManagedTensorPtr(handle, i, (const void**)&out_mt[i]);
		out_t[i] = &out_mt[i]->dl_tensor;
	}

	return rval;
}

static int dlr_free_output_DLTensor(DLManagedTensor **out_mt, int num_outputs)
{
	int rval = 0;

	for (int i = 0; i < num_outputs; ++i) {
		if (out_mt[i]) {
			out_mt[i]->deleter(out_mt[i]);
		}
	}

	return rval;
}

typedef struct thread_arg_s {
	void *ctx;
	void *net;
} thread_arg_t;

static void* dlr_execute_one_net(void *args)
{
	struct timeval tv1, tv2;
	unsigned long tv_diff = 0;
	bool has_meta = false;
	dlr_ctx_t *p_ctx = (dlr_ctx_t*)(((thread_arg_t*)args)->ctx);
	dlr_net_cfg_t *p_net = (dlr_net_cfg_t*)(((thread_arg_t*)args)->net);

	if (ConfigAmbaEngineLocation(p_net->model_dir)) {
		printf("Error: ConfigAmbaEngineLocation\n");
		return nullptr;
	}

	int num_outputs = 0, num_inputs = 0;
	DLRModelHandle mod = nullptr;

	CreateDLRModel(&mod, p_net->model_dir, p_ctx->dev.type, p_ctx->dev.id);
	GetDLRNumOutputs(&mod, &num_outputs);
	num_inputs = p_net->input_num;

	// input/output buffers
	DLTensor* in_t[num_inputs] = {nullptr};
	DLTensor* out_t[num_outputs] = {nullptr};
	DLManagedTensor* out_mt[num_outputs] = {nullptr};

	const char* backend = nullptr;
	GetDLRBackend(&mod, &backend);
	printf("DLR backend: %s\n", backend);

	do {
		dlr_alloc_input_DLTensor(&mod, p_ctx, in_t, num_inputs);
		dlr_alloc_output_DLTensor(&mod, out_mt, out_t, num_outputs);

		if (p_ctx->show_io) {
			show_DLTensor_io(in_t, num_inputs, "input");
			show_DLTensor_io(out_t, num_outputs, "output");
			break;
		}

		GetDLRHasMetadata(&mod, &has_meta);
		if (has_meta) {
			for (int i = 0; i < num_outputs; ++i) {
				const char* out_name = nullptr;
				GetDLROutputName(&mod, i, &out_name);
			}
		}

		if (dlr_prepare_file_mode(p_net, in_t) < 0) {
			printf("Error: dlr_prepare_file_mode\n");
			break;
		}

		// set input
		for (int i = 0; i < num_inputs; ++i) {
			SetDLRInputTensorZeroCopy(&mod, p_net->input_node[i].io_name, in_t[i]);
		}

		// run module
		RunDLRModel(&mod);

		// evaluate execution time at second Run()
		if (p_ctx->print_time) {
			gettimeofday(&tv1, NULL);
			RunDLRModel(&mod);
			gettimeofday(&tv2, NULL);
			tv_diff = (unsigned long) 1000000 * (unsigned long) (tv2.tv_sec - tv1.tv_sec) +
				(unsigned long) (tv2.tv_usec - tv1.tv_usec);
			printf("model  \"%s\" execution time: %lu us\n", p_net->model_dir, tv_diff);
		}

		// out_t points to storage pool in TVM
		if (dlr_process_outputs(p_ctx, out_t, num_outputs)< 0) {
			printf("Error: dlr_process_outputs.\n");
			break;
		}
	} while(0);

	dlr_free_input_DLTensor(in_t, num_inputs);
	dlr_free_output_DLTensor(out_mt, num_outputs);
	DeleteDLRModel(&mod);

	return nullptr;
}

static int dlr_run_module(dlr_ctx_t *p_ctx)
{
	int rval = 0;
	uint32_t net_num = p_ctx->net_num;
	int launched_net = 0;

	pthread_t tid[net_num];
	int tret[net_num];
	thread_arg_t thread_arg[net_num];

	for (uint32_t i = 0; i < net_num; i++) {
		tret[i] = -1;
		dlr_net_cfg_t *p_net = &p_ctx->net_cfg[i];
		thread_arg[i].ctx = p_ctx;
		thread_arg[i].net = p_net;
		if ((tret[i] = pthread_create(&tid[i], NULL, dlr_execute_one_net, &thread_arg[i])) < 0) {
			printf("Error: launch network \"%s\".\n", p_net->model_dir);
			break;
		} else {
			printf("Succeed to launch network \"%s\".\n", p_net->model_dir);
			++ launched_net;
		}
	}
	if (launched_net > 0) {
		for (uint32_t i = 0; i < net_num; i++) {
			if (tret[i] == 0) {
				pthread_join(tid[i], NULL);
			}
		}
	}

	return rval;
}

static void sigstop(int)
{
	printf("sigstop msg, exit test_amba_dlr.\n");
}

int main(int argc, char *argv[])
{
	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);

	int rval = 0;
	dlr_ctx_t dlr_ctx;
	dlr_ctx_t *p_ctx = &dlr_ctx;

	memset(p_ctx, 0, sizeof(dlr_ctx));

	do {
		if (check_dlr_version()) {
			printf("check_dlr_version\n");
			rval = -1;
			break;
		}
		if (argc < 2) {
			usage();
			rval = -1;
			break;
		}
		if (init_param(argc, argv, p_ctx)) {
			rval = -1;
			break;
		}
		if (dlr_run_module(p_ctx) < 0) {
			printf("Error: dlr_run_module.\n");
			rval = -1;
			break;
		}
	} while(0);

	return rval;
}


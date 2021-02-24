// dlr
#include "dlr.h"
#include "dlr_tvm.h"
#include "amba_tvm.h"

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
	int64_t io_shape[4];
} dlr_io_cfg_t;

typedef struct dlr_net_cfg_s {
	char model_fn[FILENAME_LENGTH];
	uint32_t input_num;
	dlr_io_cfg_t input_node[MAX_IO_NUM];
} dlr_net_cfg_t;

typedef struct {
	uint32_t net_num;
	dlr_net_cfg_t net_cfg[MAX_NET_NUM];
	dlr_dev_t dev;

	uint32_t print_time;
}dlr_ctx_t;

#ifndef NO_ARG
#define NO_ARG (0)
#endif

#ifndef HAS_ARG
#define HAS_ARG (1)
#endif

static struct option long_options[] = {
	{"mod-dir",	HAS_ARG, 0, 'b'},
	{"in",	HAS_ARG, 0, 'i'},
	{"ifile",	HAS_ARG, 0, 'f'},
	{"ishape",	HAS_ARG, 0, 's'},

	{"print-time",	NO_ARG, 0, 'e'},

	{"help",	NO_ARG, 0, 'h'},
	{0, 0, 0, 0},
};

static const char *short_options = "b:i:f:s:eh";

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
	{"", "\tinput shape, input dim is always 4"},

	{"", "\tEnable time print. Default is disable."},

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

	printf("Run one model in file mode.\n"
		"\t# %s -b model_folder -i data -f in_img.bin -s 1,3,224,224\n", itself);
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
		memset(p_ctx->net_cfg[i].model_fn, 0, FILENAME_LENGTH);
		memset(p_ctx->net_cfg[i].input_node, 0, MAX_IO_NUM * sizeof(dlr_io_cfg_t));
	}

	p_ctx->dev.type = 1;		// default device is kDLCPU
	p_ctx->dev.id = 0;
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
			strcpy(p_ctx->net_cfg[net_idx].model_fn, optarg);
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
			strcpy(p_ctx->net_cfg[net_idx].input_node[in_idx].io_name, optarg);
			++ p_ctx->net_cfg[net_idx].input_num;
			break;
		case 'f':
			value = strlen(optarg);
			if (value >= FILENAME_LENGTH) {
				printf("Filename [%s] is too long [%d] (>%d).\n", optarg,
					value, FILENAME_LENGTH);
				return -1;
			}
			strcpy(p_ctx->net_cfg[net_idx].input_node[in_idx].io_fn, optarg);
			break;
		case 's':
			{
			int p = 0, d = 0, h = 0, w = 0;
			sscanf(optarg, "%d,%d,%d,%d", &p, &d, &h, &w);
			p_ctx->net_cfg[net_idx].input_node[in_idx].io_shape[0] = p;
			p_ctx->net_cfg[net_idx].input_node[in_idx].io_shape[1] = d;
			p_ctx->net_cfg[net_idx].input_node[in_idx].io_shape[2] = h;
			p_ctx->net_cfg[net_idx].input_node[in_idx].io_shape[3] = w;
			}
			break;
		case 'e':
			p_ctx->print_time = 1;
			break;
		case 'h':
			usage();
			return -1;
		default:
			printf("Error: unknown option found: %c\n", ch);
			return -1;
		}
	}

	return 0;
}

static int dlr_read_binary(const char* filename, float *t, int64_t size)
{
	int rval = 0;

	do {
		std::ifstream data_fin(filename, std::ios::binary);
		data_fin.seekg (0, data_fin.end);

		int file_size = data_fin.tellg();
		if (file_size != size) {
			printf("Error: input file size (%d) should be %ld.\n", file_size, size);
			rval = -1;
			break;
		}
		data_fin.seekg (0, data_fin.beg);
		data_fin.read(reinterpret_cast<char*>(t), size);
	} while(0);

	return rval;
}

static int dlr_process_classification(dlr_ctx_t *p_ctx, const float* out, int num_cls)
{
	auto out_iter = out;
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

static int dlr_prepare_file_mode(dlr_net_cfg_t *p_net, float **in_t, int64_t *size)
{
	int rval = 0;

	do {
		for (uint32_t i = 0; i < p_net->input_num; ++i) {
			if (dlr_read_binary(p_net->input_node[i].io_fn, in_t[i], size[i]) < 0) {
				printf("Error: dlr_read_binary.\n");
				rval = -1;
				break;
			}
		}
		if (rval) break;
	} while(0);

	return rval;
}

static int dlr_dump_outputs(const float **out_t, int64_t* out_size, int num)
{
	int rval = 0;

	for (int i = 0; i < num; ++i) {
		std::string out_fn = "out_" + std::to_string(i) + ".bin";
		std::ofstream out_file(out_fn, std::ios::binary);
		out_file.write(reinterpret_cast<const char*>(out_t[i]), out_size[i]);
		out_file.close();
	}

	return rval;
}

static int dlr_process_outputs(dlr_ctx_t *p_ctx,
	const float **out_t, int64_t* out_size, int num_outputs)
{
	int rval = 0;

	do {
		if (dlr_dump_outputs(out_t, out_size, num_outputs) < 0) {
			printf("Error: dlr_dump_outputs.\n");
			rval = -1;
			break;
		}
		if (num_outputs == 1) {
			dlr_process_classification(p_ctx, out_t[0],
				out_size[0] / sizeof(float));
		}
	} while(0);

	return rval;
}

static void* dlr_execute_one_net(dlr_ctx_t *p_ctx, dlr_net_cfg_t *p_net)
{
	struct timeval tv1, tv2;
	unsigned long tv_diff = 0;

	ConfigAmbaEngineLocation(p_net->model_fn);

	DLContext ctx = {static_cast<DLDeviceType>(p_ctx->dev.type), p_ctx->dev.id};
	std::vector<std::string> paths;
	paths.push_back(std::string(p_net->model_fn));

	dlr::TVMModel mod = dlr::TVMModel(paths, ctx);

	int num_outputs  = mod.GetNumOutputs();
	int num_inputs = p_net->input_num;

	// input/output buffers
	float* in_buf[num_inputs] = {nullptr};
	int64_t in_buf_size[num_inputs] = {0};

	const float* out_buf[num_outputs] = {nullptr};
	int64_t out_buf_size[num_outputs] = {0};
	int out_dim[num_outputs] = {0};

	do {
		if (mod.HasMetadata()) {
			for (int i = 0; i < num_outputs; ++i) {
				const char* out_name = mod.GetOutputName(i);
				printf("model output %d name %s\n", i, out_name);
			}
		}
		for (int i = 0; i < num_outputs; ++i) {
			mod.GetOutputSizeDim(i, &out_buf_size[i], &out_dim[i]);
			out_buf_size[i] *= sizeof(float);
			out_buf[i] = static_cast<const float*>(mod.GetOutputPtr(i));
		}

		for (int i = 0; i < num_inputs; ++i) {
			const char* type = mod.GetInputType(i);
			std::string data_type(type);
			int k = data_type.length() - 1;
			for (k = data_type.length() - 1; k >= 0; --k) {
				if (!isdigit(data_type[k])) {
					break;
				}
			}

			in_buf_size[i] = atoi(data_type.substr(k + 1).c_str()) / 8;
			for (int j = 0; j < 4; ++j) {
				in_buf_size[i] *= p_net->input_node[i].io_shape[j];
			}
			in_buf[i] = (float*)malloc(in_buf_size[i]);
			memset(in_buf[i], 0, in_buf_size[i]);
		}

		if (dlr_prepare_file_mode(p_net, in_buf, in_buf_size) < 0) {
			printf("Error: dlr_prepare_file_mode\n");
			break;
		}

		// set input
		for (int i = 0; i < num_inputs; ++i) {
			mod.SetInput(p_net->input_node[i].io_name, p_net->input_node[i].io_shape,
				in_buf[i], 4);
		}

		// run module
		mod.Run();

		// evaluate execution time at second Run()
		if (p_ctx->print_time) {
			gettimeofday(&tv1, NULL);
			mod.Run();
			gettimeofday(&tv2, NULL);
			tv_diff = (unsigned long) 1000000 * (unsigned long) (tv2.tv_sec - tv1.tv_sec) +
				(unsigned long) (tv2.tv_usec - tv1.tv_usec);
			printf("model  \"%s\" execution time: %lu us\n", p_net->model_fn, tv_diff);
		}

		// don't have to call mod.GetOutput() if using the ptr from inside buffer pool directly
		if (dlr_process_outputs(p_ctx, out_buf, out_buf_size, num_outputs)< 0) {
			printf("Error: dlr_process_outputs.\n");
			break;
		}
	} while(0);

	for (int i = 0; i < num_inputs; ++i) {
		if (in_buf[i]) {
			free(in_buf[i]);
			in_buf[i] = nullptr;
		}
	}

	return nullptr;
}

static int dlr_run_module(dlr_ctx_t *p_ctx)
{
	int rval = 0;

	dlr_net_cfg_t *p_net = &p_ctx->net_cfg[0];
	dlr_execute_one_net(p_ctx, p_net);

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


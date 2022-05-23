/*******************************************************************************
 * test_amba_tvm.cpp
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


// tvm
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <string>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/c_runtime_api.h>

// system
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <pthread.h>

// Amba SDK
#include "amba_tvm.h"

#define FILENAME_LENGTH	(2048)
#define NAME_LENGTH	(32)
#define MAX_NET_NUM	(8)
#define MAX_IO_NUM		(16)

typedef enum {
	TVM_FILE_MODE = 0,
	TVM_REGRESSION_MODE = 2,
	TVM_RUN_MODE_NUM,
} tvm_run_mode_t;

typedef struct {
	int type;
	int id;
}tvm_dev_t;

typedef struct tvm_io_cfg_s {
	char io_name[NAME_LENGTH];
	char io_fn[FILENAME_LENGTH];
} tvm_io_cfg_t;

typedef enum {
	TVM_NET_CLASSIFICATION = 1,
	TVM_NET_OBJECT_DETECT = 2,
	TVM_NET_SEGMENTATION = 3,
	TVM_NET_TYPE_NUM
} tvm_net_type_t;

typedef struct tvm_net_cfg_s {
	char model_fn[FILENAME_LENGTH];
	uint32_t input_num;
	tvm_io_cfg_t input_node[MAX_IO_NUM];

	uint32_t debug_runtime : 1;
	uint32_t net_type: 3;
	uint32_t reserve: 28;
} tvm_net_cfg_t;

typedef struct tvm_socket_s {
	int32_t server_fd;
	int32_t client_fd;
	int32_t socket_port;
	int32_t server_id;

	int32_t total_img_num;
	int32_t cur_img_cnt;
} tvm_socket_t;

typedef struct {
	uint32_t net_num;
	tvm_net_cfg_t net_cfg[MAX_NET_NUM];

	int run_mode;
	tvm_dev_t dev;
	uint32_t show_io;

	tvm_socket_t socket_cfg;
}tvm_ctx_t;

static int run_flag = 1;

#ifndef NO_ARG
#define NO_ARG (0)
#endif

#ifndef HAS_ARG
#define HAS_ARG (1)
#endif

typedef enum {
	TOTAL_IMG_NUM = 0,
	SERVER_ID = 1,
	SOCKET_PORT = 2,
	SHOW_IO = 3,
} tvm_option_t;

static struct option long_options[] = {
	{"cmpl-bin",	HAS_ARG, 0, 'b'},
	{"in",	HAS_ARG, 0, 'i'},
	{"ifile",	HAS_ARG, 0, 'f'},
	{"run-mode",	HAS_ARG, 0, 'r'},

	{"debug-runtime",	NO_ARG, 0, 'e'},
	{"model-type",	HAS_ARG, 0, 'm'},
	{"show-io",	NO_ARG, 0, SHOW_IO},

	{"img-num",	HAS_ARG, 0, TOTAL_IMG_NUM},
	{"server-id",	HAS_ARG, 0, SERVER_ID},
	{"socket-port",	HAS_ARG, 0, SOCKET_PORT},

	{"help",	NO_ARG, 0, 'h'},
	{0, 0, 0, 0},
};

static const char *short_options = "b:i:f:r:em:h";

struct hint_s {
	const char *arg;
	const char *str;
};

static const struct hint_s hint[] = {
	{"", "\tFolder path and basename of compiled files;"
		"Basename of all compiled files should be the same; One folder for each model."},
	{"", "\t\tName of input node. Use multiple -i if there are more than one input nodes."
		"Order of names should be the same as those in compiled.json file."},
	{"", "\tBinary file for network input with float format. Only for file mode and should be preprocessed."},
	{"", "\tRun mode; 0 file mode; 2 regression test mode."},

	{"", "\tEnable debug runtime in TVM. Default is disable."},
	{"", "\tModel type; 1 classification 2 object detection 3 segmentation."},
	{"", "\tShow primary i/o info of TVM compiled artifacts."},

	{"", "\tTotal number of test images for regression test."},
	{"", "\tServer id when multiple EVK are used for regression test."},
	{"", "\tSocket port in regression test mode."},

	{"", "\tprint help info"},
};

static void usage(void)
{
	const char *itself = "test_amba_tvm";
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
		"\t# %s -b compiled -i input_name --show-io\n", itself);
	printf("2. Run one classification model in file mode.\n"
		"\t# %s -b compiled -i data -f in_img.bin -m 1\n", itself);
}

static int init_param(int argc, char **argv, tvm_ctx_t *p_ctx)
{
	int ch = 0, value = 0, net_idx = 0, net_num = 0, in_idx = 0;
	int option_index = 0;
	opterr = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "-cmpl-bin")) {
			++p_ctx->net_num;
		}
	}
	if (p_ctx->net_num > MAX_NET_NUM) {
		printf("Error: only support %d net modes at most.\n", MAX_NET_NUM);
		return -1;
	}
	for (uint32_t i = 0; i < p_ctx->net_num; ++ i) {
		memset(p_ctx->net_cfg[i].model_fn, 0, FILENAME_LENGTH);
		memset(p_ctx->net_cfg[i].input_node, 0, MAX_IO_NUM * sizeof(tvm_io_cfg_t));
	}

	p_ctx->run_mode = TVM_FILE_MODE;		// default is file mode
	p_ctx->dev.type = kDLAmba;	// default device type is Amba Device
	p_ctx->dev.id = 0;			// default device id is 0

	p_ctx->socket_cfg.socket_port = 27182;
	p_ctx->socket_cfg.cur_img_cnt = 1;
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
			snprintf(p_ctx->net_cfg[net_idx].model_fn, sizeof(p_ctx->net_cfg[net_idx].model_fn), "%s", optarg);
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
		case 'r':
			p_ctx->run_mode = atoi(optarg);
			break;
		case 'e':
			p_ctx->net_cfg[net_idx].debug_runtime = 1;
			break;
		case 'm':
			p_ctx->net_cfg[net_idx].net_type = atoi(optarg);
			break;
		case SHOW_IO:
			p_ctx->show_io = 1;
			break;
		case SOCKET_PORT:
			p_ctx->socket_cfg.socket_port = atoi(optarg);
			break;
		case SERVER_ID:
			p_ctx->socket_cfg.server_id = atoi(optarg);
			break;
		case TOTAL_IMG_NUM:
			p_ctx->socket_cfg.total_img_num = atoi(optarg);
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
	if (!p_ctx->show_io && p_ctx->run_mode == TVM_FILE_MODE) {
		for (uint32_t i = 0; i < p_ctx->net_num; ++i) {
			if (p_ctx->net_cfg[i].net_type < TVM_NET_CLASSIFICATION ||
				p_ctx->net_cfg[i].net_type > TVM_NET_SEGMENTATION) {
				printf("Error: invalid network type %d for network index %d.\n", \
					p_ctx->net_cfg[i].net_type, i);
				return -1;
			}
		}
	}

	return 0;
}

static int tvm_get_DLTensor_datatype(DLTensor *t, std::string &type)
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
		tvm_get_DLTensor_datatype(t[i], datatype);
		info += " datatype: " + datatype;
		printf("%s\n", info.c_str());
	}
}

static int tvm_get_DLTensor_size(DLTensor *t)
{
	int size = 1;
	for(int i = 0; i < t->ndim; ++i) {
		size *= t->shape[i];
	}
	size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;

	return size;
}

static int tvm_init_socket(tvm_socket_t *p_socket)
{
	int rval = 0;
	struct sockaddr_in server;

	do {
		printf("Init socket io.\n");
		p_socket->server_fd = socket(AF_INET, SOCK_STREAM, 0);
		if (p_socket->server_fd < 0) {
			printf("ERROR: Unable to create parent socket!\n");
			rval = -1; break;
		}

		server.sin_family = AF_INET;
		server.sin_addr.s_addr = INADDR_ANY;
		server.sin_port = htons(p_socket->socket_port);
		printf("test_amba_tvm open port %d.\n", p_socket->socket_port);

		rval = bind(p_socket->server_fd, (struct sockaddr *)&server, sizeof(server));
		if (rval < 0) {
			printf("ERROR: Unable to bind server socket!\n");
			close(p_socket->server_fd);
			rval = -1; break;
		}
		printf("Bind socket success.\n");
		printf("Listening... \n");

		rval = listen(p_socket->server_fd, 1);
		if (rval < 0) {
			printf("ERROR: Unable to listen at server socket!\n");
			close(p_socket->server_fd);
			rval = -1; break;
		}

		p_socket->client_fd = accept(p_socket->server_fd, (struct sockaddr *)0, (socklen_t *)0);
		if (p_socket->client_fd < 0) {
			printf("ERROR: Unable to accept client!\n");
			rval = -1; break;
		}
		printf("Accept socket success.\n");
	} while(0);

	return rval;
}

static int tvm_deinit_socket(tvm_socket_t *p_socket)
{
	int rval = 0;

	if (p_socket->client_fd) {
		close(p_socket->client_fd);
	}
	if (p_socket->server_fd) {
		close(p_socket->server_fd);
	}

	return rval;
}

static int tvm_read_socket(int socket_fd, void *dest, int size)
{
	int rval = 0;
	int recv_size = 0;

	do {
		rval = recv(socket_fd, (uint8_t *)dest + recv_size, size - recv_size, 0);
		if (rval > 0) {
			recv_size += rval;
		} else {
			printf("Engine_error: Unable to receive data!\n");
			rval = -1;
			break;
		}
	} while (recv_size < size);

	return rval;
}

static ssize_t tvm_write_socket(int socket_fd, const void *buf, size_t len)
{
	size_t nleft;
	ssize_t nwritten;
	const char *ptr = NULL;

	if (socket_fd < 0 || buf == NULL || len <= 0) {
		return -1; /* error */
	}

	ptr = (char *)buf;
	nleft = len;

	while (nleft > 0) {
		nwritten = write(socket_fd, ptr, nleft);
		if (nwritten <= 0) {
			if (errno == EINTR)
				nwritten = 0; /* interupted, call write again */
			else
				return -1; /* error */
		}
		nleft -= nwritten;
		ptr += nwritten;
	}

	return (len - nleft);
}

static int tvm_proc_socket_input(tvm_socket_t *p_socket, tvm_net_cfg_t *p_net, DLTensor **in_t)
{
	int rval = 0;
	int32_t socket_fd = p_socket->client_fd;
	int32_t total_img_num = -1, cur_img_cnt = -1;
	uint32_t input_num = 0;
	uint32_t i = 0;
	char io_name[NAME_LENGTH];

	do {
		rval = tvm_read_socket(socket_fd, &total_img_num, sizeof(p_socket->total_img_num));
		if (rval < 0 || total_img_num != p_socket->total_img_num) {
			printf("Error: failed to get correct total test image number: receive %d, should be %d.\n", \
				total_img_num, p_socket->total_img_num);
			rval = -1; break;
		}

		rval = tvm_read_socket(socket_fd, &cur_img_cnt, sizeof(p_socket->cur_img_cnt));
		if (rval < 0 || cur_img_cnt != p_socket->cur_img_cnt) {
			printf("Error: failed to get correct current image count: receive %d, should be %d.\n", \
				cur_img_cnt, p_socket->cur_img_cnt);
			rval = -1; break;
		}

		rval = tvm_read_socket(socket_fd, &input_num, sizeof(p_net->input_num));
		if (rval < 0 || input_num != p_net->input_num) {
			printf("Error: failed to get correct input num: receive %d, should be %d.\n", \
				input_num, p_net->input_num);
			rval = -1; break;
		}

		/* loop for multiple inputs */
		for (i = 0; i < input_num; ++i) {
			int32_t in_size = tvm_get_DLTensor_size(in_t[i]);
			int32_t file_size = -1;
			rval = tvm_read_socket(socket_fd, &file_size, sizeof(file_size));
			if (rval < 0 || file_size != in_size) {
				printf("Error: failed to get correct input file size: receive %d, should be %d.\n", \
					file_size, in_size);
				rval = -1; break;
			}

			rval = tvm_read_socket(socket_fd, in_t[i]->data, file_size);
			if (rval < 0) {
				printf("Error: failed to get input buffer of io name %s.\n", io_name);
				rval = -1; break;
			}
		}
		if (rval) break;
	}while(0);

	return rval;
}

static int tvm_proc_socket_output(tvm_socket_t *p_socket, tvm_net_cfg_t *p_net,
	DLTensor **out_t, int num_outputs)
{
	int rval = 0;
	int32_t socket_fd = p_socket->client_fd;
	int32_t total_img_num = p_socket->total_img_num;
	int32_t cur_img_cnt = p_socket->cur_img_cnt;
	int32_t output_num = num_outputs;
	int32_t file_size = 0;

	do {
		if (tvm_write_socket(socket_fd, &total_img_num, sizeof(total_img_num)) != sizeof(total_img_num)) {
			printf("Error: failed to send total image number.\n");
			rval = -1;
			break;
		}
		if (tvm_write_socket(socket_fd, &cur_img_cnt, sizeof(cur_img_cnt)) != sizeof(cur_img_cnt)) {
			printf("Error: failed to send current image count.\n");
			rval = -1;
			break;
		}
		if (tvm_write_socket(socket_fd, &output_num, sizeof(output_num)) != sizeof(output_num)) {
			printf("Error: failed to send output number.\n");
			rval = -1;
			break;
		}
		for (int32_t i = 0; i < output_num; ++i) {
			if (tvm_write_socket(socket_fd, &i, sizeof(i)) != sizeof(i)) {
				printf("Error: failed to send outptu index.\n");
				rval = -1;
				break;
			}
			file_size = tvm_get_DLTensor_size(out_t[i]);
			if (tvm_write_socket(socket_fd, &file_size, sizeof(file_size)) != sizeof(file_size)) {
				printf("Error: failed to send file size.\n");
				rval = -1;
				break;
			}
			if (tvm_write_socket(socket_fd, out_t[i]->data, file_size) != file_size) {
				printf("Error: failed to send output buffer of index %d.\n", i);
				rval = -1;
				break;
			}
		}
		if (rval) break;

		++ p_socket->cur_img_cnt;
		if (p_socket->cur_img_cnt > p_socket->total_img_num) {
			run_flag = 0;
		}
	} while(0);

	return rval;
}

static std::string get_file_dirname(const char *file_path)
{
	std::string file(file_path), dir_path(".");
	size_t pos = file.rfind("/");

	if (pos != std::string::npos) {
		dir_path = file.substr(0, pos);
	}
	return dir_path;
}

static int tvm_read_binary(const char* filename, DLTensor *t)
{
	int rval = 0;
	int in_size = tvm_get_DLTensor_size(t);
	std::ifstream data_fin(filename, std::ios::binary);

	do {
		data_fin.seekg (0, data_fin.end);

		int file_size = data_fin.tellg();
		if (file_size != in_size) {
			printf("Error: input file size (%d) should be %d.\n", file_size, in_size);
			rval = -1;
			break;
		}
		data_fin.seekg (0, data_fin.beg);
		data_fin.read(static_cast<char*>(t->data), in_size);
	} while(0);
	data_fin.close();

	return rval;
}

static int tvm_process_classification(tvm_ctx_t *p_ctx, DLTensor* out, int num_cls)
{
	auto out_iter = static_cast<float*>(out->data);
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

	if (p_ctx->run_mode == TVM_FILE_MODE) {
		printf("Top 5 categories: %d, %d, %d, %d, %d\n", prob_id[0], prob_id[1], prob_id[2],
			prob_id[3], prob_id[4]);
		printf("Top 5 scores: %.4f, %.4f, %.4f, %.4f, %.4f\n", prob_softmax[0], prob_softmax[1],
			prob_softmax[2], prob_softmax[3], prob_softmax[4]);
	}

	return 0;
}

int tvm_load_module(tvm_net_cfg_t *p_net, tvm_dev_t *p_dev,
	tvm::runtime::Module &mod)
{
	int rval = 0;

	std::string model_fn = std::string(p_net->model_fn) + ".so";
	// due to dlopen design, *.so library filename should contain a slash "/" at least
	if (model_fn.find("/") == std::string::npos) {
		model_fn = "./" + model_fn;
	}
	// tvm module for compiled functions
	tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(model_fn);

	// json graph
	model_fn = std::string(p_net->model_fn) + ".json";
	std::ifstream json_in(model_fn);
	std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
	json_in.close();

	// parameters in binary
	model_fn = std::string(p_net->model_fn) + ".params";
	std::ifstream params_in(model_fn);
	std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
	params_in.close();

	// parameters need to be TVMByteArray type to indicate the binary data
	TVMByteArray params_arr;
	params_arr.data = params_data.c_str();
	params_arr.size = params_data.length();

	// get global function for runtime
	const tvm::runtime::PackedFunc* pfr = nullptr;
	do {
		if (!p_net->debug_runtime) {
			pfr = tvm::runtime::Registry::Get("tvm.graph_executor.create");
			if (!pfr) {
				printf("Error: TVM graph executor is not enabled.\n");
				rval = -1;
				break;
			}
		} else {
			pfr = tvm::runtime::Registry::Get("tvm.graph_executor_debug.create");
			if (!pfr) {
				printf("Error: TVM debug graph executor is not enabled.\n");
				rval = -1;
				break;
			}
		}

		mod = (*pfr)(json_data, mod_syslib, p_dev->type, p_dev->id);
		// load patameters
		tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
		load_params(params_arr);
	} while(0);

	return rval;
}

static int tvm_prepare_file_mode(tvm_net_cfg_t *p_net, DLTensor **in_t)
{
	int rval = 0;

	do {
		for (uint32_t i = 0; i < p_net->input_num; ++i) {
			if (tvm_read_binary(p_net->input_node[i].io_fn, in_t[i]) < 0) {
				printf("Error: tvm_read_binary.\n");
				rval = -1;
				break;
			}
		}
		if (rval) break;
	} while(0);

	return rval;
}

static int tvm_dump_outputs(DLTensor **out_t, int num)
{
	int rval = 0;

	for (int i = 0; i < num; ++i) {
		int out_size = tvm_get_DLTensor_size(out_t[i]);
		std::string out_fn = "out_" + std::to_string(i) + ".bin";
		std::ofstream out_file(out_fn, std::ios::binary);
		out_file.write(static_cast<char*>(out_t[i]->data), out_size);
		out_file.close();
	}

	return rval;
}

static int tvm_process_outputs(tvm_ctx_t *p_ctx,tvm_net_cfg_t *p_net,
	DLTensor **out_t, int num_outputs)
{
	int rval = 0;

	do {
		if (p_ctx->run_mode == TVM_FILE_MODE) {
			if (tvm_dump_outputs(out_t, num_outputs) < 0) {
				printf("Error: tvm_dump_outputs.\n");
				rval = -1;
				break;
			}
		} else if (p_ctx->run_mode == TVM_REGRESSION_MODE) {
			if (tvm_proc_socket_output(&p_ctx->socket_cfg, p_net, out_t, num_outputs) < 0) {
				printf("Error: tvm_proc_socket_output\n");
				rval = -1;
				break;
			}
		}

		if (num_outputs == 1 && p_net->net_type == TVM_NET_CLASSIFICATION) {
			// imagenet pretrained classification network from gluoncv
			tvm_process_classification(p_ctx, out_t[0],
				out_t[0]->shape[out_t[0]->ndim - 1]);
		}
	} while(0);

	return rval;
}

static float tvm_debug_individule_sum(std::string str)
{
	float sum = 0.0;
	size_t pos = str.find(",");

	if (pos != std::string::npos) {
		std::string num_str = str.substr(0, pos);
		std::string remain_str = str.substr(pos + 1);
		sum = atof(num_str.c_str()) * 1000000.0f + tvm_debug_individule_sum(remain_str);
	}

	return sum;
}

typedef struct thread_arg_s {
	void *ctx;
	void *net;
} thread_arg_t;

static void* tvm_execute_one_net(void *args)
{
	int rval = 0;

	tvm_ctx_t *p_ctx = (tvm_ctx_t*)(((thread_arg_t*)args)->ctx);
	tvm_net_cfg_t *p_net = (tvm_net_cfg_t*)(((thread_arg_t*)args)->net);
	tvm::runtime::Module mod;

	std::string mod_dir = get_file_dirname(p_net->model_fn);
	if (ConfigAmbaEngineLocation(mod_dir.c_str())) {
		printf("Error: ConfigAmbaEngineLocation\n");
		return nullptr;
	}

	if (tvm_load_module(p_net, &p_ctx->dev, mod) < 0) {
		printf("Error: tvm_load_module\n");
		return NULL;
	}

	// get global function for runtime
	tvm::runtime::PackedFunc get_num_outputs = mod.GetFunction("get_num_outputs");
	tvm::runtime::PackedFunc get_input = mod.GetFunction("get_input");
	tvm::runtime::PackedFunc set_input_zc = mod.GetFunction("set_input_zero_copy");
	tvm::runtime::PackedFunc run = mod.GetFunction("run");
	tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");

	int num_outputs  = get_num_outputs();
	int num_inputs = p_net->input_num;

	// get input/output tensors
	DLTensor *in_t[num_inputs] = {nullptr};
	DLTensor *out_t[num_outputs] = {nullptr};

	do {
		for (int i = 0; i < num_inputs; ++i) {
			DLTensor *ptr_idx = get_input(i).operator DLTensor*();
			DLTensor *ptr_name = get_input(p_net->input_node[i].io_name).operator DLTensor*();
			if (ptr_idx != ptr_name) {
				printf("Error: input node names are wrong or disordered.\n");
				rval = -1;
				break;
			}
			in_t[i] = ptr_idx;
		}
		for (int i = 0; i < num_outputs; ++i) {
			out_t[i] = get_output(i).operator DLTensor*();
		}
		if (p_ctx->show_io) {
			show_DLTensor_io(in_t, num_inputs, "input");
			show_DLTensor_io(out_t, num_outputs, "output");
			rval = 1;
			break;
		}

		if (p_ctx->run_mode == TVM_FILE_MODE) {
			if (tvm_prepare_file_mode(p_net, in_t) < 0) {
				printf("Error: tvm_prepare_file_mode\n");
				rval = -1;
				break;
			}
		}
	} while(0);

	do {
		if (rval) break;
		if (p_ctx->run_mode == TVM_REGRESSION_MODE) {
			if (tvm_proc_socket_input(&p_ctx->socket_cfg, p_net, in_t) < 0) {
				printf("Error: tvm_proc_socket_input\n");
				rval = -1;
				break;
			}
		}

		// run module
		run();

		if (tvm_process_outputs(p_ctx, p_net, out_t, num_outputs)< 0) {
			printf("Error: tvm_process_outputs.\n");
			rval = -1;
			break;
		}
	} while(p_ctx->run_mode && run_flag);

	if (rval == 0) {
		if (p_net->debug_runtime && p_ctx->run_mode == TVM_FILE_MODE) {
			tvm::runtime::PackedFunc run_individual = mod.GetFunction("run_individual");
			std::string debug_rt_str = run_individual(10,1,100).operator std::string();

			float op_time = tvm_debug_individule_sum(debug_rt_str);
			printf("[%s] run time: %.1f us\n", p_net->model_fn, op_time);
		}
	}

	return NULL;
}

static int tvm_run_module(tvm_ctx_t *p_ctx)
{
	int rval = 0;
	uint32_t net_num = p_ctx->net_num;
	int launched_net = 0;

	pthread_t tid[net_num];
	int tret[net_num];
	thread_arg_t thread_arg[net_num];

	for (uint32_t i = 0; i < net_num; i++) {
		tret[i] = -1;
		tvm_net_cfg_t *p_net = &p_ctx->net_cfg[i];
		thread_arg[i].ctx = p_ctx;
		thread_arg[i].net = p_net;
		if ((tret[i] = pthread_create(&tid[i], NULL, tvm_execute_one_net, &thread_arg[i])) < 0) {
			printf("Error: launch network \"%s\".\n", p_net->model_fn);
			break;
		} else {
			printf("Succeed to launch network \"%s\".\n", p_net->model_fn);
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
	run_flag = 0;
	printf("sigstop msg, exit test_amba_tvm.\n");
}

int main(int argc, char *argv[])
{
	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);

	int rval = 0;
	tvm_ctx_t tvm_ctx;
	tvm_ctx_t *p_ctx = &tvm_ctx;
	int socket_inited = 0;

	memset(p_ctx, 0, sizeof(tvm_ctx));

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
		if (p_ctx->run_mode == TVM_REGRESSION_MODE) {
			if (tvm_init_socket(&p_ctx->socket_cfg) < 0) {
				printf("Error: tvm_init_socket\n");
				rval = -1;
				break;
			}
			socket_inited = 1;
		}

		if (tvm_run_module(p_ctx) < 0) {
			printf("Error: tvm_run_module.\n");
			rval = -1;
			break;
		}
	} while(0);

	if (p_ctx->run_mode == TVM_REGRESSION_MODE) {
		if (socket_inited) {
			tvm_deinit_socket(&p_ctx->socket_cfg);
		}
		socket_inited = 0;
	}

	return rval;
}


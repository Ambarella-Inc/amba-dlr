/*******************************************************************************
 * test_amba_tvm_live.cpp
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
#include <math.h>

// Amba SDK
#include "iav_ioctl.h"
#include "lib_smartfb.h"
#include "cavalry_mem.h"
#include "vproc.h"
#include "amba_tvm.h"

#define TVM_VPROC_BIN	"/usr/local/vproc/vproc.bin"
#define FILENAME_LENGTH	(2048)
#define NAME_LENGTH	(32)
#define MAX_IO_NUM		(16)

#ifndef ALIGN_32_BYTE
#define ALIGN_32_BYTE(x) ((((x) + 31) >> 5) << 5)
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

typedef struct tvm_fb_s {
	smartfb_init_t type;
	smartfb_box_t box;
	smartfb_textbox_t textbox;
} tvm_fb_t;

typedef enum {
	TVM_QUERY_BUF_CANVAS = 0,
	TVM_QUERY_BUF_PYRAMID,
	TVM_QUERY_BUF_TYPE_NUM,
} tvm_query_type_t;

typedef struct tvm_iav_s {
	int fd_iav;
	uint8_t *dsp_mem;
	uint32_t dsp_phy_addr;
	uint32_t dsp_size;

	struct iav_yuv_cap data_cap;

	uint32_t query_buf_type;
	uint32_t query_buf_id;
} tvm_iav_t;

struct cv_mem {
	void *virt;
	unsigned long phys;
	unsigned long size;
};

typedef struct tvm_vproc_s {
	int fd_cav;
	uint32_t need_flat: 1;
	uint32_t reserve:31;

	struct cv_mem lib_mem;
	struct cv_mem deform_mem;
	struct cv_mem mean_mem;
	struct cv_mem submean_mem;
	struct cv_mem scale_mem;
	struct cv_mem imcvt_mem;
	struct cv_mem flat_mem;

	vect_desc_t deform_in;
	vect_desc_t deform_out;
	deformation_extra_t dext;

	vect_desc_t submean_in;
	vect_desc_t submean_out;
	vect_desc_t mean;

	vect_desc_t scale_in;
	vect_desc_t scale_out;

	vect_desc_t imcvt_in;
	vect_desc_t imcvt_out;

	vect_desc_t flat_in;
	vect_desc_t flat_out;
} tvm_vproc_t;

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

	uint32_t net_type: 3;
	uint32_t reserve: 29;
} tvm_net_cfg_t;

typedef struct {
	tvm_net_cfg_t net_cfg;

	int is_bgr;
	int mean_rgb[3];
	float scale;
	int  is_nhwc;

	float conf_th;
	char dataset[32];
	char framework[32];

	tvm_vproc_t vproc;

	tvm_dev_t dev;
	tvm_iav_t iav;
	tvm_fb_t fb;
}tvm_ctx_t;

static int run_flag = 1;
static const char *voc07_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle",
	"bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

static char const *coco_names[] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
	"train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
	"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
	"giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
	"snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
	"surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
	"pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
	"toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
	"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
	"hair drier", "toothbrush"};

static const int coco_label_id[] = {
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
	18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
	37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
	54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
	74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90
};

#ifndef NO_ARG
#define NO_ARG (0)
#endif

#ifndef HAS_ARG
#define HAS_ARG (1)
#endif

typedef enum {
	QUERY_BUF_TYPE = 0,
	QUERY_BUF_ID = 1,
	PREPROC_SUBMEAN = 2,
	PREPROC_SCALE = 3,
	COLOR_BGR = 4,
	CONF_TH = 5,
	TF_NHWC = 6,
} tvm_option_t;

static struct option long_options[] = {
	{"cmpl-bin",	HAS_ARG, 0, 'b'},
	{"in",	HAS_ARG, 0, 'i'},

	{"mean",	HAS_ARG, 0, PREPROC_SUBMEAN},
	{"scale",	HAS_ARG, 0, PREPROC_SCALE},
	{"bgr",	HAS_ARG, 0, COLOR_BGR},
	{"nhwc",	NO_ARG, 0, TF_NHWC},

	{"buf-type",	HAS_ARG, 0, QUERY_BUF_TYPE},
	{"buf-id",	HAS_ARG, 0, QUERY_BUF_ID},

	{"model-type",	HAS_ARG, 0, 'm'},
	{"conf-th",	HAS_ARG, 0, CONF_TH},
	{"dataset",	HAS_ARG, 0, 'd'},
	{"framework",	HAS_ARG, 0, 'w'},

	{"help",	NO_ARG, 0, 'h'},
	{0, 0, 0, 0},
};

static const char *short_options = "b:i:m:d:w:h";

struct hint_s {
	const char *arg;
	const char *str;
};

static const struct hint_s hint[] = {
	{"", "\tFolder path and basename of compiled files;"
		"Basename of all compiled files should be the same; One folder for each model."},
	{"", "\t\tName of input node. Use multiple -i if there are more than one input nodes."
		"Order of names should be the same as those in compiled.json file."},

	{"", "\tSubmean value for preproc, 3 integer value for B/G/R channels."},
	{"", "\tScale value for preproc, one float value."},
	{"", "\tColor format, 0 for RGB 1 for BGR."},
	{"", "\tNHWC order."},

	{"", "\tDSP query type, 0 for canvas 1 for pyramid."},
	{"", "\tDSP query buf id (canvas id or pyramid id)."},

	{"", "\tModel type; 1 classification 2 object detection 3 segmentation."},
	{"", "\tConfidence threshold for bboxes nms;"},
	{"", "\tDataset, could be VOC07, COCO, default is VOC07;"},
	{"", "\tNative model framework, could be mxnet, tflite, default is mxnet;"},

	{"", "\tprint help info"},
};

static void usage(void)
{
	const char *itself = "test_amba_tvm_live";
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

	printf("1. Run with MXNET Resnet model in live mode.\n"
		"\t# %s -b compiled -i data --buf-type 1 --buf-id 3 "
		"--bgr 0 --scale 0.017 --mean 123,116,103 -m 1 --conf-th 0.35\n", itself);

	printf("2. Run with MXNET Resnet-SSD model in live mode.\n"
		"\t# %s -b compiled -i data --buf-type 1 --buf-id 1 "
		"--bgr 0 --scale 0.017 --mean 123,116,103 -m 2 --conf-th 0.35\n", itself);

	printf("3. Run with TFLITE Mobilenet-SSD model in live mode.\n"
		"\t# %s -b compiled -i normalized_input_image_tensor "
		"--buf-type 1 --buf-id 1 --bgr 0 --scale 0.00784 --mean 127,127,127 "
		"-m 2 --conf-th 0.35 -d COCO -w tflite --nhwc\n", itself);
}

static int get_multi_int_args(char *optarg, int *argarr, int argcnt)
{
	int i;
	const char *delim = ", \n\t";
	char *ptr;

	ptr = strtok(optarg, delim);
	argarr[0] = atoi(ptr);

	for (i = 1; i < argcnt; ++i) {
		ptr = strtok(NULL, delim);
		if (ptr == NULL) {
			break;
		}
		argarr[i] = atoi(ptr);
	}
	if (i < argcnt) {
		printf("It's expected to have [%d] params, only get [%d].\n",
			argcnt, i);
		return -1;
	}
	return 0;
}

static int init_param(int argc, char **argv, tvm_ctx_t *p_ctx)
{
	int ch = 0, value = 0, in_idx = 0;
	int option_index = 0;
	opterr = 0;

	memset(p_ctx->net_cfg.model_fn, 0, FILENAME_LENGTH);
	memset(p_ctx->net_cfg.input_node, 0, MAX_IO_NUM * sizeof(tvm_io_cfg_t));

	p_ctx->dev.type = kDLAmba;	// default device type is Amba Device
	p_ctx->dev.id = 255;			// default device id is 255
	p_ctx->iav.query_buf_type = 0;	// canvas
	p_ctx->iav.query_buf_id = 0; 	// canvas id 0
	p_ctx->mean_rgb[0] = p_ctx->mean_rgb[1] = p_ctx->mean_rgb[2] = 0;
	p_ctx->scale = 1.0;
	p_ctx->is_bgr = 1;
	p_ctx->conf_th = 0.05;
	strcpy(p_ctx->dataset, "VOC07");
	strcpy(p_ctx->framework, "mxnet");

	while ((ch = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
		switch (ch) {
		case 'b':
			value = strlen(optarg);
			if (value >= FILENAME_LENGTH) {
				printf("Error: Filename [%s] is too long [%u] (>%u).\n", optarg,
					value, FILENAME_LENGTH);
				return -1;
			}
			snprintf(p_ctx->net_cfg.model_fn, sizeof(p_ctx->net_cfg.model_fn), "%s", optarg);
			break;
		case 'i':
			value = strlen(optarg);
			if (value >= NAME_LENGTH) {
				printf("Error: Filename [%s] is too long [%u] (>%u).\n", optarg,
					value, NAME_LENGTH);
				return -1;
			}
			in_idx = p_ctx->net_cfg.input_num;
			if (in_idx >= MAX_IO_NUM) {
				printf("IO pair number is too much: %u > %u.\n", in_idx, MAX_IO_NUM);
				return -1;
			}
			snprintf(p_ctx->net_cfg.input_node[in_idx].io_name,
				sizeof(p_ctx->net_cfg.input_node[in_idx].io_name), "%s", optarg);
			++ p_ctx->net_cfg.input_num;
			break;
		case PREPROC_SUBMEAN:
			if (get_multi_int_args(optarg, p_ctx->mean_rgb, 3)) {
				printf("Error: get_multi_int_args mean.\n");
				return -1;
			}
			break;
		case PREPROC_SCALE:
			p_ctx->scale = atof(optarg);
			break;
		case COLOR_BGR:
			p_ctx->is_bgr = atoi(optarg);
			break;
		case TF_NHWC:
			p_ctx->is_nhwc = 1;
			break;
		case QUERY_BUF_TYPE:
			value = atoi(optarg);
			if (value != 0 && value != 1) {
				printf("Error: query buf type can only be 0 or 1.\n");
				return -1;
			}
			p_ctx->iav.query_buf_type = value;
			break;
		case QUERY_BUF_ID:
			p_ctx->iav.query_buf_id = atoi(optarg);
			break;
		case 'm':
			p_ctx->net_cfg.net_type = atoi(optarg);
			break;
		case CONF_TH:
			p_ctx->conf_th = atof(optarg);
			break;
		case 'd':
			value = strlen(optarg);
			if (value >= 32) {
				printf("Error: dataset name [%s] is too long [%u] (>%u).\n", optarg,
					value, 32);
				return -1;
			}
			strcpy(p_ctx->dataset, optarg);
			break;
		case 'w':
			value = strlen(optarg);
			if (value >= 32) {
				printf("Error: dataset name [%s] is too long [%u] (>%u).\n", optarg,
					value, 32);
				return -1;
			}
			strcpy(p_ctx->framework, optarg);
			break;
		case 'h':
			usage();
			return -1;
		default:
			printf("Error: unknown option found: %c\n", ch);
			return -1;
		}
	}

	if (strlen(p_ctx->net_cfg.model_fn) == 0) {
		printf("Error: please select model by -b\n");
		return -1;
	}
	if (p_ctx->net_cfg.input_num == 0) {
		printf("Error: pleaes provide input name by -i\n");
		return -1;
	}
	if (p_ctx->net_cfg.net_type < TVM_NET_CLASSIFICATION ||
		p_ctx->net_cfg.net_type > TVM_NET_SEGMENTATION) {
		printf("Error: invalid network type %d\n", p_ctx->net_cfg.net_type);
		return -1;
	}

	return 0;
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

static int init_framebuffer_for_hdmi(tvm_fb_t *p_fb)
{
	smartfb_box_t *fb_box  = &p_fb->box;
	smartfb_textbox_t *fb_textbox = &p_fb->textbox;

	smartfb_obj_t *fb_obj[2];
	smartfb_box_t *textbox;

	memset(fb_box, 0, sizeof(smartfb_box_t));
	memset(fb_textbox, 0, sizeof(smartfb_textbox_t));

	fb_obj[0] = &fb_box->obj;
	textbox = &fb_textbox->box;
	fb_obj[1] = &textbox->obj;

	fb_obj[0]->color = SMARTFB_COLOR_RED;
	fb_box->line_thickness = 2;

	fb_obj[1]->color = fb_obj[0]->color;
	fb_textbox->font_size = 20;
	textbox->line_thickness = 0;
	fb_textbox->wrap_line = 0;
	fb_textbox->bold = 0;
	fb_textbox->italic = 0;

	return 0;
}

static int tvm_init_fb(tvm_fb_t *p_fb)
{
	int rval = 0;

	do {
		p_fb->type.vout = SMARTFB_ANALOG_VOUT;
		if (smartfb_init(&p_fb->type) < 0) {
			printf("smartfb_init err\n");
			rval = -1;
			break;
		}
		init_framebuffer_for_hdmi(p_fb);
	} while(0);

	return rval;
}

static int tvm_deinit_fb()
{
	smartfb_deinit();
	return 0;
}

static int fetch_label_id(tvm_ctx_t *p_ctx, int id)
{
	int label_id = 0;
	if (strcmp(p_ctx->dataset, "VOC07") == 0) {
		label_id = id + 1;
	} else if (strcmp(p_ctx->dataset, "COCO") == 0) {
		for (int i = 0; i < 80; ++i) {
			if (coco_label_id[i] == (id + 1)) {
				label_id = i;
				break;
			}
		}
	}

	return label_id;
}

static int tvm_draw_framebuffer_ssd(tvm_ctx_t *p_ctx, DLTensor** out, DLTensor *in)
{
	int i = 0, tmp = 0;
	tvm_fb_t * p_fb = &p_ctx->fb;
	smartfb_box_t *fb_box = &p_fb->box;
	smartfb_textbox_t *fb_textbox = &p_fb->textbox;
	smartfb_obj_t *fb_obj[2];
	smartfb_box_t *textbox;
	struct fb_var_screeninfo var;
	uint32_t xres = 0, yres = 0;
	uint16_t start_x = 0, start_y = 0, end_x = 0, end_y = 0;
	float *p_id = NULL;
	float *p_score = NULL;
	float *p_bbox = NULL;
	const char **dataset = NULL;
	int num_bbox = 0;
	int bbox_order = 0;	// 0: x1,y1,x2,y2; 1: y1,x1,y2,x2
	int class_id = -1;

	if (strcmp(p_ctx->framework, "mxnet") == 0) {
		/* voc07 dataset pretrained ssd network from gluoncv that has 3 outputs:
		* out[0]: id, [1,100,1]
		* out[1]: score, [1,100,1]
		* out[2]: bbox, [1,100,4] */
		p_id = static_cast<float*>(out[0]->data);
		p_score = static_cast<float*>(out[1]->data);
		p_bbox = static_cast<float*>(out[2]->data);
		num_bbox = 100;
		bbox_order = 0;
		dataset = voc07_names;
	} else if (strcmp(p_ctx->framework, "tflite") == 0) {
		/* coco dataset pretrained ssd network from tflite that has 4 outputs:
		* out[0]: bbox, [1,10,4], fp32
		* out[1]: id, [1,1,10], fp32
		* out[2]: score, [1,1,10], fp32
		* out[3]: bbox num, [1], int32 */
		p_id = static_cast<float*>(out[1]->data);
		p_score = static_cast<float*>(out[2]->data);
		p_bbox = static_cast<float*>(out[0]->data);
		num_bbox = *(int*)(out[3]->data);
		bbox_order = 1;
		dataset = coco_names;
	} else {
		printf("Error: invalid framework, only mxnet and tflite framework is supported.\n");
		return -1;
	}

	fb_obj[0] = &fb_box->obj;
	textbox = &fb_textbox->box;
	fb_obj[1] = &textbox->obj;

	smartfb_get_var(&var);
	xres = var.xres;
	yres = var.yres;

	if (num_bbox > 0) {
		smartfb_clear_buffer();

		for (i = 0; i < num_bbox; i++) {
			// hard-code score threshold
			if (p_score[i] < p_ctx->conf_th) {
				continue;
			}

			if (bbox_order == 0) {
				// mxnet
				start_x = (u16)(p_bbox[4 * i + 0] * xres / in->shape[3]);
				start_y = (u16)(p_bbox[4 * i + 1] * yres / in->shape[2]);
				end_x = (u16)(p_bbox[4 * i + 2] * xres / in->shape[3]);
				end_y = (u16)(p_bbox[4 * i + 3] * yres / in->shape[2]);
			} else if (bbox_order == 1) {
				// tflite
				start_y = (u16)(p_bbox[4 * i + 0] * yres);
				start_x = (u16)(p_bbox[4 * i + 1] * xres);
				end_y = (u16)(p_bbox[4 * i + 2] * yres);
				end_x = (u16)(p_bbox[4 * i + 3] * xres);
			}
			start_x = MAX(MIN(xres, start_x), 0);
			start_y = MAX(MIN(yres, start_y), 0);
			end_x = MAX(MIN(xres, end_x), 0);
			end_y = MAX(MIN(yres, end_y), 0);

			fb_box->width = end_x - start_x;
			fb_box->height = end_y - start_y;
			fb_obj[0]->offset_x = start_x;
			fb_obj[0]->offset_y = start_y;

			fb_obj[1]->offset_x = fb_obj[0]->offset_x;
			tmp = fb_obj[0]->offset_y - fb_textbox->font_size;
			if (tmp < 0) {
				fb_obj[1]->offset_y = fb_obj[0]->offset_y + fb_box->height;
			} else {
				fb_obj[1]->offset_y = tmp;
			}
			textbox->width = fb_textbox->font_size * 30;
			textbox->height = fb_textbox->font_size;
			class_id = fetch_label_id(p_ctx, static_cast<int>(p_id[i]));
			snprintf(fb_textbox->text, sizeof(fb_textbox->text), "%s  %.2f", dataset[class_id], p_score[i]);
			fb_obj[0]->color = fb_obj[1]->color = class_id % SMARTFB_COLOR_NUM;

			smartfb_draw_box(fb_box);
			smartfb_draw_textbox(fb_textbox);
		}
		smartfb_display();
	} else {
		smartfb_clear();
	}

	return 0;
}

static int tvm_draw_framebuffer_classification(tvm_ctx_t *p_ctx, int *id, float *score)
{
	int i = 0;
	tvm_fb_t * p_fb = &p_ctx->fb;
	smartfb_box_t *fb_box = &p_fb->box;
	smartfb_textbox_t *fb_textbox = &p_fb->textbox;
	smartfb_obj_t *fb_obj[2];
	smartfb_box_t *textbox;

	fb_obj[0] = &fb_box->obj;
	textbox = &fb_textbox->box;
	fb_obj[1] = &textbox->obj;

	if (score[0] > p_ctx->conf_th) {
		smartfb_clear_buffer();

		textbox->width = fb_textbox->font_size * 30;
		textbox->height = fb_textbox->font_size;
		fb_obj[1]->color = SMARTFB_COLOR_MAGENTA;

		for (i = 0; i < 5; ++ i) {
			fb_obj[1]->offset_x = 10;
			fb_obj[1]->offset_y = 10 + i * textbox->height;
			snprintf(fb_textbox->text, sizeof(fb_textbox->text), "id: %d score %.4f", id[i], score[i]);
			smartfb_draw_textbox(fb_textbox);
		}
		smartfb_display();
	} else {
		smartfb_clear();
	}

	return 0;
}

static int tvm_check_dsp_state(tvm_iav_t *p_iav)
{
	int state;
	int rval = 0;
	do {
		if (ioctl(p_iav->fd_iav, IAV_IOC_GET_IAV_STATE, &state) < 0) {
			perror("IAV_IOC_GET_IAV_STATE");
			rval = -1;
			break;
		}
		if ((state != IAV_STATE_PREVIEW) && state != IAV_STATE_ENCODING) {
			printf("Error: IAV is not in preview / encoding state, cannot get yuv buf!\n");
			rval = -1;
			break;
		}
	} while (0);

	return rval;
}

static int tvm_map_dsp_buffer(tvm_iav_t *p_iav)
{
	struct iav_querymem query_mem;
	struct iav_mem_part_info *part_info;
	int rval = 0;

	do {
		memset(&query_mem, 0, sizeof(query_mem));
		query_mem.mid = IAV_MEM_PARTITION;
		part_info = &query_mem.arg.partition;
		part_info->pid = IAV_PART_DSP;
		if (ioctl(p_iav->fd_iav, IAV_IOC_QUERY_MEMBLOCK, &query_mem) < 0) {
			perror("IAV_IOC_QUERY_MEMBLOCK");
			rval = -1;
			break;
		}

		p_iav->dsp_size = part_info->mem.length;
		p_iav->dsp_mem = static_cast<uint8_t*>(mmap(NULL, p_iav->dsp_size,
			PROT_READ, MAP_SHARED, p_iav->fd_iav, part_info->mem.addr));
		if (p_iav->dsp_mem == MAP_FAILED) {
			perror("mmap (%d) failed: %s\n");
			rval = -1;
			break;
		}
		p_iav->dsp_phy_addr = part_info->mem.addr;
	} while (0);

	return rval;
}

static int tvm_init_iav(tvm_iav_t *p_iav)
{
	int rval = 0;

	do {
		if ((p_iav->fd_iav = open("/dev/iav", O_RDWR, 0)) < 0) {
			perror("/dev/iav");
			rval = -1;
			break;
		}

		if (tvm_check_dsp_state(p_iav) < 0) {
			rval = -1;
			printf("Error: tvm_check_dsp_state\n");
			break;
		}
		if (tvm_map_dsp_buffer(p_iav) < 0) {
			rval = -1;
			printf("Error: tvm_map_dsp_buffer\n");
			break;
		}
		printf("Init iav done.\n");
	} while (0);

	return rval;
}

static void tvm_deinit_iav(tvm_iav_t *p_iav)
{
	if (p_iav->dsp_mem) {
		if (munmap(p_iav->dsp_mem, p_iav->dsp_size) < 0 ) {
			perror("munmap tvm dsp");
		}
		p_iav->dsp_mem = NULL;
	}

	if (p_iav->fd_iav >= 0) {
		close(p_iav->fd_iav);
		p_iav->fd_iav = -1;
	}
}

static int tvm_get_dsp_input(tvm_iav_t *p_iav)
{
	int rval = 0;
	struct iav_querydesc query_desc;
	struct iav_yuv_cap *data_cap = NULL;

	memset(&query_desc, 0, sizeof(query_desc));

	do {
		if (p_iav->query_buf_type == TVM_QUERY_BUF_CANVAS) {
			/* for canvas buffer, buffer id is canvas id  in 0 ~ 4*/
			query_desc.qid = IAV_DESC_CANVAS;
			query_desc.arg.canvas.canvas_id = p_iav->query_buf_id;
			query_desc.arg.canvas.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;

			if (ioctl(p_iav->fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
				if (errno != EINTR) {
					perror("IAV_IOC_QUERY_DESC");
					rval = -1;
					break;
				}
			}
			data_cap = &query_desc.arg.canvas.yuv;
		} else if (p_iav->query_buf_type == TVM_QUERY_BUF_PYRAMID) {
			/* for pyramid buffer, buffer id is pyramid layer id in 0 ~ 5 */
			query_desc.qid = IAV_DESC_PYRAMID;
			query_desc.arg.pyramid.chan_id = 0;
			query_desc.arg.pyramid.non_block_flag &= ~IAV_BUFCAP_NONBLOCK;

			if (ioctl(p_iav->fd_iav, IAV_IOC_QUERY_DESC, &query_desc) < 0) {
				if (errno != EINTR) {
					perror("IAV_IOC_QUERY_DESC");
					rval = -1;
					break;
				}
			}
			data_cap = &query_desc.arg.pyramid.layers[p_iav->query_buf_id];
		} else {
			printf("Error: Unknown input type [%d]\n", p_iav->query_buf_type);
			rval = -1;
			break;
		}

		if ((data_cap->y_addr_offset == 0) || (data_cap->uv_addr_offset == 0)) {
			printf("Error: Data buffer [%d] address is NULL!\n", p_iav->query_buf_type);
			rval = -1;
			break;
		}
		memcpy(&p_iav->data_cap, data_cap, sizeof(p_iav->data_cap));
	} while(0);

	return rval;
}

static int tvm_alloc_vproc_mem(tvm_ctx_t *p_ctx, DLTensor *in_t)
{
	int rval = 0;
	tvm_vproc_t *p_vproc = &p_ctx->vproc;
	tvm_iav_t *p_iav = &p_ctx->iav;
	vect_desc_t *defm_in = &p_vproc->deform_in;
	vect_desc_t *defm_out = &p_vproc->deform_out;
	vect_desc_t *submean_in = &p_vproc->submean_in;
	vect_desc_t *submean_out = &p_vproc->submean_out;
	vect_desc_t *mean = &p_vproc->mean;
	vect_desc_t *imcvt_in = &p_vproc->imcvt_in;
	vect_desc_t *imcvt_out = &p_vproc->imcvt_out;
	vect_desc_t *scale_in = &p_vproc->scale_in;
	vect_desc_t *scale_out = &p_vproc->scale_out;
	vect_desc_t *flat_in = &p_vproc->flat_in;
	vect_desc_t *flat_out = &p_vproc->flat_out;
	unsigned long mem_size = 0;
	uint8_t *ptr = nullptr;
	uint32_t idx = 0;
	uint32_t need_flat = 1;

	do {
		// check if flatten is needed
		int w = (p_ctx->is_nhwc)? (in_t->shape[2] * in_t->shape[3]) : in_t->shape[3];
		if ((w * in_t->dtype.bits / 8 ) % CAVALRY_PORT_PITCH_ALIGN == 0) {
			need_flat = 0;
		}

		// yuv2rgb & resize
		defm_in->shape.p = 1;
		defm_in->shape.d = 3;
		defm_in->shape.h = p_iav->data_cap.height;
		defm_in->shape.w = p_iav->data_cap.width;
		defm_in->pitch = p_iav->data_cap.pitch;
		defm_in->data_addr = p_iav->dsp_phy_addr + p_iav->data_cap.y_addr_offset;
		defm_in->data_format.sign = 0;
		defm_in->data_format.datasize = 0;
		defm_in->data_format.exp_offset = 0;
		defm_in->data_format.exp_bits = 0;
		defm_in->color_space = CS_NV12;
		p_vproc->dext.uv_offset = p_iav->data_cap.uv_addr_offset - p_iav->data_cap.y_addr_offset;

		defm_out->shape.p = static_cast<uint32_t>(in_t->shape[0]);
		defm_out->shape.d = static_cast<uint32_t>((p_ctx->is_nhwc)? in_t->shape[3] : in_t->shape[1]);
		defm_out->shape.h = static_cast<uint32_t>((p_ctx->is_nhwc)? in_t->shape[1] : in_t->shape[2]);
		defm_out->shape.w = static_cast<uint32_t>((p_ctx->is_nhwc)? in_t->shape[2] : in_t->shape[3]);
		defm_out->data_format = defm_in->data_format;
		defm_out->pitch = \
			ALIGN_32_BYTE(defm_out->shape.w * (1 << defm_out->data_format.datasize));
		defm_out->color_space = (p_ctx->is_bgr)? CS_BGR : CS_RGB;

		mem_size = defm_out->shape.p * defm_out->shape.d * defm_out->shape.h * defm_out->pitch;
		memset(&p_vproc->deform_mem, 0, sizeof(p_vproc->deform_mem));
		p_vproc->deform_mem.size = mem_size;
		if (cavalry_mem_alloc(&p_vproc->deform_mem.size, &p_vproc->deform_mem.phys,
			&p_vproc->deform_mem.virt, 1) < 0) {
			printf("Error: cavalry_mem_alloc\n");
			rval = -1;
			break;
		}
		defm_out->data_addr = p_vproc->deform_mem.phys;

		// submean; output data format should be assigned by user, typical data format is (1,0,0,0)
		submean_in->shape = defm_out->shape;
		submean_in->data_format = defm_out->data_format;
		submean_in->pitch = defm_out->pitch;
		submean_in->color_space = CS_VECT;
		submean_in->data_addr = defm_out->data_addr;

		submean_out->shape = submean_in->shape;
		submean_out->data_format.sign = 1;
		submean_out->data_format.datasize = 0;
		submean_out->data_format.exp_offset = 0;
		submean_out->data_format.exp_bits = 0;
		submean_out->pitch = submean_in->pitch;
		submean_out->color_space = CS_VECT;

		mem_size = submean_out->shape.p * submean_out->shape.d * submean_out->shape.h * submean_out->pitch;
		memset(&p_vproc->submean_mem, 0, sizeof(p_vproc->submean_mem));
		p_vproc->submean_mem.size = mem_size;
		if (cavalry_mem_alloc(&p_vproc->submean_mem.size, &p_vproc->submean_mem.phys,
			&p_vproc->submean_mem.virt, 1) < 0) {
			printf("Error: cavalry_mem_alloc\n");
			rval = -1;
			break;
		}
		submean_out->data_addr = p_vproc->submean_mem.phys;

		// mean data; data format should be assigned by user, typical data format is (0,0,0,0)
		mean->shape = submean_in->shape;
		mean->data_format.sign = 0;
		mean->data_format.datasize = 0;
		mean->data_format.exp_offset = 0;
		mean->data_format.exp_bits = 0;
		mean->pitch = submean_in->pitch;
		mean->color_space = CS_VECT;

		mem_size = mean->shape.p * mean->shape.d * mean->shape.h * mean->pitch;
		memset(&p_vproc->mean_mem, 0, sizeof(p_vproc->mean_mem));
		p_vproc->mean_mem.size = mem_size;
		if (cavalry_mem_alloc(&p_vproc->mean_mem.size, &p_vproc->mean_mem.phys,
			&p_vproc->mean_mem.virt, 1) < 0) {
			printf("Error: cavalry_mem_alloc\n");
			rval = -1;
			break;
		}
		mean->data_addr = p_vproc->mean_mem.phys;

		ptr = static_cast<uint8_t*>(p_vproc->mean_mem.virt);
		for (uint32_t p =0; p < mean->shape.p; ++p) {
			for (uint32_t d = 0; d < mean->shape.d; ++d) {
				for (uint32_t h = 0; h < mean->shape.h; ++h) {
					for (uint32_t w = 0; w < mean->shape.w; ++w) {
						idx = p * mean->shape.d * mean->shape.h * mean->pitch;
						idx += d * mean->shape.h * mean->pitch;
						idx += h * mean->pitch + w;
						ptr[idx] = static_cast<uint8_t>(p_ctx->mean_rgb[d]);
					}
				}
			}
		}
		cavalry_mem_sync_cache(p_vproc->mean_mem.size, p_vproc->mean_mem.phys, 1,0);

		// NCHW -> NHWC (AMB2OCV) to fit tf models
		if (p_ctx->is_nhwc) {
			imcvt_in->shape = submean_out->shape;
			imcvt_in->data_format = submean_out->data_format;
			imcvt_in->pitch = submean_out->pitch;
			imcvt_in->color_space = (p_ctx->is_bgr)? CS_BGR : CS_RGB;
			imcvt_in->data_addr = submean_out->data_addr;

			imcvt_out->shape = imcvt_in->shape;
			imcvt_out->data_format = imcvt_in->data_format;
			// h x (wxc), pitch is based on (wxc)
			imcvt_out->pitch = \
				ALIGN_32_BYTE(imcvt_out->shape.w * imcvt_out->shape.d * (1 << imcvt_out->data_format.datasize));
			imcvt_out->color_space = (p_ctx->is_bgr)? CS_BGR_ITL : CS_RGB_ITL;;

			mem_size = imcvt_out->shape.p * imcvt_out->shape.h * imcvt_out->pitch;
			memset(&p_vproc->imcvt_mem, 0, sizeof(p_vproc->imcvt_mem));
			p_vproc->imcvt_mem.size = mem_size;
			if (cavalry_mem_alloc(&p_vproc->imcvt_mem.size, &p_vproc->imcvt_mem.phys,
				&p_vproc->imcvt_mem.virt, 1) < 0) {
				printf("Error: cavalry_mem_alloc\n");
				rval = -1;
				break;
			}
			imcvt_out->data_addr = p_vproc->imcvt_mem.phys;
		}

		// scale, dtcvt (data format should be assigned by user which is shown in compiled.json file)
		if (p_ctx->is_nhwc) {
			scale_in->shape.p = 1;
			scale_in->shape.d = 1;
			scale_in->shape.h = imcvt_out->shape.h;
			scale_in->shape.w = imcvt_out->shape.w * imcvt_out->shape.d;
			scale_in->data_format = imcvt_out->data_format;
			scale_in->pitch = imcvt_out->pitch;
			scale_in->data_addr = imcvt_out->data_addr;
		} else {
			scale_in->shape = submean_out->shape;
			scale_in->data_format = submean_out->data_format;
			scale_in->pitch = submean_out->pitch;
			scale_in->data_addr = submean_out->data_addr;
		}
		scale_in->color_space = CS_VECT;

		// data format should be fp32 if no flatten is needed and dtcvt is the last VPROC preproc op
		if (need_flat) {
			scale_out->data_format.sign = 1;
			scale_out->data_format.datasize = 0;
			scale_out->data_format.exp_offset = (int8_t)(log2(1.0 / p_ctx->scale) + 0.5);
			scale_out->data_format.exp_bits = 0;
		} else {
			scale_out->data_format.sign = 1;
			scale_out->data_format.datasize = 2;
			scale_out->data_format.exp_offset = 0;
			scale_out->data_format.exp_bits = 7;
		}
		scale_out->shape = scale_in->shape;
		scale_out->pitch = \
			ALIGN_32_BYTE(scale_out->shape.w * (1 << scale_out->data_format.datasize));
		scale_out->color_space = CS_VECT;

		mem_size = scale_out->shape.p * scale_out->shape.d * scale_out->shape.h * scale_out->pitch;
		memset(&p_vproc->scale_mem, 0, sizeof(p_vproc->scale_mem));
		p_vproc->scale_mem.size = mem_size;
		if (cavalry_mem_alloc(&p_vproc->scale_mem.size, &p_vproc->scale_mem.phys,
			&p_vproc->scale_mem.virt, 1) < 0) {
			printf("Error: cavalry_mem_alloc\n");
			rval = -1;
			break;
		}
		scale_out->data_addr = p_vproc->scale_mem.phys;

		// flatten, reshap data to 1-dim
		if (need_flat) {
			flat_in->shape = scale_out->shape;
			flat_in->data_format = scale_out->data_format;
			flat_in->pitch = scale_out->pitch;
			flat_in->color_space = CS_VECT;
			flat_in->data_addr = scale_out->data_addr;

			flat_out->shape.p = 1;
			flat_out->shape.d = 1;
			flat_out->shape.h = 1;
			flat_out->shape.w = flat_in->shape.p * flat_in->shape.d * flat_in->shape.h * flat_in->shape.w;
			flat_out->data_format.sign = 1;
			flat_out->data_format.datasize = 2;
			flat_out->data_format.exp_offset = 0;
			flat_out->data_format.exp_bits = 7;
			flat_out->pitch = ALIGN_32_BYTE(flat_out->shape.w * (1 << flat_out->data_format.datasize));
			flat_out->color_space = CS_VECT;

			mem_size = flat_out->pitch;
			memset(&p_vproc->flat_mem, 0, sizeof(p_vproc->flat_mem));
			p_vproc->flat_mem.size = mem_size;
			if (cavalry_mem_alloc(&p_vproc->flat_mem.size, &p_vproc->flat_mem.phys,
				&p_vproc->flat_mem.virt, 1) < 0) {
				printf("Error: cavalry_mem_alloc\n");
				rval = -1;
				break;
			}
			flat_out->data_addr = p_vproc->flat_mem.phys;
		}
		p_vproc->need_flat = need_flat;
	} while(0);

	return rval;
}

static int tvm_vproc_data_process(tvm_ctx_t *p_ctx)
{
	int rval = 0;
	tvm_vproc_t *p_vproc = &p_ctx->vproc;
	tvm_iav_t *p_iav = &p_ctx->iav;
	vect_desc_t *defm_in = &p_vproc->deform_in;
	vect_desc_t *defm_out = &p_vproc->deform_out;

	defm_in->data_addr = p_iav->dsp_phy_addr + p_iav->data_cap.y_addr_offset;
	p_vproc->dext.uv_offset = static_cast<int32_t>(p_iav->data_cap.uv_addr_offset -
		p_iav->data_cap.y_addr_offset);

	do {
		if (vproc_image_deformation(defm_in, defm_out, &p_vproc->dext) < 0) {
			printf("Error: vproc_image_deformation.\n");
			rval = -1;
			break;
		}
		if (vproc_submean(&p_vproc->submean_in, &p_vproc->mean,
			&p_vproc->submean_out) < 0) {
			printf("Error: vproc_submean.\n");
			rval = -1;
			break;
		}
		if (p_ctx->is_nhwc) {
			if (vproc_imcvt(&p_vproc->imcvt_in, &p_vproc->imcvt_out)) {
				printf("Error: vproc_imcvt\n");
				rval = -1;
				break;
			}
		}
		if (vproc_scale_ext(&p_vproc->scale_in, &p_vproc->scale_out, p_ctx->scale) < 0) {
			printf("Error: vproc_scale_ext.\n");
			rval = -1;
			break;
		}
		if (p_vproc->need_flat) {
			if (vproc_flatten(&p_vproc->flat_in, &p_vproc->flat_out)) {
				printf("Error: vproc_flatten.\n");
				rval = -1;
				break;
			}
		}
	} while(0);

	return rval;
}

static void tvm_free_vproc_mem(tvm_ctx_t *p_ctx)
{
	tvm_vproc_t *p_vproc = &p_ctx->vproc;

	if (p_vproc->lib_mem.virt && p_vproc->lib_mem.size) {
		if (cavalry_mem_free(p_vproc->lib_mem.size, p_vproc->lib_mem.phys,
			p_vproc->lib_mem.virt) < 0) {
			printf("Error: cavalry_mem_free lib_mem\n");
		}
	}
	if (p_vproc->deform_mem.virt && p_vproc->deform_mem.size) {
		if (cavalry_mem_free(p_vproc->deform_mem.size, p_vproc->deform_mem.phys,
			p_vproc->deform_mem.virt) < 0) {
			printf("Error: cavalry_mem_free deform_mem\n");
		}
	}
	if (p_vproc->submean_mem.virt && p_vproc->submean_mem.size) {
		if (cavalry_mem_free(p_vproc->submean_mem.size, p_vproc->submean_mem.phys,
			p_vproc->submean_mem.virt) < 0) {
			printf("Error: cavalry_mem_free submean_mem\n");
		}
	}
	if (p_vproc->mean_mem.virt && p_vproc->mean_mem.size) {
		if (cavalry_mem_free(p_vproc->mean_mem.size, p_vproc->mean_mem.phys,
			p_vproc->mean_mem.virt) < 0) {
			printf("Error: cavalry_mem_free mean_mem\n");
		}
	}
	if (p_vproc->scale_mem.virt && p_vproc->scale_mem.size) {
		if (cavalry_mem_free(p_vproc->scale_mem.size, p_vproc->scale_mem.phys,
			p_vproc->scale_mem.virt) < 0) {
			printf("Error: cavalry_mem_free scale_mem\n");
		}
	}
	if (p_ctx->is_nhwc && p_vproc->imcvt_mem.virt && p_vproc->imcvt_mem.size) {
		if (cavalry_mem_free(p_vproc->imcvt_mem.size, p_vproc->imcvt_mem.phys,
			p_vproc->imcvt_mem.virt) < 0) {
			printf("Error: cavalry_mem_free imcvt_mem\n");
		}
	}
	if (p_vproc->need_flat && p_vproc->flat_mem.virt && p_vproc->flat_mem.size) {
		if (cavalry_mem_free(p_vproc->flat_mem.size, p_vproc->flat_mem.phys,
			p_vproc->flat_mem.virt) < 0) {
			printf("Error: cavalry_mem_free flat_mem\n");
		}
	}
}

static int tvm_init_vproc(tvm_ctx_t *p_ctx)
{
	tvm_vproc_t *p_vproc = &p_ctx->vproc;
	uint32_t size = 0;
	int rval = 0;

	do {
		if ((p_vproc->fd_cav = cavalry_mem_get_fd()) < 0) {
			if ((p_vproc->fd_cav = open(CAVALRY_DEV_NODE, O_RDWR, 0)) < 0) {
				perror(CAVALRY_DEV_NODE);
				rval = -1;
				break;
			}
			if (cavalry_mem_init(p_vproc->fd_cav, 0) < 0) {
				printf("Error: cavalry_mem_init.\n");
				rval = -1;
				break;
			}
		}

		if (vproc_init(TVM_VPROC_BIN, &size) < 0) {
			printf("Error: vproc_init\n");
			rval = -1;
			break;
		}

		// alloc vproc main mem
		memset(&p_vproc->lib_mem, 0, sizeof(p_vproc->lib_mem));
		p_vproc->lib_mem.size = size;
		if (cavalry_mem_alloc(&p_vproc->lib_mem.size, &p_vproc->lib_mem.phys,
			&p_vproc->lib_mem.virt, 0) < 0) {
			printf("Error: cavalry_mem_alloc\n");
			rval = -1;
			break;
		}
		if (vproc_load(p_vproc->fd_cav, static_cast<uint8_t*>(p_vproc->lib_mem.virt),
			p_vproc->lib_mem.phys, p_vproc->lib_mem.size) < 0) {
			printf("Error: vproc_load\n");
			rval = -1;
			break;
		}
	} while (0);

	return rval;
}

static void tvm_deinit_vproc(tvm_vproc_t *p_vproc)
{
	vproc_exit();
	// cavalry_mem_exit and fd_cav closing is done at tvm::runtime::Module destruction
}

static int tvm_read_buffer(tvm_ctx_t *p_ctx, DLTensor *t, DLTensor *phys_t)
{
	int rval = 0;
	tvm_vproc_t *p_vproc = &p_ctx->vproc;
	uint32_t need_flat = p_vproc->need_flat;

	do {
		if (p_ctx->dev.type == kDLAmba) {
			/* assign vproc output buffer physcial addr if there is preprocessing;
			* assign Y/UV dsp addr directly if no preprocessing is needed */
			if (need_flat) {
				phys_t->data = reinterpret_cast<void*>(p_vproc->flat_mem.phys);
			} else {
				phys_t->data = reinterpret_cast<void*>(p_vproc->scale_mem.phys);
			}
			phys_t->device = t->device;
			phys_t->ndim = t->ndim;
			phys_t->dtype = t->dtype;
			phys_t->shape = t->shape;
			phys_t->strides = nullptr;
			phys_t->byte_offset = 0;
		}
	} while(0);

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
	tvm_draw_framebuffer_classification(p_ctx, prob_id, prob_softmax);

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
		pfr = tvm::runtime::Registry::Get("tvm.graph_executor.create");
		if (!pfr) {
			printf("Error: TVM graph runtime is not enabled.\n");
			rval = -1;
			break;
		}

		mod = (*pfr)(json_data, mod_syslib, p_dev->type, p_dev->id);
		// load patameters
		tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
		load_params(params_arr);
	} while(0);

	return rval;
}

static int tvm_prepare_live_mode(tvm_ctx_t *p_ctx, DLTensor *in_t)
{
	int rval = 0;

	do {
		if (tvm_alloc_vproc_mem(p_ctx, in_t) < 0) {
			printf("Error: tvm_alloc_vproc_mem\n");
			rval = -1;
			break;
		}
	} while(0);

	return rval;
}

static int tvm_proc_live_mode(tvm_ctx_t *p_ctx, DLTensor *in_t, DLTensor *phys_t)
{
	int rval = 0;

	do {
		if (tvm_get_dsp_input(&p_ctx->iav) < 0) {
			printf("Error: tvm_get_dsp_input\n");
			rval = -1;
			break;
		}
		if (tvm_vproc_data_process(p_ctx) < 0) {
			printf("Error: tvm_vproc_data_process.\n");
			rval = -1;
			break;
		}
		if (tvm_read_buffer(p_ctx, in_t, phys_t) < 0) {
			printf("Error: tvm_read_buffer.\n");
			rval = -1;
			break;
		}
	} while(0);

	return rval;
}

static int tvm_process_outputs(tvm_ctx_t *p_ctx,tvm_net_cfg_t *p_net,
		DLTensor **out_t, DLTensor **in_t)
{
	int rval = 0;

	do {
		if (p_net->net_type == TVM_NET_CLASSIFICATION) {
			// imagenet pretrained classification network from gluoncv
			rval = tvm_process_classification(p_ctx, out_t[0],
					out_t[0]->shape[out_t[0]->ndim - 1]);
		} else if (p_net->net_type == TVM_NET_OBJECT_DETECT) {
			rval = tvm_draw_framebuffer_ssd(p_ctx, out_t, in_t[0]);
		}
	} while(0);

	return rval;
}

static int tvm_execute_one_net(tvm_ctx_t *p_ctx)
{
	int rval = 0;
	tvm_net_cfg_t *p_net = &p_ctx->net_cfg;
	tvm::runtime::Module mod;

	std::string mod_dir = get_file_dirname(p_net->model_fn);
	if (ConfigAmbaEngineLocation(mod_dir.c_str())) {
		printf("Error: ConfigAmbaEngineLocation\n");
		return -1;
	}

	// device type and id must be updated before TVM module is loaded

	// 32-bit device id: 31~24: blank; 23~8: DSP pitch; 7~0: device id for Amba DSP

	/* if DSP Y/UV data is feed into target subgraph directly, should query DSP data pitch first;
	if vproc is used to do preprocessing, then no need to specify data pitch in device id.
	p_ctx->dev.id = (p_ctx->dev.id & 0xFF)	| (p_ctx->iav.data_cap.pitch << 8);*/

	if (tvm_load_module(p_net, &p_ctx->dev, mod) < 0) {
		printf("Error: tvm_load_module\n");
		rval = -1;
		return rval;
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
	DLTensor phys_in_t;
	DLTensor *out_t[num_outputs] = {nullptr};

	do {
		for (int i = 0; i < num_outputs; ++i) {
			out_t[i] = get_output(i).operator DLTensor*();
		}

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
		if (num_inputs > 1) {
			printf("Error: only one input node is supported in this live mode example.\n");
			rval = -1;
			break;
		}
		if (in_t[0]->ndim != 4) {
			printf("Error: tensor dimension size should be 4 instead of %d\n", in_t[0]->ndim);
			rval = -1;
			break;
		}
		if (tvm_prepare_live_mode(p_ctx, in_t[0]) < 0) {
			printf("Error: tvm_prepare_live_mode\n");
			rval = -1;
			break;
		}
	} while(0);

	do {
		if (rval) break;
		if (tvm_proc_live_mode(p_ctx, in_t[0], &phys_in_t) < 0) {
			printf("Error: tvm_proc_live_mode\n");
			rval = -1;
			break;
		}
		if (p_ctx->dev.type == kDLAmba) {
			set_input_zc(0, &phys_in_t);
		}

		// run module
		run();

		if (tvm_process_outputs(p_ctx, p_net, out_t, in_t)< 0) {
			printf("Error: tvm_process_outputs.\n");
			rval = -1;
			break;
		}
	} while(run_flag);

	/* free mem allocated for vproc */
	tvm_free_vproc_mem(p_ctx);

	return rval;
}

static void sigstop(int)
{
	run_flag = 0;
	printf("sigstop msg, exit test_amba_tvm_live.\n");
}

int main(int argc, char *argv[])
{
	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);

	int rval = 0;
	tvm_ctx_t tvm_ctx;
	tvm_ctx_t *p_ctx = &tvm_ctx;
	int fb_inited = 0;

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
		if (tvm_init_iav(&p_ctx->iav) < 0) {
			printf("Error: tvm_init_iav.\n");
			rval = -1;
			break;
		}
		if (tvm_get_dsp_input(&p_ctx->iav) < 0) {
			printf("Error: tvm_get_dsp_input\n");
			rval = -1;
			break;
		}
		if (tvm_init_fb(&p_ctx->fb) < 0) {
			printf("Error: tvm_init_fb.\n");
			rval = -1;
			break;
		}
		fb_inited = 1;
		if (tvm_init_vproc(p_ctx) < 0) {
			printf("Error: tvm_init_vproc.\n");
			rval = -1;
			break;
		}
		if (tvm_execute_one_net(p_ctx) < 0) {
			printf("Error: tvm_execute_one_net.\n");
			rval = -1;
			break;
		}
	} while(0);

	if (fb_inited) {
		tvm_deinit_fb();
		fb_inited = 0;
	}
	tvm_deinit_iav(&p_ctx->iav);
	tvm_deinit_vproc(&p_ctx->vproc);

	return rval;
}


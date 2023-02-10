################################################################################
## make.inc
##
## History:
##    2022/05/20 - [Monica Yang] Created
##
## Copyright [2022] Ambarella International LP.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
################################################################################


ifeq ($(BUILD_AMBARELLA_UNIT_TESTS_AMBA_NEO), y)

LOCAL_PATH	:= $(call my-dir)

################################################

include $(CLEAR_VARS)

LOCAL_TARGET	:= test_amba_dlr
LOCAL_SRCS	:= $(LOCAL_PATH)/test_amba_dlr.cpp

LOCAL_CFLAGS 	+= -I$(AMBA_DLR_ROOT_DIR)/prebuild/amba/include \
		-I$(AMBA_DLR_ROOT_DIR)/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/json \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dlpack/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dmlc-core/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/src/runtime

LOCAL_LDFLAGS	:= -lpthread -lm \
		-L$(AMBA_DLR_ROOT_DIR)/prebuild/amba/lib/linux \
		-ldlr -ltvm_runtime -lamba_tvm \
		-Wl,-rpath-link=$(FAKEROOT_DIR)/usr/lib

LOCAL_LIBS   	:= libcavalry_mem.so libnnctrl.so libvproc.so

include $(BUILD_APP)

.PHONY: $(LOCAL_TARGET)

$(LOCAL_TARGET): $(LOCAL_MODULE)
	@mkdir -p $(UNIT_TEST_PATH)/
	@cp -dpRf $< $(UNIT_TEST_PATH)/
	@echo "Build $@ Done."

$(call add-target-into-build, $(LOCAL_TARGET))

################################################

OPENCV_PATH	:= $(PREBUILD_OSS_DIR)/opencv

include $(CLEAR_VARS)

LOCAL_TARGET	:= test_amba_dlr_live
LOCAL_SRCS	:= $(LOCAL_PATH)/test_amba_dlr_live.cpp

LOCAL_CFLAGS 	+= -I$(AMB_TOPDIR)/packages/cavalry_mem/inc \
		-I$(AMB_TOPDIR)/packages/vproc/$(AMBA_CAVALRY_ARCH)/inc \
		-I$(AMB_TOPDIR)/packages/smartfb \
		-I$(AMB_TOPDIR)/include/cavalry/$(AMBA_CAVALRY_ARCH) \
		-I$(AMB_TOPDIR)/packages/eazyai/inc \
		-I$(OPENCV_PATH)/include/opencv \
		-I$(OPENCV_PATH)/include \
		-I$(AMBA_DLR_ROOT_DIR)/prebuild/amba/include \
		-I$(AMBA_DLR_ROOT_DIR)/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/json \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dlpack/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dmlc-core/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/src/runtime

LOCAL_LDFLAGS	:= -lpthread -lm \
		-L$(AMBA_DLR_ROOT_DIR)/prebuild/amba/lib/linux \
		-ldlr -ltvm_runtime -lamba_tvm \
		-L$(OPENCV_PATH)/usr/lib \
		-lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
		-Wl,-rpath-link=$(FAKEROOT_DIR)/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/libjpeg-turbo/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/libpng/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/zlib/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/tbb/usr/lib

LOCAL_LIBS   	:= libcavalry_mem.so libnnctrl.so libvproc.so libeazyai.so

include $(BUILD_APP)

.PHONY: $(LOCAL_TARGET)

$(LOCAL_TARGET): $(LOCAL_MODULE)
	@mkdir -p $(UNIT_TEST_PATH)/
	@cp -dpRf $< $(UNIT_TEST_PATH)/
	@echo "Build $@ Done."

$(call add-target-into-build, $(LOCAL_TARGET))

################################################

include $(CLEAR_VARS)

LOCAL_TARGET	:= test_amba_tvm
LOCAL_SRCS	:= $(LOCAL_PATH)/test_amba_tvm.cpp

LOCAL_CFLAGS 	+= -I$(AMBA_DLR_ROOT_DIR)/prebuild/amba/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dlpack/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dmlc-core/include \
		-std=c++17

LOCAL_LDFLAGS	:= -lpthread -lm \
		-L$(AMBA_DLR_ROOT_DIR)/prebuild/amba/lib/linux \
		-ltvm_runtime -lamba_tvm -lnnctrl -lvproc -lcavalry_mem \
		-Wl,-rpath-link=$(FAKEROOT_DIR)/usr/lib

LOCAL_LIBS   	:= libcavalry_mem.so libnnctrl.so libvproc.so

include $(BUILD_APP)

.PHONY: $(LOCAL_TARGET)

$(LOCAL_TARGET): $(LOCAL_MODULE)
	@mkdir -p $(UNIT_TEST_PATH)/
	@cp -dpRf $< $(UNIT_TEST_PATH)/
	@echo "Build $@ Done."

$(call add-target-into-build, $(LOCAL_TARGET))

################################################

include $(CLEAR_VARS)

LOCAL_TARGET	:= test_amba_tvm_live
LOCAL_SRCS	:= $(LOCAL_PATH)/test_amba_tvm_live.cpp

LOCAL_CFLAGS 	+= -I$(AMB_TOPDIR)/packages/cavalry_mem/inc \
		-I$(AMB_TOPDIR)/packages/vproc/$(AMBA_CAVALRY_ARCH)/inc \
		-I$(AMB_TOPDIR)/packages/smartfb \
		-I$(AMB_TOPDIR)/include/cavalry/$(AMBA_CAVALRY_ARCH) \
		-I$(AMBA_DLR_ROOT_DIR)/prebuild/amba/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dlpack/include \
		-I$(AMBA_DLR_ROOT_DIR)/3rdparty/tvm/3rdparty/dmlc-core/include \
		-std=c++17

LOCAL_LDFLAGS	:= -lpthread -lm \
		-L$(AMBA_DLR_ROOT_DIR)/prebuild/amba/lib/linux \
		-ltvm_runtime -lamba_tvm -lnnctrl -lvproc -lcavalry_mem \
		-lsmartfb -ltextinsert_v2 \
		-Wl,-rpath-link=$(FAKEROOT_DIR)/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/freetype/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/zlib/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/bzip2/usr/lib \
		-Wl,-rpath-link=$(PREBUILD_OSS_DIR)/libpng/usr/lib

LOCAL_LIBS   	:= libcavalry_mem.so libnnctrl.so \
		libvproc.so libsmartfb.so libtextinsert_v2.so

include $(BUILD_APP)

.PHONY: $(LOCAL_TARGET)

$(LOCAL_TARGET): $(LOCAL_MODULE)
	@mkdir -p $(UNIT_TEST_PATH)/
	@cp -dpRf $< $(UNIT_TEST_PATH)/
	@echo "Build $@ Done."

$(call add-target-into-build, $(LOCAL_TARGET))

################################################

endif

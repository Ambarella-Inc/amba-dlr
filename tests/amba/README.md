﻿# Amba-Neo SDK UG

This is a user guide to illustrate how to build Ambarella Linux SDK with AWS Neo DLR/TVM integration.


## 1. DLR/TVM Source Code

DLR/TVM source code with Ambarella CVFlow engine implementation is stored as github repository. User need to contact Ambarella IT-support for access permission. Please be noted that both amba-dlr and amba-tvm access permission are needed.

Command to get source code

	$ git clone https://username:personaltoken@github.com/Ambarella-Inc/amba-dlr.git -b latest_release
	$ cd amba-dlr && git submodule update --init --recursive

Git doesn’t allow users to login with passwords any more, only with [personal access tokens](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) . For above steps users can paste the personal access token when GitHub asks for password. For an automated setup, users will need to [use GitHub CLI to authenticate requests](https://docs.github.com/en/get-started/getting-started-with-git/caching-your-github-credentials-in-git).

Please follow below command to update local source code when there is further update in remote git repository.

	$ cd amba-dlr && git pull & git submodule update

## 2. Build with Ambarella Linux SDK

There is unit test app that illustrate how to leverage the Ambarella CVFlow engine implemented DLR/TVM. Below steps show how to build unit test apps in Ambarella Linux SDK.

### 2.1 Get unit test source code

Unit test includes apps for DLR and TVM. The source code files are located in amba-dlr github repository. The folder path is amba-dlr/tests/amba/amba_neo.

To build these unit test apps in SDK, users need to copy the files in folder amba_neo to SDK at location ambarella/unit_test/private/amba_neo. Please don't forget file AmbaConfig and make.inc which are important for SDK makefile framework.

### 2.2 Enable DLR/TVM unit test

After amba_neo is copied to SDK, please do menuconfig in SDK command line to enable unit test compilation. Unit test relies on amba-dlr source code. User needs to specify the amba-dlr folder path (AMBA_DLR_ROOT_DIR) in user side with absolute folder path.

	$ make menuconfig  
	  -> Ambarella Unit Test Configuration (BUILD_AMBARELLA_UNIT_TESTS [=y])  
	    -> Ambarella Private Linux Unit test configs (BUILD_AMBARELLA_PRIVATE_LINUX_UNIT_TESTS [=y])  
	      -> Build Amba-Neo unit tests (BUILD_AMBARELLA_UNIT_TESTS_AMBA_NEO [=y])  
	        ->AMBA_DLR_ROOT_DIR [=$(AMB_TOPDIR)/../../amba-dlr]

### 2.3 Build firmware and burn it to CV2x EVK

Build firmware with DLR/TVM unit test apps.

	$ make -j4

Build DLR/TVM unit test apps respectively.

	$ make test_amba_dlr -j4

	$ make test_amba_dlr_live -j4

	$ make test_amba_tvm -j4

	$ make test_amba_tvm_live -j4


##  3. Runtime in Silicon Side

Steps:

1. Do model compilation in AWS Sagemaker online service and get model_compiled.tar.gz.

There are preparation steps to do compilation job in Sagemaker service. More details are [here](https://docs.aws.amazon.com/sagemaker/latest/dg/neo-troubleshooting-target-devices-ambarella.html).

2. Download model_compiled.tar.gz to EVK and untar it

3. Copy untared dynamic libraries to evk /usr/lib folder

		$cp amba_files/lib* /usr/lib

4. Run test_amba_tvm / test_amba_tvm_live / test_amba_dlr / test_amba_dlr_live to do model inference

NOTES: "test_amba_tvm / test_amba_tvm_live / test_amba_dlr / test_amba_dlr_live" are only sample code and users should write their own app in real produce design.

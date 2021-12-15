/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file cuda_module.cc
 */
#include "cuda_module.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/runtime/registry.h>

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <thread>

#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <fstream>

#include "../file_utils.h"
#include "../meta_data.h"
#include "../pack_args.h"
#include "../thread_storage_scope.h"
#include "cuda_common.h"

namespace tvm {
namespace runtime {

class CUDAModuleNode;

void limitSM(CUDAModuleNode* m_, int device_id, int max_core, int n_stream);

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class CUDAModuleNode : public runtime::ModuleNode {
 public:
  std::thread t;
  cudaStream_t my_stream;
  explicit CUDAModuleNode(std::string data, std::string fmt,
                          std::unordered_map<std::string, FunctionInfo> fmap,
                          std::string cuda_source)
      : data_(data), fmt_(fmt), fmap_(fmap), cuda_source_(cuda_source) {
    std::fill(module_.begin(), module_.end(), nullptr);
    cudaStreamCreate(&my_stream);
    // CUDAThreadEntry::ThreadLocal()->stream = my_stream;
  }
  // destructor
  ~CUDAModuleNode() {
    t.join();
    cudaStreamDestroy(my_stream);
    for (size_t i = 0; i < module_.size(); ++i) {
      if (module_[i] != nullptr) {
        CUDA_CALL(cudaSetDevice(static_cast<int>(i)));
        CUDA_DRIVER_CALL(cuModuleUnload(module_[i]));
      }
    }
  }

  const char* type_key() const final { return "cuda"; }

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final;

  void SaveToFile(const std::string& file_name, const std::string& format) final {
    std::string fmt = GetFileFormat(file_name, format);
    std::string meta_file = GetMetaFilePath(file_name);
    if (fmt == "cu") {
      ICHECK_NE(cuda_source_.length(), 0);
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, cuda_source_);
    } else {
      ICHECK_EQ(fmt, fmt_) << "Can only save to format=" << fmt_;
      SaveMetaDataToFile(meta_file, fmap_);
      SaveBinaryToFile(file_name, data_);
    }
  }

  void SaveToBinary(dmlc::Stream* stream) final {
    stream->Write(fmt_);
    stream->Write(fmap_);
    stream->Write(data_);
  }

  std::string GetSource(const std::string& format) final {
    if (format == fmt_) return data_;
    if (cuda_source_.length() != 0) {
      return cuda_source_;
    } else {
      if (fmt_ == "ptx") return data_;
      return "";
    }
  }
  
  void launchLimitSM(int device_id, int max_sm) {
     t = std::thread(limitSM, this, device_id, max_sm, 16);
     sleep(5);
     printf("wake up launch (expected some kernel wake up) %p\n", this);

  }

  // get a CUfunction from primary context in device_id
  CUfunction GetFunc(int device_id, const std::string& func_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    CUfunction func;
    CUresult result = cuModuleGetFunction(&func, module_[device_id], func_name.c_str());
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetFunction " << func_name << " failed with error: " << msg;
    }
    return func;
  }
  // get a global var from primary context in device_id
  CUdeviceptr GetGlobal(int device_id, const std::string& global_name, size_t expect_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // must recheck under the lock scope
    if (module_[device_id] == nullptr) {
      CUDA_DRIVER_CALL(cuModuleLoadData(&(module_[device_id]), data_.c_str()));
    }
    CUdeviceptr global;
    size_t nbytes;

    CUresult result = cuModuleGetGlobal(&global, &nbytes, module_[device_id], global_name.c_str());
    ICHECK_EQ(nbytes, expect_nbytes);
    if (result != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(result, &msg);
      LOG(FATAL) << "CUDAError: cuModuleGetGlobal " << global_name << " failed with error: " << msg;
    }
    return global;
  }

 private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The cuda source.
  std::string cuda_source_;
  // the internal modules per GPU, to be lazily initialized.
  std::array<CUmodule, kMaxNumGPUs> module_;
  // internal mutex when updating the module
  std::mutex mutex_;
};

void limitSM(CUDAModuleNode* m_, int device_id, int max_core, int n_stream) {
    cudaStream_t stream[n_stream];
    CUresult result;
    for (int i = 0; i < n_stream; ++i) {
	cudaStreamCreate(&stream[i]);
    	CUfunction func = m_->GetFunc(device_id, "sleepKernel");
	int *ptr_maxcore = &max_core;
	result = cuLaunchKernel(func, 68, 1, 1, 1, 1, 1, 0, static_cast<CUstream>(stream[i]), (void**)&ptr_maxcore, nullptr);
	if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
		std::ostringstream os;
		const char* msg;
		cuGetErrorName(result, &msg);
		os << "CUDALaunch Error: sleepKernel " << msg << "\n"
			<< ")\n";
		LOG(FATAL) << os.str();
	}
    }
    for (int i = 0; i < n_stream; ++i) {
	cudaStreamDestroy(stream[i]);
    }
}

using std::string;

string readFileIntoString(const string& path) {
	std::ifstream input_file(path);
	if (!input_file.is_open()) {
		std::cerr << "Could not open the file - '"
			<< path << "'\n";
		exit(EXIT_FAILURE);
	}
	return string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}

// a wrapped function class to get packed func.
class CUDAWrappedFunc {
 public:
  mutable bool first_operator;
  static int is_first_operator_shared;
  static int count;
  mutable int id;
  mutable int operator_count;
  mutable bool is_unique_wrapper;
  CUDAWrappedFunc() {
    first_operator = true;
  }
  ~CUDAWrappedFunc() {
  }
  // initialize the CUDA function.
  void Init(CUDAModuleNode* m, ObjectPtr<Object> sptr, const std::string& func_name,
            size_t num_void_args, const std::vector<std::string>& launch_param_tags) {
    first_operator = true;
    operator_count = 0;
    m_ = m;
    sptr_ = sptr;
    func_name_ = func_name;
    std::fill(fcache_.begin(), fcache_.end(), nullptr);
    launch_param_config_.Init(num_void_args, launch_param_tags);
  }
  // invoke the function with void arguments
  void operator()(TVMArgs args, TVMRetValue* rv, void** void_args) const {
    int device_id;



    CUDA_CALL(cudaGetDevice(&device_id));
    if (fcache_[device_id] == nullptr) {
      fcache_[device_id] = m_->GetFunc(device_id, func_name_);
    }

    if (first_operator) {
	id = count;
	count++;
        first_operator = false;
	if (is_first_operator_shared) {
		is_first_operator_shared = false;
		is_unique_wrapper = true;
	} else {
		is_unique_wrapper = false;
	}
    }

    operator_count++;
    if (is_unique_wrapper && operator_count == 1) {
	    int max_sm = 0;
	    std::istringstream is(readFileIntoString("/mnt/tvm-getstarted/sm_cores.txt"));
	    is >> max_sm;
	    m_->launchLimitSM(device_id, max_sm);
    }

    // sleep(5);

    // CUstream strm = static_cast<CUstream>(CUDAThreadEntry::ThreadLocal()->stream);
    // cudaStream_t strm_;
    // cudaStreamCreate(&strm_);
    // CUstream strm = static_cast<CUstream>(strm_);
    ThreadWorkLoad wl = launch_param_config_.Extract(args);

    // clock_t tic = clock();
    CUresult result = cuLaunchKernel(fcache_[device_id], wl.grid_dim(0), wl.grid_dim(1),
                                     wl.grid_dim(2), wl.block_dim(0), wl.block_dim(1),
				     wl.block_dim(2), wl.dyn_shmem_size, m_->my_stream, void_args, nullptr);

    if (id == count-1) {
    	cuStreamSynchronize(m_->my_stream);
    }
    // cudaStreamDestroy(strm_);
    // clock_t toc = clock();
    // double s = (double)(toc - tic) / CLOCKS_PER_SEC;

    // std::cout << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
    //      << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2) << std::endl;

    // printf("end main task\n");


    // std::istringstream is2( readFileIntoString("/mnt/tvm-getstarted/last_tune_size.txt") );
    // int N, L, M;
    // is2 >> N >> L >> M;

    // char fname[100] = {0};
    // sprintf(fname, "%dx%dx%d.csv", N, L, M);
    // string fname_(fname);

    // std::ofstream outfile;
    // outfile.open(fname_, std::ios_base::app);
    // int max_sm = 0;
    // outfile << max_sm << "," << 1000*s << ","
    //      << "grid=(" << wl.grid_dim(0) << "x" << wl.grid_dim(1) << "x" << wl.grid_dim(2) << ")" << ","
    //      << "block=(" << wl.block_dim(0) << "x" << wl.block_dim(1) << "x" << wl.block_dim(2) << ")"
    //      << std::endl;


    // exit(0);
    // printf("wait thread join\n");
    // t.join();

    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) {
      const char* msg;
      cuGetErrorName(result, &msg);
      std::ostringstream os;
      os << "CUDALaunch Error: " << msg << "\n"
         << " grid=(" << wl.grid_dim(0) << "," << wl.grid_dim(1) << "," << wl.grid_dim(2) << "), "
         << " block=(" << wl.block_dim(0) << "," << wl.block_dim(1) << "," << wl.block_dim(2)
         << ")\n";
      std::string cuda = m_->GetSource("");
      if (cuda.length() != 0) {
        os << "// func_name=" << func_name_ << "\n"
           << "// CUDA Source\n"
           << "// -----------\n"
           << cuda;
      }
      LOG(FATAL) << os.str();
    }

  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // The name of the function.
  std::string func_name_;
  // Device function cache per device.
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUfunction, kMaxNumGPUs> fcache_;
  // launch parameters configuration
  LaunchParamConfig launch_param_config_;
};

int CUDAWrappedFunc::is_first_operator_shared = true;
int CUDAWrappedFunc::count = 0;

class CUDAPrepGlobalBarrier {
 public:
  CUDAPrepGlobalBarrier(CUDAModuleNode* m, ObjectPtr<Object> sptr) : m_(m), sptr_(sptr) {
    std::fill(pcache_.begin(), pcache_.end(), 0);
  }

  void operator()(const TVMArgs& args, TVMRetValue* rv) const {
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    if (pcache_[device_id] == 0) {
      pcache_[device_id] =
          m_->GetGlobal(device_id, runtime::symbol::tvm_global_barrier_state, sizeof(unsigned));
    }
    CUDA_DRIVER_CALL(cuMemsetD32(pcache_[device_id], 0, 1));
  }

 private:
  // internal module
  CUDAModuleNode* m_;
  // the resource holder
  ObjectPtr<Object> sptr_;
  // mark as mutable, to enable lazy initialization
  mutable std::array<CUdeviceptr, kMaxNumGPUs> pcache_;
};

PackedFunc CUDAModuleNode::GetFunction(const std::string& name,
                                       const ObjectPtr<Object>& sptr_to_self) {
  ICHECK_EQ(sptr_to_self.get(), this);
  ICHECK_NE(name, symbol::tvm_module_main) << "Device function do not have main";
  if (name == symbol::tvm_prepare_global_barrier) {
    return PackedFunc(CUDAPrepGlobalBarrier(this, sptr_to_self));
  }
  auto it = fmap_.find(name);
  if (it == fmap_.end()) return PackedFunc();
  const FunctionInfo& info = it->second;
  CUDAWrappedFunc f;
  f.Init(this, sptr_to_self, name, info.arg_types.size(), info.launch_param_tags);
  return PackFuncVoidAddr(f, info.arg_types);
}

Module CUDAModuleCreate(std::string data, std::string fmt,
                        std::unordered_map<std::string, FunctionInfo> fmap,
                        std::string cuda_source) {
  auto n = make_object<CUDAModuleNode>(data, fmt, fmap, cuda_source);
  return Module(n);
}

// Load module from module.
Module CUDAModuleLoadFile(const std::string& file_name, const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  LoadBinaryFromFile(file_name, &data);
  LoadMetaDataFromFile(meta_file, &fmap);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}

Module CUDAModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
  return CUDAModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("runtime.module.loadfile_cubin").set_body_typed(CUDAModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadfile_ptx").set_body_typed(CUDAModuleLoadFile);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cuda").set_body_typed(CUDAModuleLoadBinary);
}  // namespace runtime
}  // namespace tvm

import tensorrt as trt 
import os
import torch
import cv2
import numpy as np
from cuda import cudart
from collections import OrderedDict 
from pathlib import Path



input_name = "input"
output_name = "output"
project_path = os.getenv("PROJECT_PATH")
print("PROJECT_PATH:", project_path)
if project_path is None:
    print("Please set the PROJECT_PATH environment variable to the root directory of the project.")
    exit(1)

middle_path = project_path + "/example/base/middle_files/"
logger = trt.Logger(trt.Logger.ERROR)  
trt_file =  Path(middle_path + 'model.engine')
if trt_file.exists():                                                       # Load engine from file and skip building process if it existed
      with open(trt_file, "rb") as f:
          engine_bytes = f.read()
      if engine_bytes == None:
          print("Fail getting serialized engine")
          exit(1)
      print("Succeed getting serialized engine")

engine = trt.Runtime(logger).deserialize_cuda_engine(engine_bytes)  
context = engine.create_execution_context() 

input_img = cv2.imread(middle_path + 'face.png').astype(np.float32) 
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)
tensor_name_list = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

buffer = OrderedDict()                                                      # Prepare the memory buffer on host and device
for name in tensor_name_list:
    mode = engine.get_tensor_mode(name)
    data_type = engine.get_tensor_dtype(name)
    buildtime_shape = engine.get_tensor_shape(name)
    runtime_shape = context.get_tensor_shape(name)
    n_byte = trt.volume(runtime_shape) * np.dtype(trt.nptype(data_type)).itemsize
    host_buffer = np.empty(runtime_shape, dtype=trt.nptype(data_type))
    device_buffer = cudart.cudaMalloc(n_byte)[1]
    buffer[name] = [host_buffer, device_buffer, n_byte]
    print(f"{'Input ' if mode == trt.TensorIOMode.INPUT else 'Output'}->{data_type}, {buildtime_shape}, {runtime_shape}, {name}")
for name in tensor_name_list:
    context.set_tensor_address(name, buffer[name][1])  
buffer[input_name][0] = np.ascontiguousarray(input_img) 
for name in tensor_name_list:                                               # Copy input data from host to device
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        cudart.cudaMemcpy(buffer[name][1], buffer[name][0].ctypes.data, buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

context.execute_async_v3(0)     
for name in tensor_name_list:                                               # Copy output data from device to host
    if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        cudart.cudaMemcpy(buffer[name][0].ctypes.data, buffer[name][1], buffer[name][2], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
for _, device_buffer, _ in buffer.values():                                 # Free the GPU memory buffer after all work
        cudart.cudaFree(device_buffer)
output = buffer[output_name][0]
output = np.squeeze(output, 0)
output = np.clip(output, 0, 255)
output = np.transpose(output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite(middle_path + "face_trt.png", output)
print("output saved to "+ middle_path + "face_trt.png") 
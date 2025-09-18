import torch
import onnx
import tensorrt as trt
import os
project_path = os.getenv("PROJECT_PATH")
print("PROJECT_PATH:", project_path)
if project_path is None:
    print("Please set the PROJECT_PATH environment variable to the root directory of the project.")
    exit(1)

middle_path = project_path + "/example/base/middle_files/"
onnx_model = onnx.load(middle_path + "srcnn.onnx")
device = torch.device('cuda:0')
# create builder and network
logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(logger)
EXPLICIT_BATCH = 1 << (int)(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(EXPLICIT_BATCH)

# parse onnx
parser = trt.OnnxParser(network, logger)

if not parser.parse(onnx_model.SerializeToString()):
    error_msgs = ''
    for error in range(parser.num_errors):
        error_msgs += f'{parser.get_error(error)}\n'
    raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)
profile = builder.create_optimization_profile()

profile.set_shape('input', [1, 3, 256, 256],[1, 3, 256, 256],[1, 3, 256, 256])
config.add_optimization_profile(profile)
# create engine
with torch.cuda.device(device):
    serialized_engine = builder.build_serialized_network(network, config)

with open(middle_path + 'model.engine', mode='wb') as f:
    f.write(serialized_engine)
    print("generating file done!")
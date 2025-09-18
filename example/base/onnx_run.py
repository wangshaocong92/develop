import onnxruntime
import cv2
import numpy as np
import os 
import subprocess
def debug_env_var(var_name: str):
    """深度调试环境变量读取问题"""
    print(f"\n=== 调试 {var_name} 环境变量 ===")
    
    # 1. 检查Python进程获取情况
    py_value = os.getenv(var_name)
    print(f"Python获取: {py_value if py_value else '<空值>'}")

    # 2. 检查系统shell环境
    try:
        shell_cmd = f"echo ${var_name}" if not os.name == 'nt' else f"echo %{var_name}%"
        shell_value = subprocess.check_output(shell_cmd, shell=True).decode().strip()
        print(f"系统shell获取: {shell_value if shell_value else '<空值>'}")
    except Exception as e:
        print(f"Shell检查失败: {str(e)}")

    # 3. 检查环境变量来源
    print("\n可能的原因：")
    if py_value != shell_value:
        print("- 环境变量在Python父进程之外设置（如shell配置文件）")
        print("- Python进程启动后环境变量被清除")
    else:
        print("- 环境变量未正确导出（尝试在终端执行：export {var_name}=\"您的路径\"）")

    # 4. 解决方案
    print("\n解决方案：")
    print("1. 在启动Python前执行：source ~/.bashrc (或对应shell配置文件)")
    print("2. 使用绝对路径直接设置：os.environ['{var_name}'] = \"您的路径\"")
    print("3. 通过启动脚本统一加载环境变量")

debug_env_var("PROJECT_PATH")

project_path = os.getenv("PROJECT_PATH")
print("PROJECT_PATH:", project_path)
if project_path is None:
    print("Please set the PROJECT_PATH environment variable to the root directory of the project.")
    exit(1)

middle_path = project_path + "/example/base/middle_files/"

input_img = cv2.imread(middle_path + 'face.png').astype(np.float32) 
input_img = np.transpose(input_img, [2, 0, 1])
input_img = np.expand_dims(input_img, 0)


print(input_img.shape)

ort_session = onnxruntime.InferenceSession(middle_path + "srcnn.onnx")
ort_inputs = {'input': input_img}
ort_output = ort_session.run(['output'], ort_inputs)[0]

ort_output = np.squeeze(ort_output, 0)
ort_output = np.clip(ort_output, 0, 255)
ort_output = np.transpose(ort_output, [1, 2, 0]).astype(np.uint8)
cv2.imwrite(middle_path + "face_ort.png", ort_output)
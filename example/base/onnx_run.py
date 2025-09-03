import onnxruntime
import cv2
import numpy as np

middle_path = "./example/base/middle_files/"

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
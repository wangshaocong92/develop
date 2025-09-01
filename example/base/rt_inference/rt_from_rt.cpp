#include <cstdlib>
#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

using namespace nvinfer1;

const char* IN_NAME = "input";
const char* OUT_NAME = "output";
static const int IN_H = 256;
static const int IN_W = 256;
static const int BATCH_SIZE = 1;
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
        const ICudaEngine& engine = context.getEngine();

        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers.
        assert(engine.getNbIOTensors() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // Note that indices are guaranteed to be less than IEngine::getNbBindings()
        int inputIndex = 0;
        int outputIndex = 0;
        for(auto i = 0 ; i <  engine.getNbIOTensors();i ++)
        {
            engine.getIOTensorName(i) == IN_NAME ? inputIndex = i : outputIndex = i;

        }
        auto shape = engine.getTensorShape(IN_NAME);
        std::cout << "input tensor shape: ";
        for (int i = 0; i < shape.nbDims; i++)
        {
            std::cout << shape.d[i] << " ";
        }
        std::cout << std::endl;
        shape = engine.getTensorShape(OUT_NAME);
        std::cout << "output tensor shape: ";
        for (int i = 0; i < shape.nbDims; i++)
        {
            std::cout << shape.d[i] << " ";
        }
        std::cout << std::endl;
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * 768 * 768 * sizeof(float)));


        context.setTensorAddress(IN_NAME, buffers[0]);
        context.setTensorAddress(OUT_NAME, buffers[1]);

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueueV3(stream);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * 768 * 768 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));

        std::cout << "inference done!" << std::endl;
}

cv::Mat nchwToImage(const float* nchw_data, int batch_size, int channels, int height, int width, int batch_idx = 0) {
    // 参数校验
    if (batch_idx >= batch_size) {
        throw std::runtime_error("Batch index out of range");
    }

    // 1. 提取指定batch的数据
    const float* batch_data = nchw_data + batch_idx * channels * height * width;

    // 2. 创建各通道Mat
    std::vector<cv::Mat> channel_mats(channels);
    for (int c = 0; c < channels; ++c) {
        channel_mats[c] = cv::Mat(height, width, CV_32FC1, 
                                 const_cast<float*>(batch_data + c * height * width));
    }

    // 3. 合并通道(CHW -> HWC)
    cv::Mat merged;
    cv::merge(channel_mats, merged);
    
    // 4. 转换回uint8并调整值域[0,255]
    // cv::Mat output;
    // merged.convertTo(output, CV_8UC3, 255.0);
// 
    // std::cout << "output Matrix:\n" << output << std::endl;

    return merged;
}


int main(int argc, char** argv)
{
        // create a model using the API directly and serialize it to a stream
        char *trtModelStream{ nullptr };
        size_t size{ 0 };

        std::ifstream file("/data/develop/example/base/middle_files/model.engine", std::ios::binary);
        if (file.good()) {
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                trtModelStream = new char[size];
                assert(trtModelStream);
                file.read(trtModelStream, size);
                file.close();
        }

        Logger m_logger;
        std::unique_ptr<IRuntime> runtime(createInferRuntime(m_logger));
        assert(runtime != nullptr);
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(trtModelStream, size));
        assert(engine != nullptr);
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        assert(context != nullptr);

        auto image =
            cv::imread("/data/develop/example/base/middle_files/face.png",
                       cv::IMREAD_COLOR);
        image.convertTo(image, CV_32FC3);
        std::vector<cv::Mat> channels;
        cv::split(image, channels); // 分离BGR通道

        // 3. 创建NCHW格式输出 (1x3xHxW)
        int height = image.rows;
        int width = image.cols;
        std::vector<float> nchw_data(1 * 3 * height * width);

        // 4. 填充数据到连续内存
        for (int c = 0; c < 3; ++c) {
          memcpy(&nchw_data[c * height * width], channels[c].data,
                 height * width * sizeof(float));
        }

        // generate input data
        float data[BATCH_SIZE * 3 * IN_H * IN_W];
        for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++) {
           data[i] = (i < nchw_data.size()) ? nchw_data[i] : 1.0f; 
        }
       
        // Run inference
        float prob[BATCH_SIZE * 3 * 768 * 768];
        doInference(*context, data, prob, BATCH_SIZE);
        nchwToImage(prob, BATCH_SIZE, 3, 768, 768,0).copyTo(image);

        cv::imwrite("/data/develop/example/base/middle_files/face_out0.png", image);
        std::cout << "output saved to /data/develop/example/base/middle_files/face_out.png" << std::endl;

        return 0;
}
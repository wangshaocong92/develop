#include <fstream>
#include <iostream>

#include <NvInfer.h>
#include <memory>
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
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // Release stream and buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));

        std::cout << "inference done!" << std::endl;
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

        // generate input data
        float data[BATCH_SIZE * 3 * IN_H * IN_W];
        for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++)
                data[i] = 1;

        // Run inference
        float prob[BATCH_SIZE * 3 * IN_H * IN_W /4];
        doInference(*context, data, prob, BATCH_SIZE);
        return 0;
}
#pragma once 

#include <cmath>


#include <limits>
#include <vector>
namespace kernel {
namespace cpu
{
    void softmax_forward(float* input, const int seqlen)
    {
        /// 求最大值
        float max_val = std::numeric_limits<float>::min();
        for(auto  i = 0 ; i < seqlen; i++)
        {
            max_val = std::max(max_val, input[i]);
        }
        /// 求和
        double sum = 0.0;
        for(auto  i = 0 ; i < seqlen; i++)
        {
            sum += std::exp(input[i] - max_val);
        }
        /// 求 softmax
        for(auto  i = 0 ; i < seqlen; i++)
        {
            input[i] = std::exp(input[i] - max_val) / sum; 
        }
        return;
    }


    void online_softmax_forward(float* input, const int seqlen)
    {
        /// 定义步长
        const int step = 32;
        auto steps = seqlen / step + (seqlen % step != 0 ? 1 : 0);
        std::vector<float> max_vals(steps, std::numeric_limits<float>::min());
        std::vector<double> sum_vals(steps, 0.0);
        int current_step = 0;
        float max_val = std::numeric_limits<float>::min();
        for(auto  i = 0 ; i < seqlen; i+=step)
        {
            /// 求最大值
            for(auto j = i; j < std::min(i + step, seqlen); j++)
            {
                max_vals[current_step] = std::max(max_vals[current_step], input[j]);
                max_val = std::max(max_val, input[j]);
            }
            /// 求和
            for(auto j = i; j < std::min(i + step, seqlen); j++)
            {
                sum_vals[current_step] += std::exp(input[j] - max_vals[current_step]);
            }
            current_step++;
        }

        ///求和
        double sum = 0.0;
        for(auto i = 0; i < steps; i++)
        {
            sum_vals[i] *= std::exp(max_vals[i] - max_val);
            sum += sum_vals[i];
        }
        for(auto  i = 0 ; i < seqlen; i++)
        {
            input[i] = std::exp(input[i] - max_val) / sum; 
        }
        return;
    }


}
}
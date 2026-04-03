/*
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 

示例 1：
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
示例 2：

输入：height = [4,2,0,3,2,5]
输出：9
 
提示：

n == height.length
1 <= n <= 2 * 104
0 <= height[i] <= 105
 

*/


#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
 int calc(vector<int>& height, int index1, int index2) {
   auto h = height[index1] > height[index2] ? height[index2] : height[index1];
   auto total = 0;
   for (auto i = index1 + 1; i < index2; i++) { total += h - height[i]; }
   return total;
 }
    int trap(vector<int>& height) {
      /// 获取整个height 的最大值，然后从两边开始往中间处理
      /// 最大值可能有多个我们使用一个vec 来记录他们的地址
      int total = 0;
      int max = INT16_MIN;
      for (auto i = 0; i < height.size(); i++) {
        if (height[i] > max) { max = height[i]; }
      }

      std::vector<int> max_indexs;
      for (auto i = 0; i < height.size(); i++) {
        if (height[i] == max) { max_indexs.push_back(i); }
      }
      //// 先处理max 之间的，在处理他们之外的
      for (auto i = 0; i < max_indexs.size(); i++) {
        if (i + 1 < max_indexs.size()) total += calc(height, max_indexs[i], max_indexs[i + 1]);
      }


      auto end = max_indexs.front();
      auto rend = max_indexs.back();

      for (auto i = 0; i < end;) {
        if (height[i] == 0 || i + 1 == end) {
          i++;
          continue;
        }

        bool exist = false;
        for (auto j = i + 1; j < end + 1; j++) {
          if (height[i] <= height[j]) {
            total += calc(height, i, j);
            if (j == end) {
              exist = true;
            } else {
              i = j;
            }

            break;
          }
        }

        if (exist) { break; }
      }

      for (int i = height.size() - 1; i > rend;) {
        if (height[i] == 0 || i - 1 == rend) {
          i--;
          continue;
        }
        bool exist = false;
        for (int j = i - 1; j > rend - 1; j--) {
          if (height[i] <= height[j]) {
            total += calc(height, j, i);
            if (j == rend) {
              exist = true;
            } else {
              i = j;
            }

            break;
          }
        }

        if (exist) { break; }
      }

      return total;
    }
};


int main() {
  Solution s;
  vector<int> height{4, 2, 3};
  std::cout << s.trap(height) << std::endl;
}

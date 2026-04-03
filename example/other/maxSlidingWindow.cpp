/*
239. 滑动窗口最大值
困难

提示
给你一个整数数组 nums，有一个大小为 k
的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k
个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。



示例 1：

输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
输出：[3,3,5,5,6,7]
解释：
滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
示例 2：

输入：nums = [1], k = 1
输出：[1]


提示：

1 <= nums.length <= 105
-104 <= nums[i] <= 104
1 <= k <= nums.length


*/

#include <climits>
#include <iostream>
#include <vector>

using namespace std;
class Solution {
 public:
  vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    auto index = 0;
    int max = INT_MIN;

    std::vector<int> ret;
    ret.reserve(nums.size() - k);
    for (auto i = 0; i < nums.size() - k + 1; i++) {
      if (i == 0 || index < i) {
        max = INT_MIN;
        index = 0;
        for (auto j = i; j < i + k; j++) {
          if (nums[j] >= max) {
            max = nums[j];
            index = j;
          }
        }
        ret.push_back(max);
        continue;
      }
      if (nums[i + k - 1] >= max) {
        max = nums[i + k - 1];
        index = i + k - 1;
        ret.push_back(max);
      } else {
        ret.push_back(max);
      }
    }

    return ret;
  }
};


int main() {
  Solution s;

  vector<int> value{1, 3, -1, -3, 5, 3, 6, 7};
  auto ret = s.maxSlidingWindow(value, 3);
  for (auto i : ret) { cout << i << endl; }
}
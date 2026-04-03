/*

560. 和为 K 的子数组
中等
相关标签
premium lock icon
相关企业
提示
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。

子数组是数组中元素的连续非空序列。



示例 1：

输入：nums = [1,1,1], k = 2
输出：2
示例 2：

输入：nums = [1,2,3], k = 3
输出：2


提示：

1 <= nums.length <= 2 * 104
-1000 <= nums[i] <= 1000
-107 <= k <= 107
*/

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;
class Solution {
 public:
  int subarraySum(vector<int>& nums, int k) {
    int sub_array_count = 0;
    for (auto i = 0; i < nums.size(); i++) {
      if (nums[i] == k) { sub_array_count++; }
      auto total = nums[i];
      for (auto j = i + 1; j < nums.size(); j++) {
        total += nums[j];
        if (total == k) { sub_array_count++; }
      }
    }
    return sub_array_count;
  }
};


int main() {
  Solution s;
  std::vector<int> nums{28, 54, 7, -70, 22, 65, -6};
  std::cout << s.subarraySum(nums, 100);

  return 0;
}
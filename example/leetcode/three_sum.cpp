/*

给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k
且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。





示例 1：

输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
解释：
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意，输出的顺序和三元组的顺序并不重要。
示例 2：

输入：nums = [0,1,1]
输出：[]
解释：唯一可能的三元组和不为 0 。
示例 3：

输入：nums = [0,0,0]
输出：[[0,0,0]]
解释：唯一可能的三元组和为 0 。

提示：

3 <= nums.length <= 3000
-105 <= nums[i] <= 105
*/

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
using namespace std;
class Solution {
 public:
  vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    std::map<int,vector<int>> si;
    for (auto i = 0; i < nums.size(); i++) {
      if (nums[i] >= 0) {
        si[nums[i]].push_back(i);
      }
    }
    for (auto i = 0; i < nums.size() - 2; i++) {
      if (nums[i] > 0) continue;  /// 最小的大于0，那结果肯定不是零
      if (i > 0 && nums[i] == nums[i - 1]) continue;
      for (auto j = i + 1; j < nums.size() - 1; j++) {
        if (nums[j] + nums[i] > 0) continue;  /// 最小的大于0，那结果肯定不是零
        if (j > i + 1 && nums[j] == nums[j - 1]) continue;
        int target = -(nums[i] + nums[j]);
        if (nums[j] > target) break;
        if (target > nums.back()) continue;
        auto iter =si.find(target);
        if (iter != si.end()) 
        {
            for(auto k : iter->second)
            {
                if(k > j)
                {
                    result.push_back({nums[i], nums[j], iter->first}); 
                }
            }
          
        }
       
      }
    }
    return result;
  }
};


int main() {
  // -4 -1 -1 0 1 2
  vector<int> nums = {1,1,-2};
  Solution s;
  auto result = s.threeSum(nums);
  for (auto& item : result) {
    for (auto& num : item) { printf("%d ", num); }
    printf("\n");
  }
}

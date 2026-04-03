/*
438. 找到字符串中所有字母异位词
中等
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词
的子串，返回这些子串的起始索引。不考虑答案输出的顺序。



示例 1:

输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
 示例 2:

输入: s = "abab", p = "ab"
输出: [0,1,2]
解释:
起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。


提示:

1 <= s.length, p.length <= 3 * 104
s 和 p 仅包含小写字母


*/

#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;

class Solution {
 public:
  vector<int> findAnagrams(string s, string p) {
    /// 先把子串抽析出来
    std::map<char, int> pmap;
    for (auto i = p.begin(); i != p.end(); i++) {
      if (pmap.find(*i) == pmap.end()) {
        pmap[*i] = 1;
      } else {
        pmap[*i]++;
      }
    }

    std::vector<int> ret;
    std::map<char, int> s_sub_map;
    std::vector<char> sub_string;
    int index = -1;
    int sindex = -1;
    for (auto i = s.begin(); i != s.end(); i++) {
      sindex++;
      if (pmap.find(*i) == pmap.end()) {
        s_sub_map.clear();
        sub_string.clear();
        index = -1;
      } else {
        if (index == -1) { index = sindex; }
        if (s_sub_map.find(*i) == pmap.end()) {
          s_sub_map[*i] = 1;
        } else {
          s_sub_map[*i]++;
        }
        sub_string.push_back(*i);
        if (s_sub_map == pmap) {
          ret.push_back(index);
          index++;
          auto b = sub_string.begin();
          s_sub_map[*b]--;
          sub_string.erase(sub_string.begin());
        } else {
          if (s_sub_map[*i] <= pmap[*i])
            continue;
          else {
            /// 去掉一个i，连带i之前的全部去掉
            auto iter = sub_string.begin();
            for (; iter != sub_string.end(); iter++) {
              s_sub_map[*iter]--;
              index++;
              if (*iter == *i) { break; }
            }
            std::vector<char> tmp_sub(iter + 1, sub_string.end());
            sub_string = tmp_sub;
          }
        }
      }
    }


    return ret;
  }
};


int main() {
  Solution s;

  for (auto i : s.findAnagrams("abab", "ab")) cout << i << endl;
}
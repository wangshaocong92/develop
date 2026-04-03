/*
给定两个字符串 s 和 t，长度分别是 m 和 n，返回 s 中的 最短窗口 子串，使得该子串包含 t
中的每一个字符（包括重复字符）。如果没有这样的子串，返回空字符串 ""。

测试用例保证答案唯一。



示例 1：

输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"
解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
示例 2：

输入：s = "a", t = "a"
输出："a"
解释：整个字符串 s 是最小覆盖子串。
示例 3:

输入: s = "a", t = "aa"
输出: ""
解释: t 中两个字符 'a' 均应包含在 s 的子串中，
因此没有符合条件的子字符串，返回空字符串。


提示：

m == s.length
n == t.length
1 <= m, n <= 105
s 和 t 由英文字母组成


进阶：你能设计一个在 O(m + n) 时间内解决此问题的算法吗？

*/


#include <climits>
#include <iostream>
#include <map>
#include <string>
#include <vector>


using namespace std;
class Solution {
 public:
  static bool sbiggert(const std::map<char, int>& tmap, const std::map<char, int>& smap) {
    for (auto i : tmap) {
      auto iter = smap.find(i.first);
      if (iter == smap.end()) return false;
      if (iter->second < i.second) return false;
    }
    return true;
  }
  string minWindow(string s, string t) {
    std::map<char, int> tmap;
    for (auto i = t.begin(); i != t.end(); i++) {
      if (tmap.find(*i) == tmap.end())
        tmap[*i] = 1;
      else
        tmap[*i]++;
    }
    int length = INT_MAX;
    string ret = "";
    auto forword = [&tmap, &s](std::map<char, int> s_submap, decltype(s.begin()) iter) -> decltype(s.begin()) {
      for (; iter != s.end(); iter++) {
        if (s_submap.find(*iter) == s_submap.end()) {
          s_submap[*iter] = 1;
        } else {
          s_submap[*iter]++;
        }

        // cout << "forword: " << *iter <<endl;
        if (sbiggert(tmap, s_submap)) { return iter; }
      }
      return s.end();
    };

    auto backword = [&tmap, &length, &ret,&s](std::map<char, int>& s_submap,
                                               decltype(s.begin()) & siter,
                                               decltype(s.begin()) eiter)  {
      int len = 0;
      auto tmp_eiter = eiter;
      s_submap.clear();
      while (true) {
        if (s_submap.find(*tmp_eiter) == s_submap.end()) {
          s_submap[*tmp_eiter] = 1;
        } else {
          s_submap[*tmp_eiter]++;
        }

        // cout << "backword: " << *tmp_eiter <<endl;
        len++;
        if (sbiggert(tmap, s_submap)) {
          s_submap[*tmp_eiter]--;
          if (len < length) {
            length = len;
            ret = string(tmp_eiter, eiter + 1);
          }

          while (true) {
            if(s_submap.find(*(eiter + 1)) == s_submap.end())
            {
                eiter++;
                if(eiter == s.end())
                    break;
            }else {
                break;
            }
          }
          siter = eiter;
          return;
        }

        if (siter == eiter) break;
        tmp_eiter--;
      }

      /// 正常来说在while内部肯定是return，若没有肯定有问题
    };

    auto siter = s.begin();
    auto siter2 = s.begin();
    std::map<char, int> s_submap;
    while (true) {
      auto iter = forword(s_submap, siter);
      if (iter == s.end()) return ret;
      backword(s_submap, siter2, iter);
      siter = iter + 1;
      // cout << "while: " << *siter << endl;
    }
    return ret;
  }
};


int main() {
  Solution s;
  cout << s.minWindow("bdab", "ab") << endl;
  return 0;
}
cmake_minimum_required(VERSION 3.18)
project(leetcode)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0") 

add_executable(three_sum three_sum.cpp)
install(TARGETS three_sum DESTINATION bin)

add_executable(trap trap.cpp)
install(TARGETS trap DESTINATION bin)

add_executable(lengthOfLongestSubstring lengthOfLongestSubstring.cpp)
install(TARGETS lengthOfLongestSubstring DESTINATION bin)


add_executable(findAnagrams findAnagrams.cpp)
install(TARGETS findAnagrams DESTINATION bin)

add_executable(subarraySum subarraySum.cpp)
install(TARGETS subarraySum DESTINATION bin)


add_executable(maxSlidingWindow maxSlidingWindow.cpp)
install(TARGETS maxSlidingWindow DESTINATION bin)


add_executable(minWindow minWindow.cpp)
install(TARGETS minWindow DESTINATION bin)
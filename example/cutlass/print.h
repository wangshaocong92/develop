#pragma once


#include <cute/util/print.hpp>

template<class T>
void printn(const T &t)
{
    cute::print(t);cute::print("\n");
}
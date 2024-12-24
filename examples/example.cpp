#include "cheburaxa.hpp"
#include <chrono>
#include <iostream>

int main()
{
    constexpr std::size_t n = 256;
    constexpr std::size_t r = 8;
    std::vector<std::size_t> U_subsets(n * (r + 1));
    std::vector<std::size_t> V_subsets(n * (r + 1));
    std::vector<double> A(n * n, 0.0);
    std::vector<double> U(n * r);
    std::vector<double> V(n * r);
    for (std::size_t i = 0; i < n; ++i) A[i * (n + 1)] = 1.0;
    double min_err = std::numeric_limits<double>::infinity();
    auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < 20; ++i)
    {
        auto info = cheburaxa::approximate_matrix<true>(n, n, r, A.data(), U.data(), U_subsets.data(), V.data(), V_subsets.data());
        min_err = std::min(min_err, info.back().approximation_error);
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration<double>(end - start).count() << ", minimal value = " << min_err << std::endl;
    return 0;
}
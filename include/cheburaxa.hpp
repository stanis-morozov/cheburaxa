#ifndef CHEBURAXA_HPP
#define CHEBURAXA_HPP

#include "cheburaxa_linalg.hpp"
#include <memory>
#include <random>
#include <thread>
#include <atomic>
#include <mutex>

namespace cheburaxa
{
    // compute equidistant solution
    // It is equal to R^{-1} Q[:, :r]^T (a  - (q, a) / \|q\|_1 sgn(q))
    // where q is the last column of Q
    // Q is an Fortran-ordered (r + 1) x (r + 1) unitary matrix
    // L is an Fortran-ordered r x r lower triangular matrix
    // a is a vector of size r + 1
    // sol is a solution vector of size r
    // y is a work array of size r + 1
    template <class T>
        T remez_equidistant(std::size_t r, const T *Q, const T *L, const T *a, T *sol, T *y)
        {
            const T *q = Q + r * (r + 1);
            T w_norm = linalg::dot_product(r + 1, q, a) / linalg::norm_1(r + 1, q);

            for (std::size_t i = 0; i < r + 1; i++) {
                y[i] = a[i] - w_norm * std::copysign(1.0, q[i]);
            }

            linalg::transposed_matvec(r + 1, r, Q, y, sol);
            linalg::solve_lower_triangular_transposed(r, L, sol);
            return std::abs(w_norm);
        }

    // Computes maximal value on subset of vector elements
    template <class T>
        T max_abs_on_subset(std::size_t size, const std::size_t *subset, const T *x)
        {
            T max = 0.0;
            for (std::size_t i = 0; i < size; ++i) {
                max = std::max(max, std::abs(x[subset[i]]));
            }
            return max;
        }

    struct Info
    {
        std::size_t err_info = 0;
        std::size_t iter_num = 0;
        double approximation_error = 0;

        Info(std::size_t _err_info, std::size_t _iter_num, double _approximation_error) {err_info = _err_info; iter_num = _iter_num; approximation_error = _approximation_error;}
        Info() = default;
    };

    // Solves problem sol = argmin_x \|V x - a\|_C
    template <bool row_major = false, class T = double>
        Info approximate_vector(std::size_t n, std::size_t r, const T *V, const T *a, std::size_t stride_a, T *sol, std::size_t stride_sol, std::size_t *subset)
        {
            std::unique_ptr<T[]> mem(new T[n + (r + 1) * (2 * r + 6) - 1]());
            T *curr_Q = mem.get();
            T *curr_L = curr_Q + (r + 1) * (r + 1);
            T *curr_a = curr_L + (r + 1) * r;
            T *w = curr_a + r + 1;
            T *z = w + r + 1;
            T *f = z + r + 1;
            T *curr_sol = f + r + 1;
            T *full_residual = curr_sol + r;
            const T *q = curr_Q + r * (r + 1);   

            for (std::size_t i = 0; i < r + 1; i++) {
                curr_a[i] = a[subset[i] * stride_a];
            }
            

            linalg::copy_rows<row_major>(n, r, r + 1, subset, V, curr_L);
            linalg::qr(r + 1, r, curr_L, curr_Q, w);

            T prev_mx_val_subset = -1;
            for (std::size_t niter = 0; true; ++niter) {

                T curr_norm = remez_equidistant(r, curr_Q, curr_L, curr_a, curr_sol, z);

                linalg::compute_residual<row_major>(n, r, a, stride_a, V, curr_sol, full_residual);

                std::size_t j = linalg::idx_norm_inf(n, full_residual);

                

                T mx_val_subset = max_abs_on_subset(r + 1, subset, full_residual);

                if (prev_mx_val_subset >= mx_val_subset) {
                    for (std::size_t i = 0; i < r; ++i) sol[i * stride_sol] = curr_sol[i];
                    return Info(std::size_t(0), niter, std::abs(full_residual[j]));
                }
                prev_mx_val_subset = mx_val_subset;

                linalg::copy_row<row_major>(n, r, j, V, w);
                linalg::solve_lower_triangular(r, curr_L, w);
                linalg::matvec(r + 1, r, curr_Q, w, z);

                T q_dot_a = linalg::dot_product(r + 1, q, curr_a);
                T z_dot_a = linalg::dot_product(r + 1, z, curr_a);

                // f = q_k e_k + z_k q - q_k z (see cycle below)
                // (f, a) = q_k a_j + z_k (q, a) - q_k (z, a)
                // No that z_k (q, a) - q_k (z, a) is independent of k-th value of a
                // So we can compute all k + 1 dot products (f, a) by the formula
                // (a_j - (z, a)) q + (q, a) z
                // And q and z are still orthogonal, so that should be pretty robust
                linalg::alpha_x_plus_beta_y(r + 1, a[j * stride_a] - z_dot_a, q, q_dot_a, z, w);

                for (std::size_t k = 0; k < r + 1; k++) {
                    linalg::alpha_x_plus_beta_y(r + 1, z[k], q, -q[k], z, f);
                    f[k] = q[k];

                    w[k] = std::abs(w[k]) / linalg::norm_1(r + 1, f);
                }

                std::size_t replace_idx = std::max_element(w, w + r + 1) - w;

                if (w[replace_idx] <= curr_norm) {
                    for (std::size_t i = 0; i < r; ++i) sol[i * stride_sol] = curr_sol[i];
                    return Info(std::size_t(1), niter + 1, std::abs(full_residual[j]));
                }

                linalg::copy_row<false>(r + 1, r + 1, replace_idx, curr_Q, z);
                linalg::row_diff<row_major>(n, r, j, subset[replace_idx], V, w);
                subset[replace_idx] = j;
                curr_a[replace_idx] = a[j * stride_a];

                linalg::rank1_qr_update(r + 1, r, curr_Q, curr_L, z, w);
            }
            return Info(std::size_t(0), std::size_t(0), std::abs(full_residual[0]));
        }

        void random_subsets(std::size_t num, std::size_t max, std::size_t size, std::size_t *subset, std::size_t seed = 0)
        {
            if (seed == 0)
            {
                std::random_device rd;
                seed = rd();
            }
            std::mt19937 g(seed);
            

            std::unique_ptr<std::size_t []> tmp(new std::size_t[max]);

            for (std::size_t j = 0; j < num; ++j)
            {
                std::iota(tmp.get(), tmp.get() + max, 0);
                for (std::size_t i = 0; i < size; ++i)
                {
                    std::uniform_int_distribution<std::size_t> d(i, max - 1);
                    std::size_t k = d(g);
                    std::swap(tmp[i], tmp[k]);
                }
                std::copy(tmp.get(), tmp.get() + size, subset + size * j);
            }
        }

        template <bool transpose_A = false, bool row_major = false, class T = double>
        Info approximate_matrix(std::size_t m, std::size_t n, std::size_t r, const T *A, const T *U, T *V, std::size_t *V_subsets, std::size_t num_threads = std::thread::hardware_concurrency())
        {
            std::size_t size = transpose_A ? n : m;
            std::size_t num = transpose_A ? m : n;
            std::size_t row_stride_a = row_major ? 1 : m;
            std::size_t column_stride_a = row_major ? n : 1;
            std::size_t shift_a = transpose_A ? column_stride_a : row_stride_a;
            std::size_t stride_a = transpose_A ? row_stride_a : column_stride_a;
            std::size_t stride_V = transpose_A ? m : n;
            std::size_t err_sum(0);
            double err(T(0.0));
            std::size_t iter_sum(0);
            num_threads = std::max(num_threads, std::size_t(1));
            std::atomic<std::size_t> iteration(0);
            std::mutex update_mutex;

            auto thread_func = [&] ()
            {
                std::size_t thread_err_sum = 0;
                double thread_err = 0;
                std::size_t thread_iter_sum = 0;
                while (true)
                {
                    std::size_t i = iteration.fetch_add(1, std::memory_order_relaxed);
                    if (i >= num) break;
                    Info tmp_info;
                    #if __cpp_if_constexpr < 201606L
                    if (row_major)
                    #else
                    if constexpr (row_major)
                    #endif
                    {
                        tmp_info = cheburaxa::approximate_vector<true>(size, r, U, A + i * shift_a, stride_a, V + i * r, 1, V_subsets + i * (r + 1));
                    }
                    else
                    {
                        tmp_info = cheburaxa::approximate_vector<false>(size, r, U, A + i * shift_a, stride_a, V + i, stride_V, V_subsets + i * (r + 1));
                    }
                    thread_err_sum += tmp_info.err_info;
                    thread_iter_sum += tmp_info.iter_num;
                    thread_err = std::max(thread_err, tmp_info.approximation_error);
                }

                std::lock_guard<std::mutex> guard(update_mutex);
                err_sum += thread_err_sum;
                iter_sum += thread_iter_sum;
                err = std::max(err, thread_err);
            };
            std::vector<std::thread> threads(num_threads - 1);
            for (std::size_t i = 0; i < threads.size(); ++i)
            {
                threads[i] = std::move(std::thread(thread_func));
            }
            thread_func();
            for (std::size_t i = 0; i < threads.size(); ++i)
            {
                threads[i].join();
            }
            return Info(err_sum, iter_sum, err);
        }

        template <bool row_major, class T = double>
        std::vector<Info> approximate_matrix(std::size_t m, std::size_t n, std::size_t r, const T *A, T *U, std::size_t *U_subsets, T *V, std::size_t *V_subsets, std::size_t num_threads = std::thread::hardware_concurrency(), T tolerance = 1e-7, std::size_t seed = 0)
        {
            if (seed == 0)
            {
                std::random_device rd;
                seed = rd();
            }
            std::mt19937 g(seed);
            random_subsets(m, m, r + 1, U_subsets, g());
            random_subsets(n, n, r + 1, V_subsets, g());
            std::normal_distribution<T> d;
            for (std::size_t i = 0; i < m * r; ++i)
            {
                U[i] = d(g);
            }
            for (std::size_t i = 0; i < m; ++i)
            {
                T nrm = 0;
                for (std::size_t j = 0; j < r; ++j)
                {
                    #if __cpp_if_constexpr < 201606L
                    if (row_major)
                    #else
                    if constexpr (row_major)
                    #endif
                    {
                        nrm += U[i * r + j] * U[i * r + j];
                    }
                    else
                    {
                        nrm += U[i + j * m] * U[i + j * m];
                    }
                }
                nrm = std::sqrt(nrm);
                for (std::size_t j = 0; j < r; ++j)
                {
                    #if __cpp_if_constexpr < 201606L
                    if (row_major)
                    #else
                    if constexpr (row_major)
                    #endif
                    {
                        U[i * r + j] /= nrm;
                    }
                    else
                    {
                        U[i + j * m] /= nrm;
                    }
                }
                #if __cpp_if_constexpr < 201606L
                if (row_major)
                #else
                if constexpr (row_major)
                #endif
                {
                    U[i * r] = std::abs(U[i * r]);
                }
                {
                    U[i] = std::abs(U[i]);
                }
            }

            std::vector<Info> info;
            T max_A = std::abs(A[linalg::idx_norm_inf(m * n, A)]);

            info.push_back(cheburaxa::approximate_matrix<false, row_major>(m, n, r, A, U, V, V_subsets, num_threads));
            info.push_back(cheburaxa::approximate_matrix<true, row_major>(m, n, r, A, V, U, U_subsets, num_threads));
            T curr_err = info.back().approximation_error;
            T prev_err = 2 * curr_err;

            while (prev_err - curr_err > tolerance * max_A)
            {
                info.push_back(cheburaxa::approximate_matrix<false, row_major>(m, n, r, A, U, V, V_subsets, num_threads));
                
                info.push_back(cheburaxa::approximate_matrix<true, row_major>(m, n, r, A, V, U, U_subsets, num_threads));
                prev_err = curr_err;
                curr_err = info.back().approximation_error;
            }
            return info;
        }
}

#endif//CHEBURAXA_HPP

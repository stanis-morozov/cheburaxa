#ifndef CHEBURAXA_LINALG_HPP
#define CHEBURAXA_LINALG_HPP

#include <cmath>
#include <algorithm>

namespace cheburaxa
{
    // Required linear algebra operations and methods
    namespace linalg
    {
        // Calculates rotation [c s ; -s c] for the vector [f; g] that zeros second component of the vector
        template <class T>
            static inline void calculate_rotation(T &f, T &g,
                                                  T &c, T &s)
            {
                // Fast exit if g is 0
                if (std::fpclassify(g) == FP_ZERO)
                {
                    c = T(1.0);
                    s = T(0.0);
                    return;
                }
                // Fast exit if f is 0
                if (std::fpclassify(f) == FP_ZERO)
                {
                    c = T(0.0);
                    s = T(1.0);
                    std::swap(f, g);
                    return;
                }

                // Normalization of numbers, so that max of f * f and g * g guaranteed to not overflow or underflow
                int exp = std::max(std::ilogb(f), std::ilogb(g));
                f = std::scalbn(f, -exp);
                g = std::scalbn(g, -exp);

                // Compute c and s
                T r = std::sqrt(f * f + g * g);
                c = f / r;
                s = g / r;

                // Restore exponent for f
                f = std::scalbn(r, exp);
                g = T(0.0);
            }

        // Applies rotation [c s ; -s c] to vectors x and y of length n
        template <class T>
            static inline void apply_rotation(const T &c, const T &s,
                                              std::size_t n,
                                              T *x,
                                              T *y)
            {
                // Pointer to an end of vector X
                const T *X = x + n;

                // Apply rotation for every pair of elements from x and y
                for (; x < X; ++x, ++y)
                {
                    T a = *x * c + *y * s;
                    T b = *y * c - *x * s;
                    *x = a;
                    *y = b;
                }
            }

        // Computes dot product of vectors x and y of length n
        template <class T>
            static inline T dot_product(std::size_t n,
                                        const T *x,
                                        const T *y)
            {
                T s(0.0);
                const T *X = x + n;
                for (; x < X; ++x, ++y)
                {
                    s += *x * *y;
                }
                return s;
            }

        // Computes y = alpha * x + y
        template <class T>
            static inline void alpha_x_plus_y(std::size_t n,
                                              const T &alpha,
                                              const T *x,
                                              T *y)
            {
                const T *Y = y + n;
                for (; y < Y; ++x, ++y)
                {
                    *y += *x * alpha;
                }
            }

        // Computes z = alpha * x + beta * y
        template <class T>
            static inline void alpha_x_plus_beta_y(std::size_t n,
                                                   const T &alpha,
                                                   const T *x,
                                                   const T beta,
                                                   const T *y,
                                                   T *z)
            {
                const T *Z = z + n;
                for (; z < Z; ++x, ++y, ++z)
                {
                    *z = *x * alpha + *y * beta;
                }
            }

        // Computes 1-norm of vector x of length n
        template <class T>
            static inline T norm_1(std::size_t n,
                                   const T *x)
            {
                T s = 0;
                const T *X = x + n;
                for (; x < X; ++x)
                {
                    s += std::abs(*x);
                }
                return s;
            }

        // Computes the smallest i such that |x_i| = \|x\|_1
        template <class T>
            static inline std::size_t idx_norm_inf(std::size_t n,
                                                   const T *x)
            {
                return std::max_element(x, x + n, [](T a, T b)
                        {
                        return std::abs(a) < std::abs(b);
                        }
                        ) - x;
            }

        // Computes difference of two matrix rows: x = a_{i_1} - a_{i_2}
        // A -- m x n Fortran ordered matrix
        template <bool row_major, class T>
            static inline void row_diff(std::size_t m, std::size_t n,
                                        std::size_t i1, std::size_t i2,
                                        const T* A,
                                        T *x)
            {
                const T* a1 = A + i1 * (row_major ? n : 1);
                const T* a2 = A + i2 * (row_major ? n : 1);
                const T* X = x + n;
                for (; x < X; ++x, a1 += (row_major ? 1 : m), a2 += (row_major ? 1 : m))
                {
                    *x = *a1 - *a2;
                }
            }

        // Copies i-th row of matrix A to vector x: x = a_i
        // A -- m x n Fortran ordered matrix
        template <bool row_major, class T>
            static inline void copy_row(std::size_t m, std::size_t n,
                                        std::size_t i,
                                        const T* A,
                                        T* x)
            {
                const T* a = A + i * (row_major ? n : 1);
                const T* X = x + n;
                for (; x < X; ++x, a += (row_major ? 1 : m))
                {
                    *x = *a;
                }
            }

        // Computes y = A x
        // A -- m x n Fortran ordered matrix
        template <class T>
            static inline void matvec(std::size_t m, std::size_t n,
                                      const T *A,
                                      const T *x,
                                      T *y)
            {
                const T *eA = A + m * n;
                T *Y = y + m;
                for (; y < Y; ++y) *y = 0;
                for (y = Y - m; A < eA; ++x, y -= m)
                {
                    for (; y < Y; ++y, ++A)
                    {
                        *y += *A * *x;
                    }
                }
            }

        // Computes y = A^T x
        // A -- m x n Fortran ordered matrix
        template <class T>
            static inline void transposed_matvec(std::size_t m, std::size_t n,
                                                 const T *A,
                                                 const T *x,
                                                 T *y)
            {
                const T *eA = A + m * n;
                const T *X = x + m;
                for (; A < eA; x -= m, ++y)
                {
                    *y = 0;
                    for (; x < X; ++x, ++A)
                    {
                        *y += *A * *x;
                    }
                }
            }

        // Computes r = b - A x
        // A -- m x n Fortran ordered matrix
        template <bool row_major, class T>
            static inline void compute_residual(std::size_t m, std::size_t n,
                                                const T *b,
                                                std::size_t stride_b,
                                                const T *A,
                                                const T *x,
                                                T *r)
            {
                const T *eA = A + m * n;
                T *R = r + m;
                for (; r < R; ++r, b += stride_b) *r = *b;
                for (r = R - m; A < eA; x += (row_major ? -n : 1), r += (row_major ? 1 : -m))
                {
                    const T *e = A + (row_major ? n : m);
                    for (; A < e; r += (row_major ? 0 : 1), x += (row_major ? 1 : 0), ++A)
                    {
                        *r -= *A * *x;
                    }
                }
            }

        // Computes k rows given by indices i from the matrix A to matrix B:
        // b_k = a_{i_k}
        // A -- m x n Fortran ordered matrix
        // B -- k x n Fortran ordered matrix
        template <bool row_major, class T>
            static inline void copy_rows(std::size_t m, std::size_t n,
                                         std::size_t k, std::size_t *i,
                                         const T* A,
                                         T* B)
            {
                const T* a = A + *i * (row_major ? n : 1);
                const T* Xc = B + k;
                const T* X = B + k * n;
                for (; B < Xc; B -= k * n - 1, ++i)
                {
                    a = A + *i * (row_major ? n : 1);
                    for (; B < X; B += k, a += (row_major ? 1 : m))
                    {
                        *B = *a;
                    }
                }
            }


        // Solves L x = b inplace
        // L -- n x n Fortran ordered lower triangular matrix (only lower triangular referenced)
        template <class T>
            void solve_lower_triangular(std::size_t n,
                                        const T *L,
                                        T *b)
            {
                for (std::size_t j = 0; j < n; ++j, L += j, b += j - n)
                {
                    *b /= *L;
                    const T& v = *b;
                    const T *B = b + n - j;
                    ++b;
                    ++L;
                    for (; b < B; ++b, ++L)
                    {
                        *b -= v * *L;
                    }
                }
            }

        // Solves L^T x = b inplace
        // L -- n x n Fortran ordered lower triangular matrix (only lower triangular referenced)
        template <class T>
            void solve_lower_triangular_transposed(std::size_t n,
                                                   const T *L,
                                                   T *b)
            {
                const T* B = b;
                b += n - 1;
                L += n * n - 1;

                for (std::size_t j = 0; j < n; ++j, L += n * (n - j) - 1, b += n - j)
                {
                    *b /= *L;
                    const T& v = *b;
                    --b;
                    L -= n;
                    for (; b >= B; --b, L -= n)
                    {
                        *b -= v * *L;
                    }
                }
            }

        // Inplace computation of reflection vector u such that:
        // (I - u u^* / s) x = r e_1, where e_1 = [1 0 ... ]^T, r = -sign(x_1) \|x\|_2, (I - u u^* / s) is unitary matrix
        // returns [r, s] pair
        template <class T>
            static inline std::pair<T, T> calculate_reflection(std::size_t n, T *x)
            {
                T nrm(0.0);
                T *X = x;
                int exp_nrm = 0;
                for (x = X + n - 1; x > X; --x)
                {
                    int exp = std::max(exp_nrm, std::ilogb(*x));
                    T tmp = std::scalbn(*x, -exp);
                    nrm = std::scalbn(nrm, 2 * (-exp + exp_nrm));
                    exp_nrm = exp;
                    nrm += tmp * tmp;
                }
                if (std::fpclassify(nrm) == FP_ZERO)
                {
                    T r = *x;
                    *x = T(0.0);
                    return std::make_pair(r, nrm);
                }
                int exp = std::max(exp_nrm, std::ilogb(*X));
                *X = std::scalbn(*X, -exp);
                nrm = std::scalbn(nrm, 2 * (-exp + exp_nrm));
                exp_nrm = exp;
                T r = std::copysign(std::sqrt(nrm + *X * *X), *X);
                for (x = X + n - 1; x > X; --x)
                {
                    *x = std::scalbn(*x, -exp_nrm);
                }
                *X += r;
                nrm += *X * *X;
                nrm /= 2;
                r = std::scalbn(r, exp_nrm);
                return std::make_pair(-r, nrm);
            }


        // Computes QR decomposition of matrix A
        // A -- m x n Fortran ordered matrix, m >= n
        // Q -- m x m Fortran ordered matrix
        // Matrix A is replaced by R^T -- on exit contains n x m Fortran ordered lower triangular matrix
        // work -- work array of size m
        template <class T>
            void qr(std::size_t m, std::size_t n,
                    T *A,
                    T *Q,
                    T *work)
            {
                T* a = A;
                T* L = A;

                auto t = calculate_reflection(m, a);
                const T &r = t.first;
                const T &s = t.second;

                if (std::fpclassify(s) != FP_ZERO)
                {
                    for (std::size_t k = 0; k < m; ++k)
                    {
                        for (std::size_t i = 0; i < m; ++i)
                        {
                            Q[i + k * m] = -a[i] * a[k] / s;
                        }
                        Q[k + k * m] += T(1.0);
                    }
                    for (std::size_t k = 1; k < n; ++k)
                    {
                        T v = -dot_product(m, a, a + k * m) / s;
                        alpha_x_plus_y(m, v, a, a + k * m);
                    }
                }
                else
                {
                    for (std::size_t k = 0; k < m; ++k)
                    {
                        for (std::size_t i = 0; i < m; ++i)
                        {
                            Q[i + k * m] = T(0.0);
                        }
                        Q[k + k * m] += T(1.0);
                    }
                }
                L[0] = r;
                ++L;
                for (std::size_t k = 1; k < n; ++k, ++L)
                {
                    *L = A[k * m];
                }
                a += m + 1;
                Q += m;

                T *b = work;
                for (std::size_t j = 1; j < n; ++j, a += m + 1, Q += m)
                {

                    auto t = calculate_reflection(m - j, a);
                    const T &r = t.first;
                    const T &s = t.second;

                    if (std::fpclassify(s) != FP_ZERO)
                    {
                        matvec(m, m - j, Q, a, b);
                        for (std::size_t k = 0; k < m - j; ++k)
                        {
                            alpha_x_plus_y(m, -a[k] / s, b, Q + k * m);
                        }
                        for (std::size_t k = 1; k < n - j; ++k)
                        {
                            T v = dot_product(m - j, a, a + k * m);
                            alpha_x_plus_y(m - j, -v / s, a, a + k * m);
                        }
                    }
                    for (std::size_t k = 0; k < j; ++k, ++L)
                    {
                        *L = T(0.0);
                    }
                    L[0] = r;
                    ++L;
                    for (std::size_t k = 1; k < n - j; ++k, ++L)
                    {
                        *L = a[k * m];
                    }
                }
                for (; L < A + m * n; ++L) *L = T(0.0);
            }

        // Performs one rank QR update: Q L^T + Q x y^T = \hat{Q} \hat{L}^T
        // Update is performed in-place
        // Q is a Fortran-ordered m x m matrix
        // L is a Fortran-ordered n x m matrix
        // x is a vector of size m
        // y is a vector of size n
        template <class T>
            void rank1_qr_update(std::size_t m, std::size_t n,
                                 T *Q,
                                 T *L,
                                 T *x,
                                 const T *y)
            {
                T c, s;
                // use rotations to zeroout m-th element of x, then (m - 1)-th element and so on
                // updates Q and L accordingly
                // Only first component of x is nonzero after this cycle
                // And L becomes lower Hessenberg
                for (std::size_t i = m - 1; i > 0; --i) {
                    calculate_rotation(x[i - 1], x[i], c, s);
                    apply_rotation(c, s, n - i + 1, L + (i - 1) * n + i - 1, L + i * n + i - 1);
                    apply_rotation(c, s, m, Q + (i - 1) * m, Q + i * m);
                }

                // Now we add y x^T to L
                // As only first component of x is nonzero L remains lower Hessenberg
                alpha_x_plus_y(n, x[0], y, L);

                // Now zeroout upperdiagonal of L using rotations
                // And update Q accordingly
                for (std::size_t i = 0; i < std::min(n, m - 1); ++i) {
                    calculate_rotation(L[i * n + i],  L[(i + 1) * n + i], c, s);
                    apply_rotation(c, s, n - i - 1, L + i * n + i + 1, L + (i + 1) * n + i + 1);
                    apply_rotation(c, s, m, Q + i * m, Q + (i + 1) * m);
                }
            }
    }
}

#endif//CHEBURAXA_LINALG_HPP
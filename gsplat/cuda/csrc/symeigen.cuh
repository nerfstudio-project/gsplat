#ifndef SYMEIGEN
#define SYMEIGEN
#include <glm/glm.hpp>

namespace glm_modification
{
	// Incorporate the transferSign, pythag, equal, and findEigenvaluesSymReal functions from the glm library, 
	// with small modifications on findEgienvaluesSymReal to ensure numerical stability for big Gaussian kernels.
	// https://github.com/g-truc/glm/blob/33b4a621a697a305bc3a7610d290677b96beb181/glm/gtx/pca.inl
	// https://github.com/g-truc/glm/blob/33b4a621a697a305bc3a7610d290677b96beb181/glm/ext/scalar_relational.inl
	template<typename genType>
	__forceinline__ __device__ bool equal(genType const& x, genType const& y, genType const& epsilon)
	{
		return abs(x - y) <= epsilon;
	}

	template<typename T>
	__forceinline__ __device__ static T transferSign(T const& v, T const& s)
	{
		return ((s) >= 0 ? glm::abs(v) : -glm::abs(v));
	}

	template<typename T>
	__forceinline__ __device__ static T pythag(T const& a, T const& b) {
		static const T epsilon = static_cast<T>(0.0000001);
		T absa = glm::abs(a);
		T absb = glm::abs(b);
		if(absa > absb) {
			absb /= absa;
			absb *= absb;
			return absa * glm::sqrt(static_cast<T>(1) + absb);
		}
		if(glm_modification::equal<T>(absb, 0, epsilon)) return static_cast<T>(0);
		absa /= absb;
		absa *= absa;
		return absb * glm::sqrt(static_cast<T>(1) + absa);
	}


	template<glm::length_t D, typename T, glm::qualifier Q>
	__forceinline__ __device__ unsigned int findEigenvaluesSymReal
	(
		glm::mat<D, D, T, Q> const& covarMat,
		glm::vec<D, T, Q>& outEigenvalues,
		glm::mat<D, D, T, Q>& outEigenvectors
	)
	{

		T a[D * D]; // matrix -- input and workspace for algorithm (will be changed inplace)
		T d[D]; // diagonal elements
		T e[D]; // off-diagonal elements

		for(glm::length_t r = 0; r < D; r++)
			for(glm::length_t c = 0; c < D; c++)
				a[(r) * D + (c)] = covarMat[c][r];

		// 1. Householder reduction.
		glm::length_t l, k, j, i;
		T scale, hh, h, g, f;
		static const T epsilon = static_cast<T>(0.0000001);

		for(i = D; i >= 2; i--)
		{
			l = i - 1;
			h = scale = 0;
			if(l > 1)
			{
				for(k = 1; k <= l; k++)
				{
					scale += glm::abs(a[(i - 1) * D + (k - 1)]);
				}
				if(glm_modification::equal<T>(scale, 0, epsilon))
				{
					e[i - 1] = a[(i - 1) * D + (l - 1)];
				}
				else
				{
					for(k = 1; k <= l; k++)
					{
						a[(i - 1) * D + (k - 1)] /= scale;
						h += a[(i - 1) * D + (k - 1)] * a[(i - 1) * D + (k - 1)];
					}
					f = a[(i - 1) * D + (l - 1)];
					g = ((f >= 0) ? -glm::sqrt(h) : glm::sqrt(h));
					e[i - 1] = scale * g;
					h -= f * g;
					a[(i - 1) * D + (l - 1)] = f - g;
					f = 0;
					for(j = 1; j <= l; j++)
					{
						a[(j - 1) * D + (i - 1)] = a[(i - 1) * D + (j - 1)] / h;
						g = 0;
						for(k = 1; k <= j; k++)
						{
							g += a[(j - 1) * D + (k - 1)] * a[(i - 1) * D + (k - 1)];
						}
						for(k = j + 1; k <= l; k++)
						{
							g += a[(k - 1) * D + (j - 1)] * a[(i - 1) * D + (k - 1)];
						}
						e[j - 1] = g / h;
						f += e[j - 1] * a[(i - 1) * D + (j - 1)];
					}
					hh = f / (h + h);
					for(j = 1; j <= l; j++)
					{
						f = a[(i - 1) * D + (j - 1)];
						e[j - 1] = g = e[j - 1] - hh * f;
						for(k = 1; k <= j; k++)
						{
							a[(j - 1) * D + (k - 1)] -= (f * e[k - 1] + g * a[(i - 1) * D + (k - 1)]);
						}
					}
				}
			}
			else
			{
				e[i - 1] = a[(i - 1) * D + (l - 1)];
			}
			d[i - 1] = h;
		}
		d[0] = 0;
		e[0] = 0;
		for(i = 1; i <= D; i++)
		{
			l = i - 1;
			if(!glm_modification::equal<T>(d[i - 1], 0, epsilon))
			{
				for(j = 1; j <= l; j++)
				{
					g = 0;
					for(k = 1; k <= l; k++)
					{
						g += a[(i - 1) * D + (k - 1)] * a[(k - 1) * D + (j - 1)];
					}
					for(k = 1; k <= l; k++)
					{
						a[(k - 1) * D + (j - 1)] -= g * a[(k - 1) * D + (i - 1)];
					}
				}
			}
			d[i - 1] = a[(i - 1) * D + (i - 1)];
			a[(i - 1) * D + (i - 1)] = 1;
			for(j = 1; j <= l; j++)
			{
				a[(j - 1) * D + (i - 1)] = a[(i - 1) * D + (j - 1)] = 0;
			}
		}

		// 2. Calculation of eigenvalues and eigenvectors (QL algorithm)
		glm::length_t m, iter;
		T s, r, p, dd, c, b;
		const glm::length_t MAX_ITER = 30;

		for(i = 2; i <= D; i++)
		{
			e[i - 2] = e[i - 1];
		}
		e[D - 1] = 0;

		for(l = 1; l <= D; l++)
		{
			iter = 0;
			do
			{
				for(m = l; m <= D - 1; m++)
				{
					dd = glm::abs(d[m - 1]) + glm::abs(d[m - 1 + 1]);
					if(glm_modification::equal<T>(glm::abs(e[m - 1]), 0, epsilon))
						break;
				}
				if(m != l)
				{
					if(iter++ == MAX_ITER)
					{
						return 0; // Too many iterations in FindEigenvalues
					}
					g = (d[l - 1 + 1] - d[l - 1]) / (2 * e[l - 1]);
					r = pythag<T>(g, 1);
					g = d[m - 1] - d[l - 1] + e[l - 1] / (g + transferSign(r, g));
					s = c = 1;
					p = 0;
					for(i = m - 1; i >= l; i--)
					{
						f = s * e[i - 1];
						b = c * e[i - 1];
						e[i - 1 + 1] = r = pythag(f, g);
						if(glm_modification::equal<T>(r, 0, epsilon))
						{
							d[i - 1 + 1] -= p;
							e[m - 1] = 0;
							break;
						}
						s = f / r;
						c = g / r;
						g = d[i - 1 + 1] - p;
						r = (d[i - 1] - g) * s + 2 * c * b;
						d[i - 1 + 1] = g + (p = s * r);
						g = c * r - b;
						for(k = 1; k <= D; k++)
						{
							f = a[(k - 1) * D + (i - 1 + 1)];
							a[(k - 1) * D + (i - 1 + 1)] = s * a[(k - 1) * D + (i - 1)] + c * f;
							a[(k - 1) * D + (i - 1)] = c * a[(k - 1) * D + (i - 1)] - s * f;
						}
					}
					if(glm_modification::equal<T>(r, 0, epsilon) && (i >= l))
						continue;
					d[l - 1] -= p;
					e[l - 1] = g;
					e[m - 1] = 0;
				}
			} while(m != l);
		}

		// 3. output
		for(i = 0; i < D; i++)
			outEigenvalues[i] = d[i];
		for(i = 0; i < D; i++)
			for(j = 0; j < D; j++)
				outEigenvectors[i][j] = a[(j) * D + (i)];

		return D;
	}
}

#endif
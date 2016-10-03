#include "gmm_ang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef VL_DISABLE_SSE2
#include "vl/mathop_sse2.h"
#endif

#ifndef VL_DISABLE_AVX
#include "vl/mathop_avx.h"
#endif

/* ---------------------------------------------------------------- */
#ifndef ANG_GMM_INSTANTIATING
/* ---------------------------------------------------------------- */

#define ANG_GMM_MIN_VARIANCE 1e-6
#define ANG_GMM_MIN_POSTERIOR 1e-2
#define ANG_GMM_MIN_PRIOR 1e-6

#endif

/* ---------------------------------------------------------------- */
#ifdef ANG_GMM_INSTANTIATING
/* ---------------------------------------------------------------- */

void
VL_XCAT(ang_get_hard_posteriors_score_, SFX)
(TYPE * scores,
 vl_size numClusters,
 vl_size numData,
 TYPE const * priors,
 TYPE const * means,
 vl_size dimension,
 TYPE const * covariances,
 TYPE const * data,
 TYPE const * pst_w)
{
  vl_index i_d, i_cl;
  vl_size dim;

  TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
  TYPE * invCovariances ;
  TYPE * prefix;

#if (FLT == VL_TYPE_FLOAT)
  VlFloatVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_f(VlDistanceMahalanobis) ;
#else
  VlDoubleVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_d(VlDistanceMahalanobis) ;
#endif

  invCovariances = vl_malloc(sizeof(TYPE) * numClusters * dimension) ;
  prefix = vl_malloc(numClusters * sizeof(TYPE));

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,dim) num_threads(vl_get_max_threads())
#endif
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE logSigma = 0 ;
	TYPE logWeights;
    if (priors[i_cl] < ANG_GMM_MIN_PRIOR) {
      logWeights = - (TYPE) VL_INFINITY_D ;
    } else {
      logWeights = log(priors[i_cl]);
    }
    for(dim = 0 ; dim < dimension ; ++ dim) {
      logSigma += log(covariances[i_cl*dimension + dim]);
      invCovariances [i_cl*dimension + dim] = (TYPE) 1.0 / covariances[i_cl*dimension + dim];
    }
    TYPE logCovariances = logSigma;
	prefix[i_cl] = logWeights - halfDimLog2Pi - 0.5 * logCovariances;
  } /* end of parallel region */

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL) \
num_threads(vl_get_max_threads())
#endif
  for (i_d = 0 ; i_d < (signed)numData ; ++ i_d) {
    TYPE maxPosterior = (TYPE)(-VL_INFINITY_D) ;
	int index = -1;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
      TYPE p = prefix[i_cl]
      - 0.5 * distFn (dimension,
                      data + i_d * dimension,
                      means + i_cl * dimension,
                      invCovariances + i_cl * dimension) ;
      if (p > maxPosterior) { 
		  maxPosterior = p ; 
		  index = i_cl;
	  }
    }

	scores[i_d] = pst_w[index];
  } /* end of parallel region */

  vl_free(invCovariances);
  vl_free(prefix);
}
/* ---------------------------------------------------------------- */
/*                                            Posterior assignments */
/* ---------------------------------------------------------------- */

void
VL_XCAT(ang_get_soft_posteriors_score_, SFX)
(TYPE * index_max_posteriors,
 vl_size numClusters,
 vl_size numData,
 TYPE const * priors,
 TYPE const * means,
 vl_size dimension,
 TYPE const * covariances,
 TYPE const * data,
 TYPE const * pst_w)
{
  vl_index i_d, i_cl;
  vl_size dim;
  double LL = 0;

  TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
  TYPE * logCovariances ;
  TYPE * logWeights ;
  TYPE * invCovariances ;
  TYPE * prefix;
  TYPE * posteriors;

#if (FLT == VL_TYPE_FLOAT)
  VlFloatVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_f(VlDistanceMahalanobis) ;
#else
  VlDoubleVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_d(VlDistanceMahalanobis) ;
#endif

  logCovariances = vl_malloc(sizeof(TYPE) * numClusters) ;
  invCovariances = vl_malloc(sizeof(TYPE) * numClusters * dimension) ;
  logWeights = vl_malloc(numClusters * sizeof(TYPE)) ;
  posteriors = vl_malloc(numClusters * sizeof(TYPE));
  prefix = vl_malloc(numClusters * sizeof(TYPE));

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,dim) num_threads(vl_get_max_threads())
#endif
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE logSigma = 0 ;
    if (priors[i_cl] < ANG_GMM_MIN_PRIOR) {
      logWeights[i_cl] = - (TYPE) VL_INFINITY_D ;
    } else {
      logWeights[i_cl] = log(priors[i_cl]);
    }
    for(dim = 0 ; dim < dimension ; ++ dim) {
      logSigma += log(covariances[i_cl*dimension + dim]);
      invCovariances [i_cl*dimension + dim] = (TYPE) 1.0 / covariances[i_cl*dimension + dim];
    }
    logCovariances[i_cl] = logSigma;
	prefix[i_cl] = logWeights[i_cl] - halfDimLog2Pi - 0.5 * logCovariances[i_cl];
  } /* end of parallel region */

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL) \
num_threads(vl_get_max_threads())
#endif
  for (i_d = 0 ; i_d < (signed)numData ; ++ i_d) {
    TYPE clusterPosteriorsSum = 0;
    TYPE maxPosterior = (TYPE)(-VL_INFINITY_D) ;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
      TYPE p = prefix[i_cl]
      - 0.5 * distFn (dimension,
                      data + i_d * dimension,
                      means + i_cl * dimension,
                      invCovariances + i_cl * dimension) ;
      posteriors[i_cl] = p ;
      if (p > maxPosterior) { 
		  maxPosterior = p ; 
	  }
    }

	for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
	  TYPE p = posteriors[i_cl] ;
	  p =  exp(p - maxPosterior);
	  posteriors[i_cl] = p ;
	  clusterPosteriorsSum += p ;
	}

	TYPE score = 0;
	for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
	  score += posteriors[i_cl] / clusterPosteriorsSum * pst_w[i_cl];
	}

	index_max_posteriors[i_d] = score;
  } /* end of parallel region */

  vl_free(logCovariances);
  vl_free(logWeights);
  vl_free(invCovariances);
  vl_free(prefix);
  vl_free(posteriors);

}

void
VL_XCAT(ang_get_gmm_data_posteriors_, SFX)
(TYPE * index_max_posteriors,
 vl_size numClusters,
 vl_size numData,
 TYPE const * priors,
 TYPE const * means,
 vl_size dimension,
 TYPE const * covariances,
 TYPE const * data)
{
  vl_index i_d, i_cl;
  vl_size dim;
  double LL = 0;

  TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
  TYPE * logCovariances ;
  TYPE * logWeights ;
  TYPE * invCovariances ;
  TYPE * prefix;
  TYPE * posteriors;

#if (FLT == VL_TYPE_FLOAT)
  VlFloatVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_f(VlDistanceMahalanobis) ;
#else
  VlDoubleVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_d(VlDistanceMahalanobis) ;
#endif

  logCovariances = vl_malloc(sizeof(TYPE) * numClusters) ;
  invCovariances = vl_malloc(sizeof(TYPE) * numClusters * dimension) ;
  logWeights = vl_malloc(numClusters * sizeof(TYPE)) ;
  posteriors = vl_malloc(numClusters * sizeof(TYPE));
  prefix = vl_malloc(numClusters * sizeof(TYPE));

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,dim) num_threads(vl_get_max_threads())
#endif
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE logSigma = 0 ;
    if (priors[i_cl] < ANG_GMM_MIN_PRIOR) {
      logWeights[i_cl] = - (TYPE) VL_INFINITY_D ;
    } else {
      logWeights[i_cl] = log(priors[i_cl]);
    }
    for(dim = 0 ; dim < dimension ; ++ dim) {
      logSigma += log(covariances[i_cl*dimension + dim]);
      invCovariances [i_cl*dimension + dim] = (TYPE) 1.0 / covariances[i_cl*dimension + dim];
    }
    logCovariances[i_cl] = logSigma;
	prefix[i_cl] = logWeights[i_cl] - halfDimLog2Pi - 0.5 * logCovariances[i_cl];
  } /* end of parallel region */

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL) \
num_threads(vl_get_max_threads())
#endif
  for (i_d = 0 ; i_d < (signed)numData ; ++ i_d) {
    TYPE clusterPosteriorsSum = 0;
    TYPE maxPosterior = (TYPE)(-VL_INFINITY_D) ;
	TYPE index = -1;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
      TYPE p = prefix[i_cl]
      - 0.5 * distFn (dimension,
                      data + i_d * dimension,
                      means + i_cl * dimension,
                      invCovariances + i_cl * dimension) ;
      posteriors[i_cl] = p ;
      if (p > maxPosterior) { 
		  maxPosterior = p ; 
		  index = i_cl;
	  }
    }

	index_max_posteriors[i_d] = index;

	if (false) {
		for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
		  TYPE p = posteriors[i_cl] ;
		  p =  exp(p - maxPosterior);
		  posteriors[i_cl] = p ;
		  clusterPosteriorsSum += p ;
		}

		for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
		  posteriors[i_cl] /= clusterPosteriorsSum ;
		}
	}
  } /* end of parallel region */

  vl_free(logCovariances);
  vl_free(logWeights);
  vl_free(invCovariances);
  vl_free(prefix);
  vl_free(posteriors);

}

/** @fn vl_get_gmm_data_posterior_f(float*,vl_size,vl_size,float const*,float const*,vl_size,float const*,float const*)
 ** @brief Get Gaussian modes posterior probabilities
 ** @param posteriors posterior probabilities (output)/
 ** @param numClusters number of modes in the GMM model.
 ** @param numData number of data elements.
 ** @param priors prior mode probabilities of the GMM model.
 ** @param means means of the GMM model.
 ** @param dimension data dimension.
 ** @param covariances diagonal covariances of the GMM model.
 ** @param data data.
 ** @return data log-likelihood.
 **
 ** This is a helper function that does not require a ::VlGMM object
 ** instance to operate.
 **/

double
VL_XCAT(vl_get_gmm_data_posteriors_, SFX)
(TYPE * posteriors,
 vl_size numClusters,
 vl_size numData,
 TYPE const * priors,
 TYPE const * means,
 vl_size dimension,
 TYPE const * covariances,
 TYPE const * data)
{
  vl_index i_d, i_cl;
  vl_size dim;
  double LL = 0;

  TYPE halfDimLog2Pi = (dimension / 2.0) * log(2.0*VL_PI);
  TYPE * logCovariances ;
  TYPE * logWeights ;
  TYPE * invCovariances ;

#if (FLT == VL_TYPE_FLOAT)
  VlFloatVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_f(VlDistanceMahalanobis) ;
#else
  VlDoubleVector3ComparisonFunction distFn = vl_get_vector_3_comparison_function_d(VlDistanceMahalanobis) ;
#endif

  logCovariances = vl_malloc(sizeof(TYPE) * numClusters) ;
  invCovariances = vl_malloc(sizeof(TYPE) * numClusters * dimension) ;
  logWeights = vl_malloc(numClusters * sizeof(TYPE)) ;

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,dim) num_threads(vl_get_max_threads())
#endif
  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
    TYPE logSigma = 0 ;
    if (priors[i_cl] < ANG_GMM_MIN_PRIOR) {
      logWeights[i_cl] = - (TYPE) VL_INFINITY_D ;
    } else {
      logWeights[i_cl] = log(priors[i_cl]);
    }
    for(dim = 0 ; dim < dimension ; ++ dim) {
      logSigma += log(covariances[i_cl*dimension + dim]);
      invCovariances [i_cl*dimension + dim] = (TYPE) 1.0 / covariances[i_cl*dimension + dim];
    }
    logCovariances[i_cl] = logSigma;
  } /* end of parallel region */

#if defined(_OPENMP)
#pragma omp parallel for private(i_cl,i_d) reduction(+:LL) \
num_threads(vl_get_max_threads())
#endif
  for (i_d = 0 ; i_d < (signed)numData ; ++ i_d) {
    TYPE clusterPosteriorsSum = 0;
    TYPE maxPosterior = (TYPE)(-VL_INFINITY_D) ;

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++ i_cl) {
      TYPE p =
      logWeights[i_cl]
      - halfDimLog2Pi
      - 0.5 * logCovariances[i_cl]
      - 0.5 * distFn (dimension,
                      data + i_d * dimension,
                      means + i_cl * dimension,
                      invCovariances + i_cl * dimension) ;
      posteriors[i_cl + i_d * numClusters] = p ;
      if (p > maxPosterior) { maxPosterior = p ; }
    }

    for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
      TYPE p = posteriors[i_cl + i_d * numClusters] ;
      p =  exp(p - maxPosterior);
      posteriors[i_cl + i_d * numClusters] = p ;
      clusterPosteriorsSum += p ;
    }

    LL +=  log(clusterPosteriorsSum) + (double) maxPosterior ;

	for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
	  posteriors[i_cl + i_d * numClusters] /= clusterPosteriorsSum ;
	}
  } /* end of parallel region */

  vl_free(logCovariances);
  vl_free(logWeights);
  vl_free(invCovariances);

  return LL;
}
/* ---------------------------------------------------------------- */
#else /* ANG_GMM_INSTANTIATING */
/* ---------------------------------------------------------------- */

#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define ANG_GMM_INSTANTIATING
#include "gmm_ang.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define ANG_GMM_INSTANTIATING
#include "gmm_ang.c"
#endif

/* ANG_GMM_INSTANTIATING */
#endif

#undef SFX
#undef TYPE
#undef FLT
#undef ANG_GMM_INSTANTIATING

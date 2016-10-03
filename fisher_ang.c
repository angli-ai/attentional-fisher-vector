/** @file fisher_ang.c
 ** @brief Fisher - Declaration
 ** @author Ang Li
 **/

#include "fisher_ang.h"
#include <vl/gmm.h>
#include <vl/mathop.h>

#include <mex.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef ANG_FISHER_INSTANTIATING

static vl_size
VL_XCAT(_ang_fisher_encode_, SFX)
(TYPE * enc, int* num_feats_used,
 TYPE const * means, vl_size dimension, vl_size numClusters,
 bool const * isValidCluster, 
 TYPE const * covariances,
 TYPE const * priors,
 TYPE const * data, vl_size numData, TYPE const * data_w, float data_thresh,
 int flags)
{
  vl_size dim;
  vl_index i_cl, i_d;
  vl_size numTerms = 0 ;
  TYPE * posteriors ;
  TYPE * sqrtInvSigma;
  vl_size numValidData = numData;
  vl_size numValidClusters = numClusters;

  bool * isValidData = 0;
  int * idxClusters = 0;

  assert(numClusters >= 1) ;
  assert(dimension >= 1) ;

  posteriors = vl_malloc(sizeof(TYPE) * numClusters * numData);
  sqrtInvSigma = vl_malloc(sizeof(TYPE) * dimension * numClusters);

  if (flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE) {
	  idxClusters = vl_malloc(sizeof(int) * numClusters);
	  idxClusters[0] = 0;
	  for (i_cl = 1; i_cl < (signed)numClusters; ++ i_cl)
		  if (isValidCluster[i_cl - 1])
			idxClusters[i_cl] = idxClusters[i_cl - 1] + 1;
		else idxClusters[i_cl] = idxClusters[i_cl - 1];
	  if (isValidCluster[numClusters - 1])
	  	numValidClusters = idxClusters[numClusters - 1] + 1;
	  else numValidClusters = idxClusters[numClusters - 1];
  }

  assert(numValidClusters >= 1);
  memset(enc, 0, sizeof(TYPE) * 2 * dimension * numValidClusters) ;

  for (i_cl = 0 ; i_cl < (signed)numClusters ; ++i_cl) {
    for(dim = 0; dim < dimension; dim++) {
      sqrtInvSigma[i_cl*dimension + dim] = sqrt(1.0 / covariances[i_cl*dimension + dim]);
    }
  }

  VL_XCAT(vl_get_gmm_data_posteriors_, SFX)(posteriors, numClusters, numData,
                                            priors,
                                            means, dimension,
                                            covariances,
                                            data) ;

  /*get index of good data points*/
  if (flags & SELECTIVE_FISHER_FLAG_FEAT_REDUCE) {
	  /* need to throw some data points*/

	  isValidData = vl_malloc(sizeof(bool) * numData);

	  if (flags & SELECTIVE_FISHER_FLAG_FEAT_HARD) {
		  /* drop features below threshold probabilities */
		  numValidData = 0;
		  for (i_d = 0; i_d < (signed)numData; ++ i_d) {
			  vl_index best = 0 ;
			  TYPE bestValue = posteriors[i_d * numClusters] ;
			  for (i_cl = 1 ; i_cl < (signed)numClusters; ++ i_cl) {
				TYPE p = posteriors[i_cl + i_d * numClusters] ;
				if (p > bestValue) {
				  bestValue = p ;
				  best = i_cl ;
				}
			  }
			  if (data_w[best] > data_thresh) {
				  isValidData[i_d] = true;
				  ++ numValidData;
			  } else isValidData[i_d] = false;
		  }
	  } else if (flags & SELECTIVE_FISHER_FLAG_FEAT_SOFT) {
		  numValidData = 0;
		  for (i_d = 0; i_d < (signed)numData; ++ i_d) {
			  TYPE value = 0;
			  for (i_cl = 0; i_cl < (signed)numClusters; ++ i_cl)
				  value += data_w[i_cl] * posteriors[i_cl + i_d * numClusters];
			  if (value > data_thresh) {
				  isValidData[i_d] = true;
				  ++ numValidData;
			  } else isValidData[i_d] = false;
		  }
	  }
  }

  /* sparsify posterior assignments with the FAST option */
  if ((flags & VL_FISHER_FLAG_FAST)) {
    for(i_d = 0; i_d < (signed)numData; i_d++) {
		if ((flags & SELECTIVE_FISHER_FLAG_FEAT_REDUCE) && !isValidData[i_d])
			continue;
      /* find largest posterior assignment for datum i_d */
      vl_index best = -1;
      TYPE bestValue = -1;
      for (i_cl = 0 ; i_cl < (signed)numClusters; ++ i_cl) {
		  if ((flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE) && !isValidCluster[i_cl])
			  continue;
        TYPE p = posteriors[i_cl + i_d * numClusters] ;
        if (p > bestValue) {
          bestValue = p ;
          best = i_cl ;
        }
      }
	  assert(best >= 0);
	  /* make all posterior assignments zero but the best one */
	  /*memset(posteriors + i_d * numClusters, 0, sizeof(TYPE) * numClusters);*/
	  /*posteriors[best + i_d * numClusters] = (TYPE)1;*/
	  for (i_cl = 0 ; i_cl < (signed)numClusters; ++ i_cl) {
		posteriors[i_cl + i_d * numClusters] =
		(TYPE)(i_cl == best) ;
	  }
    }
  } else if (flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE) {
	  /* renormalize posteriors*/
	  for (i_d = 0; i_d < (signed)numData; ++ i_d) {
		  if ((flags & SELECTIVE_FISHER_FLAG_FEAT_REDUCE) && !isValidData[i_d])
			  continue;
		  double sum = 0;
		  for (i_cl = 0; i_cl < (signed)numClusters; ++ i_cl)
			  if (isValidCluster[i_cl])
				  sum += posteriors[i_cl + i_d * numClusters];
		  for (i_cl = 0; i_cl < (signed)numClusters; ++ i_cl)
			  if (isValidCluster[i_cl])
			  	posteriors[i_cl + i_d * numClusters] /= sum;
	  }
  }

*num_feats_used = numValidData;

#if defined(_OPENMP)
#pragma omp parallel for default(shared) private(i_cl, i_d, dim) num_threads(vl_get_max_threads()) reduction(+:numTerms)
#endif
  for(i_cl = 0; i_cl < (signed)numClusters; ++ i_cl) {
    TYPE uprefix;
    TYPE vprefix;

    if ((flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE) && !isValidCluster[i_cl])
        continue;

	int idx = (flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE) ? idxClusters[i_cl] : i_cl;
    TYPE * uk = enc + idx*dimension ;
    TYPE * vk = enc + idx*dimension + numValidClusters * dimension ;
    
    for(i_d = 0; i_d < (signed)numData; i_d++) {
		if ((flags & SELECTIVE_FISHER_FLAG_FEAT_REDUCE) && !isValidData[i_d])
			continue;
      TYPE p = posteriors[i_cl + i_d * numClusters] ;
      numTerms += 1;
	  if ((flags & VL_FISHER_FLAG_FAST) && p < 1e-9)
		  continue;
      for(dim = 0; dim < dimension; dim++) {
        TYPE diff = data[i_d*dimension + dim] - means[i_cl*dimension + dim] ;
        diff *= sqrtInvSigma[i_cl*dimension + dim] ;
        *(uk + dim) += p * diff ;
        *(vk + dim) += p * (diff * diff - 1);
      }
    }

    uprefix = 1/(numValidData*sqrt(priors[i_cl]));
    vprefix = 1/(numValidData*sqrt(2*priors[i_cl]));

    for(dim = 0; dim < dimension; dim++) {
      *(uk + dim) = *(uk + dim) * uprefix;
      *(vk + dim) = *(vk + dim) * vprefix;
    }
  }

  vl_free(posteriors);
  vl_free(sqrtInvSigma) ;
  vl_free(isValidData);
  vl_free(idxClusters);

  if (flags & VL_FISHER_FLAG_SQUARE_ROOT) {
    for(dim = 0; dim < 2 * dimension * numValidClusters ; dim++) {
      TYPE z = enc [dim] ;
      if (z >= 0) {
        enc[dim] = VL_XCAT(vl_sqrt_, SFX)(z) ;
      } else {
        enc[dim] = - VL_XCAT(vl_sqrt_, SFX)(- z) ;
      }
    }
  }

  if (flags & VL_FISHER_FLAG_NORMALIZED) {
    TYPE n = 0 ;
    for(dim = 0 ; dim < 2 * dimension * numValidClusters ; dim++) {
      TYPE z = enc [dim] ;
      n += z * z ;
    }
    n = VL_XCAT(vl_sqrt_, SFX)(n) ;
    n = VL_MAX(n, 1e-12) ;
    for(dim = 0 ; dim < 2 * dimension * numValidClusters ; dim++) {
      enc[dim] /= n ;
    }
  }

  return numTerms ;
}

#else
/* not ANG_FISHER_INSTANTIATING */

#ifndef __DOXYGEN__
#define FLT VL_TYPE_FLOAT
#define TYPE float
#define SFX f
#define ANG_FISHER_INSTANTIATING
#include "fisher_ang.c"

#define FLT VL_TYPE_DOUBLE
#define TYPE double
#define SFX d
#define ANG_FISHER_INSTANTIATING
#include "fisher_ang.c"
#endif

/* not ANG_FISHER_INSTANTIATING */
#endif

/* ================================================================ */
#ifndef ANG_FISHER_INSTANTIATING

/** @brief Fisher vector encoding of a set of vectors.
 ** @param dataType the type of the input data (::VL_TYPE_DOUBLE or ::VL_TYPE_FLOAT).
 ** @param enc Fisher vector (output).
 ** @param means Gaussian mixture means.
 ** @param dimension dimension of the data.
 ** @param numClusters number of Gaussians mixture components.
 ** @param covariances Gaussian mixture diagonal covariances.
 ** @param priors Gaussian mixture prior probabilities.
 ** @param data vectors to encode.
 ** @param numData number of vectors to encode.
 ** @param flags options.
 ** @return number of averaging operations.
 **
 ** @a means and @a covariances have @a dimension rows and @a numCluster columns.
 ** @a priors is a vector of size @a numCluster. @a data has @a dimension
 ** rows and @a numData columns. @a enc is a vecotr of size equal
 ** to twice the product of @a dimension and @a numClusters.
 ** All these vectors and matrices have the same class, as specified
 ** by @a dataType.
 **
 ** @a flag can be used to control several options:
 ** ::VL_FISHER_FLAG_SQUARE_ROOT, ::VL_FISHER_FLAG_NORMALIZED,
 ** ::VL_FISHER_FLAG_IMPROVED, and ::VL_FISHER_FLAG_FAST.
 **
 ** The function returns the number of averaging operations actually
 ** computed.  The upper bound is the number of input features by the
 ** number of GMM modes; however, in practice assignments are usually
 ** failry sparse, so this number is less. In particular, with the
 ** ::VL_FISHER_FLAG_FAST, this number should be equal to the number
 ** of input features only. This information can be used for
 ** diagnostic purposes.
 **
 ** @sa @ref fisher
 **/

VL_EXPORT vl_size
ang_fisher_encode
(void * enc, int* num_feats_used, vl_type dataType,
 void const * means, vl_size dimension, vl_size numClusters,
 bool const * isValidCluster,
 void const * covariances,
 void const * priors,
 void const * data,  vl_size numData,
 void const * data_w,
 float data_thresh,
 int flags
)
{
  switch(dataType) {
    case VL_TYPE_FLOAT:
      return _ang_fisher_encode_f
      ((float *) enc,
              (int *) num_feats_used,
       (float const *) means, dimension, numClusters,
	   isValidCluster,
       (float const *) covariances,
       (float const *) priors,
       (float const *) data, numData,
	   (float const *) data_w,
	   data_thresh,
       flags);
    case VL_TYPE_DOUBLE:
      return _ang_fisher_encode_d
      ((double *) enc,
              (int *) num_feats_used,
       (double const *) means, dimension, numClusters,
	   isValidCluster,
       (double const *) covariances,
       (double const *) priors,
       (double const *) data, numData,
	   (double const *) data_w,
	   data_thresh,
       flags);
      break;
    default:
      abort();
  }
}
/* not ANG_FISHER_INSTANTIATING */
#endif

#undef SFX
#undef TYPE
#undef FLT
#undef ANG_FISHER_INSTANTIATING

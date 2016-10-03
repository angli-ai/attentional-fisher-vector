/** @file   vl_fisher.c
 ** @brief  vl_fisher MEX definition.
 ** @author Andrea Vedaldi
 ** @author David Novotny
 **/

/*
Copyright (C) 2007-12 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "fisher_ang.h"
#include <mexutils.h>
#include <string.h>
#include <stdio.h>

enum {
  opt_verbose,
  opt_normalized,
  opt_square_root,
  opt_improved,
  opt_fast,
  opt_feat_hard,
  opt_feat_soft,
  opt_feat_lsvm,
  opt_gmm_reduce,
  opt_feat_reduce,
  opt_small_pst_reduce
} ;

vlmxOption  options [] = {
  {"Verbose",             0,   opt_verbose                  },
  {"Normalized",          0,   opt_normalized               },
  {"SquareRoot",          0,   opt_square_root              },
  {"Improved",            0,   opt_improved                 },
  {"Fast",                0,   opt_fast                     },
  {"feathard",            0,   opt_feat_hard	},
  {"featsoft",	0, 	opt_feat_soft},
  {"featlsvm",	0,	opt_feat_lsvm},
  {"gmmreduce", 0, 	opt_gmm_reduce},
  {"smallpstreduce", 0, opt_small_pst_reduce},
} ;

/* driver */
void
mexFunction (int nout VL_UNUSED, mxArray * out[], int nin, const mxArray * in[])
{
  enum {IN_DATA = 0, IN_MEANS, IN_COVARIANCES, IN_PRIORS, IN_FEATS_W, IN_FEATS_THRESH, IN_GMM_W, IN_GMM_THRESH, IN_END} ;
  enum {OUT_ENC, OUT_NUM} ;

  int opt ;
  int next = IN_END ;
  mxArray const  *optarg ;

  vl_size numClusters = 10;
  vl_size dimension ;
  vl_size numData ;
  int flags = 0 ;

  void * covariances = NULL;
  void * means = NULL;
  void * priors = NULL;
  void * data = NULL ;
  void * feats_w = NULL;
  void * feats_b = NULL;
  void * gmm_w = NULL;
  vl_size numTerms ;

  int verbosity = 0 ;

  vl_type dataType ;
  mxClassID classID ;

  VL_USE_MATLAB_ENV ;

  /* -----------------------------------------------------------------
   *                                               Check the arguments
   * -------------------------------------------------------------- */

  if (nin < 9) {
    vlmxError (vlmxErrInvalidArgument,
               "At least 9 arguments required.");
  }
  if (nout > 2) {
    vlmxError (vlmxErrInvalidArgument,
               "At most two output argument.");
  }

  classID = mxGetClassID (IN(DATA)) ;
  switch (classID) {
    case mxSINGLE_CLASS: dataType = VL_TYPE_FLOAT ; break ;
    case mxDOUBLE_CLASS: dataType = VL_TYPE_DOUBLE ; break ;
    default:
      vlmxError (vlmxErrInvalidArgument,
                 "DATA is neither of class SINGLE or DOUBLE.") ;
  }

  if (mxGetClassID (IN(MEANS)) != classID) {
    vlmxError(vlmxErrInvalidArgument, "MEANS is not of the same class as DATA.") ;
  }
  if (mxGetClassID (IN(COVARIANCES)) != classID) {
    vlmxError(vlmxErrInvalidArgument, "COVARIANCES is not of the same class as DATA.") ;
  }
  if (mxGetClassID (IN(PRIORS)) != classID) {
    vlmxError(vlmxErrInvalidArgument, "PRIORS is not of the same class as DATA.") ;
  }
  if (mxGetClassID (IN(FEATS_W)) != classID) {
    vlmxError(vlmxErrInvalidArgument, "FEATS_W is not of the same class as DATA.") ;
  }
  if (mxGetClassID (IN(GMM_W)) != classID) {
    vlmxError(vlmxErrInvalidArgument, "GMM_W is not of the same class as DATA.") ;
  }

  dimension = mxGetM (IN(DATA)) ;
  numData = mxGetN (IN(DATA)) ;
  numClusters = mxGetN (IN(MEANS)) ;

  if (dimension == 0) {
    vlmxError (vlmxErrInvalidArgument, "SIZE(DATA,1) is zero.") ;
  }
  if (!vlmxIsMatrix(IN(MEANS), dimension, numClusters)) {
    vlmxError (vlmxErrInvalidArgument, "MEANS is not a matrix or does not have the correct size.") ;
  }
  if (!vlmxIsMatrix(IN(COVARIANCES), dimension, numClusters)) {
    vlmxError (vlmxErrInvalidArgument, "COVARIANCES is not a matrix or does not have the correct size.") ;
  }
  if (!vlmxIsVector(IN(PRIORS), numClusters)) {
    vlmxError (vlmxErrInvalidArgument, "PRIORS is not a vector or does not have the correct size.") ;
  }
  if (!vlmxIsMatrix(IN(DATA), dimension, numData)) {
    vlmxError (vlmxErrInvalidArgument, "DATA is not a matrix or does not have the correct size.") ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose : ++ verbosity ; break ;
      case opt_normalized: flags |= VL_FISHER_FLAG_NORMALIZED ; break ;
      case opt_square_root: flags |= VL_FISHER_FLAG_SQUARE_ROOT ; break ;
      case opt_improved: flags |= VL_FISHER_FLAG_IMPROVED ; break ;
      case opt_fast: flags |= VL_FISHER_FLAG_FAST ; break ;
	  case opt_feat_reduce: flags |= SELECTIVE_FISHER_FLAG_FEAT_REDUCE ; break;
	  case opt_feat_hard: flags |= SELECTIVE_FISHER_FLAG_FEAT_HARD ; break ;
	  case opt_feat_soft: flags |= SELECTIVE_FISHER_FLAG_FEAT_SOFT ; break ;
	  case opt_feat_lsvm: flags |= SELECTIVE_FISHER_FLAG_FEAT_LSVM ; break ;
	  case opt_gmm_reduce: flags |= SELECTIVE_FISHER_FLAG_GMM_REDUCE ; break ;
      default : abort() ;
    }
  }

  float feats_thresh = 0.5;
  float gmm_thresh = 0.5;
  if (flags & SELECTIVE_FISHER_FLAG_FEAT_REDUCE) {
	  if (!vlmxIsVector(IN(FEATS_W), numClusters)) {
		vlmxError (vlmxErrInvalidArgument, "FEATS_W is not a vector or does not have the correct size.") ;
	  }
	  if (!vlmxIsVector(IN(FEATS_THRESH), 1)) {
		vlmxError (vlmxErrInvalidArgument, "FEATS_THRESH is not a vector or does not have the correct size.") ;
	  }
	  switch (mxGetClassID(IN(FEATS_THRESH))) {
		case mxSINGLE_CLASS: feats_thresh = *(float*)mxGetPr(IN(FEATS_THRESH)); break ;
		case mxDOUBLE_CLASS: feats_thresh = *(double*)mxGetPr(IN(FEATS_THRESH)); break ;
		default: mexPrintf("feats_thresh type error\n");
	  }
  }
  if (flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE) {
	  if (!vlmxIsVector(IN(GMM_W), numClusters)) {
		vlmxError (vlmxErrInvalidArgument, "GMM_W is not a vector or does not have the correct size.") ;
	  }
	  if (!vlmxIsVector(IN(GMM_THRESH), 1)) {
		vlmxError (vlmxErrInvalidArgument, "GMM_THRESH is not a vector or does not have the correct size.") ;
	  }
	  switch (mxGetClassID(IN(GMM_THRESH))) {
		case mxSINGLE_CLASS: gmm_thresh = *(float*)mxGetPr(IN(GMM_THRESH)); break ;
		case mxDOUBLE_CLASS: gmm_thresh = *(double*)mxGetPr(IN(GMM_THRESH)); break ;
		default: mexPrintf("gmm_thresh type error\n");
	  }
  }
  /* -----------------------------------------------------------------
   *                                                        Do the job
   * -------------------------------------------------------------- */

  data = mxGetPr(IN(DATA)) ;
  means = mxGetPr(IN(MEANS)) ;
  covariances = mxGetPr(IN(COVARIANCES)) ;
  priors = mxGetPr(IN(PRIORS)) ;
  feats_w = mxGetPr(IN(FEATS_W)) ;
  gmm_w = mxGetPr(IN(GMM_W));

  int numValidClusters = numClusters;

  bool* isValidCluster = 0;
  int i;

  if (flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE) {
	  numValidClusters = 0;
	  isValidCluster = vl_malloc(sizeof(bool) * numClusters);
	  memset(isValidCluster, 0, sizeof(bool) * numClusters);
	  for (i = 0; i < numClusters; ++ i)
		  if (dataType == VL_TYPE_FLOAT) {
			  if (((float*)gmm_w)[i] > gmm_thresh) {
				  ++ numValidClusters;
				  isValidCluster[i] = true;
			  }
		  } else {
			  if (((double*)gmm_w)[i] > gmm_thresh) {
				  isValidCluster[i] = true;
				  ++ numValidClusters;
			  }
		  }
  }

  if (verbosity) {
    mexPrintf("selective_fisher: num data: %d\n", numData) ;
    mexPrintf("selective_fisher: num clusters: %d\n", numClusters) ;
    mexPrintf("selective_fisher: data dimension: %d\n", dimension) ;
	mexPrintf("selective_fisher: num valid clusters: %d\n", numValidClusters);
    mexPrintf("vl_fisher: code dimension: %d\n", numClusters * dimension) ;
    mexPrintf("vl_fisher: square root: %s\n", VL_YESNO(flags & VL_FISHER_FLAG_SQUARE_ROOT)) ;
    mexPrintf("vl_fisher: normalized: %s\n", VL_YESNO(flags & VL_FISHER_FLAG_NORMALIZED)) ;
    mexPrintf("vl_fisher: fast: %s\n", VL_YESNO(flags & VL_FISHER_FLAG_FAST)) ;
    mexPrintf("vl_fisher: feat_hard: %s\n", VL_YESNO(flags & SELECTIVE_FISHER_FLAG_FEAT_HARD)) ;
    mexPrintf("vl_fisher: feat_soft: %s\n", VL_YESNO(flags & SELECTIVE_FISHER_FLAG_FEAT_SOFT)) ;
    mexPrintf("vl_fisher: feat_lsvm: %s\n", VL_YESNO(flags & SELECTIVE_FISHER_FLAG_FEAT_LSVM)) ;
    mexPrintf("vl_fisher: gmm_reduce: %s\n", VL_YESNO(flags & SELECTIVE_FISHER_FLAG_GMM_REDUCE)) ;
    mexPrintf("vl_fisher: small_pst_reduce: %s\n", VL_YESNO(flags & SELECTIVE_FISHER_FLAG_SMALL_PST_REDUCE)) ;
#if defined(_OPENMP)
	mexPrintf("openmp: yes, #threads = %d\n", vl_get_max_threads());
#else
	mexPrintf("openmp: no\n");
#endif
#if defined(NDEBUG)
	mexPrintf("release mode.\n");
#else
	mexPrintf("debug mode.(slow) \n");
#endif
  }

  /* -------------------------------------------------------------- */
  /*                                                       Encoding */
  /* -------------------------------------------------------------- */

  OUT(ENC) = mxCreateNumericMatrix (dimension * numValidClusters * 2, 1, classID, mxREAL) ;
  
  int numFeats = -1;

  numTerms = ang_fisher_encode (mxGetData(OUT(ENC)), &numFeats, dataType,
                               means, dimension, numClusters,
							   isValidCluster,
                               covariances,
                               priors,
                               data, numData,
							   feats_w, feats_thresh,
                               flags) ;
  
  OUT(NUM) = mxCreateDoubleScalar((double)numFeats);

  vl_free(isValidCluster);

  if (verbosity) {
    mexPrintf("vl_fisher: sparsity of assignments: %.2f%% (%d non-negligible assignments)\n",
              100.0 * (1.0 - (double)numTerms/((double)numData*(double)numClusters)),
              numTerms) ;
  }
}

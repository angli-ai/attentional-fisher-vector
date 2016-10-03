#include <vl/fisher.h>
#include <mexutils.h>
#include <string.h>
#include <stdio.h>
#include "gmm_ang.h"

enum {
	opt_hard,
	opt_soft
};

vlmxOption options [] = {
	{"hard", 0, opt_hard},
	{"soft", 0, opt_soft}
};

/* driver */
void
mexFunction (int nout VL_UNUSED, mxArray * out[], int nin, const mxArray * in[])
{
  enum {IN_DATA = 0, IN_MEANS, IN_COVARIANCES, IN_PRIORS, IN_FEAT_W, IN_END} ;
  enum {OUT_ENC} ;

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
  void * featw = NULL;
  vl_size numTerms ;

  int verbosity = 0 ;

  vl_type dataType ;
  mxClassID classID ;

  VL_USE_MATLAB_ENV ;

  /* -----------------------------------------------------------------
   *                                               Check the arguments
   * -------------------------------------------------------------- */

  if (nin < 4) {
    vlmxError (vlmxErrInvalidArgument,
               "At least four arguments required.");
  }
  if (nout > 2) {
    vlmxError (vlmxErrInvalidArgument,
               "At most one output argument.");
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
  if (mxGetClassID (IN(FEAT_W)) != classID) {
    vlmxError(vlmxErrInvalidArgument, "FEAT_W is not of the same class as DATA.") ;
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
  if (!vlmxIsVector(IN(FEAT_W), numClusters)) {
    vlmxError (vlmxErrInvalidArgument, "FEAT_W is not a vector or does not have the correct size.") ;
  }
  if (!vlmxIsMatrix(IN(DATA), dimension, numData)) {
    vlmxError (vlmxErrInvalidArgument, "DATA is not a matrix or does not have the correct size.") ;
  }

  enum {METHOD_HARD, METHOD_SOFT} method;
  method = METHOD_HARD;
  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_hard: method = METHOD_HARD; break;
	  case opt_soft: method = METHOD_SOFT; break;
      default : abort() ;
    }
  }
  data = mxGetPr(IN(DATA)) ;
  means = mxGetPr(IN(MEANS)) ;
  covariances = mxGetPr(IN(COVARIANCES)) ;
  priors = mxGetPr(IN(PRIORS)) ;
  featw = mxGetPr(IN(FEAT_W));

  OUT(ENC) = mxCreateNumericMatrix (1, numData, classID, mxREAL) ;

  if (method == METHOD_HARD) {
  if (dataType == VL_TYPE_FLOAT)
  	ang_get_hard_posteriors_score_f (mxGetData(OUT(ENC)), numClusters, numData, 
		  priors, means, dimension, covariances, data, featw);
  else ang_get_hard_posteriors_score_d (mxGetData(OUT(ENC)), numClusters, numData, 
		  priors, means, dimension, covariances, data, featw);
  } else if (method == METHOD_SOFT) {
  if (dataType == VL_TYPE_FLOAT)
  	ang_get_soft_posteriors_score_f(mxGetData(OUT(ENC)), numClusters, numData, 
		  priors, means, dimension, covariances, data, featw);
  else ang_get_soft_posteriors_score_d (mxGetData(OUT(ENC)), numClusters, numData, 
		  priors, means, dimension, covariances, data, featw);
  }
}

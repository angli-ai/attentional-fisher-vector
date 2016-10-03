/** @file gmm.h
 ** @brief GMM (@ref gmm)
 ** @author David Novotny
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2013 David Novotny and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef ANG_GMM_H
#define ANG_GMM_H

#include <stdbool.h>
#include <vl/kmeans.h>

VL_EXPORT double
vl_get_gmm_data_posteriors_f(float * posteriors,
                             vl_size numClusters,
                             vl_size numData,
                             float const * priors,
                             float const * means,
                             vl_size dimension,
                             float const * covariances,
                             float const * data) ;

VL_EXPORT double
vl_get_gmm_data_posteriors_d(double * posteriors,
                             vl_size numClusters,
                             vl_size numData,
                             double const * priors,
                             double const * means,
                             vl_size dimension,
                             double const * covariances,
                             double const * data) ;
VL_EXPORT void 
ang_get_gmm_data_posteriors_f(float * index,
                             vl_size numClusters,
                             vl_size numData,
                             float const * priors,
                             float const * means,
                             vl_size dimension,
                             float const * covariances,
                             float const * data) ;

VL_EXPORT void
ang_get_gmm_data_posteriors_d(double * index,
                             vl_size numClusters,
                             vl_size numData,
                             double const * priors,
                             double const * means,
                             vl_size dimension,
                             double const * covariances,
                             double const * data) ;

VL_EXPORT void 
ang_get_soft_posteriors_score_f(float * index,
                             vl_size numClusters,
                             vl_size numData,
                             float const * priors,
                             float const * means,
                             vl_size dimension,
                             float const * covariances,
                             float const * data,
							 float const * pst_w) ;

VL_EXPORT void
ang_get_soft_posteriors_score_d(double * index,
                             vl_size numClusters,
                             vl_size numData,
                             double const * priors,
                             double const * means,
                             vl_size dimension,
                             double const * covariances,
                             double const * data,
							 double const * pst_w) ;

VL_EXPORT void 
ang_get_hard_posteriors_score_f(float * index,
                             vl_size numClusters,
                             vl_size numData,
                             float const * priors,
                             float const * means,
                             vl_size dimension,
                             float const * covariances,
                             float const * data,
							 float const * pst_w) ;

VL_EXPORT void
ang_get_hard_posteriors_score_d(double * index,
                             vl_size numClusters,
                             vl_size numData,
                             double const * priors,
                             double const * means,
                             vl_size dimension,
                             double const * covariances,
                             double const * data,
							 double const * pst_w) ;
/* ANG_GMM_H */
#endif

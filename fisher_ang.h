#ifndef ANG_FISHER_H
#define ANG_FISHER_H

#include <vl/fisher.h>
#include <stdbool.h>

#define SELECTIVE_FISHER_FLAG_FEAT_HARD (0x1 << 3)
#define SELECTIVE_FISHER_FLAG_FEAT_SOFT (0x1 << 4)
#define SELECTIVE_FISHER_FLAG_FEAT_LSVM (0x1 << 5)
#define SELECTIVE_FISHER_FLAG_GMM_REDUCE (0x1 << 6)
#define SELECTIVE_FISHER_FLAG_FEAT_REDUCE (SELECTIVE_FISHER_FLAG_FEAT_HARD | SELECTIVE_FISHER_FLAG_FEAT_SOFT)
#define SELECTIVE_FISHER_FLAG_SMALL_PST_REDUCE (0x1 << 7)

vl_size ang_fisher_encode
(void * enc, int* num_feats_used, vl_type dataType,
 void const * means, vl_size dimension, vl_size numClusters,
 bool const * isValidCluster,
 void const * covariances,
 void const * priors,
 void const * data, vl_size numData,
 void const * data_w, float data_thresh,
 int flags) ;

#endif

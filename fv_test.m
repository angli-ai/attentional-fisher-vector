function fv_test

rng(1);

numFeatures = 5000 ;
dimension = 2 ;
data = rand(dimension,numFeatures) ;

numClusters = 30 ;
[means, covariances, priors] = vl_gmm(data, numClusters);

numDataToBeEncoded = 1000;
dataToBeEncoded = rand(dimension,numDataToBeEncoded);

weights = rand(numClusters, 1);
% weights = ones(numClusters, 1);

priors = priors / sum(priors);

feat_w = weights;
feat_thresh = 0.5;
gmm_w = weights;
gmm_thresh = 0.7;

tic
encoding_ang = fv(dataToBeEncoded, means, covariances, priors, ...
    feat_w, feat_thresh, gmm_w, gmm_thresh, 'feathard', 'gmmreduce');
time2 = toc;


tic
[posteriors, likelihood] = gmm_get_posteriors(dataToBeEncoded, means, covariances, priors);
[~, gid] = max(posteriors);
gid = int32(gid);
featw = weights(gid);
encoding_true = vl_fisher(dataToBeEncoded(:, featw > feat_thresh), ...
    means(:, feat_w > gmm_thresh), covariances(:, feat_w > gmm_thresh), ...
    priors(feat_w > gmm_thresh));
time1 = toc;

likelihood
figure(1); subplot(2, 1, 1); bar(encoding_true);
subplot(2, 1, 2); bar(encoding_ang);

whos encoding_true encoding_ang
fprintf(1, 'time-old = %f, time-mex = %f\n', time1, time2);
% assert(abs(norm(encoding_true)-1) < 1e-3, 'encoding true norm ~= 1');
% assert(abs(norm(encoding_ang)-1) < 1e-3, 'my encoding norm ~= 1');
fprintf(1, 'diff = %f\n', norm(encoding_true - encoding_ang));
assert(norm(encoding_true - encoding_ang) < 1e-3, 'two encoding not same');
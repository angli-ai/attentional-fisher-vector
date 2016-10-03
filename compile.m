% compile mex
vlfeat_root='/Users/ang/lib/vlfeat/';

mex('gmm_get_posteriors.c', 'gmm_ang.c', ...
    ['-I', vlfeat_root], ['-I', fullfile(vlfeat_root, 'toolbox')], ...
    fullfile(vlfeat_root, 'toolbox', 'mex', 'mexmaci64', 'libvl.dylib'));

mex('fv.c', 'fisher_ang.c', ...
    ['-I', vlfeat_root], ['-I', fullfile(vlfeat_root, 'toolbox')], ...
    fullfile(vlfeat_root, 'toolbox', 'mex', 'mexmaci64', 'libvl.dylib'));
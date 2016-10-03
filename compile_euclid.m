% compile mex
    if exist('/gleuclid/angli/aaproj/vlfeat', 'dir')
		fprintf(1, 'euclid+vlfeat\n');
        vlfeat_root = ('/gleuclid/angli/aaproj/vlfeat');
    elseif exist('/scratch0/tools/vlfeat', 'dir')
		fprintf(1, 'lsdwks16+vlfeat\n');
        vlfeat_root = ('/scratch0/tools/vlfeat');
    elseif exist('/export/lustre_1/angl/3rdparty/vlfeat', 'dir')
		fprintf(1, 'deepthought+vlfeat\n');
        vlfeat_root = ('/export/lustre_1/angl/3rdparty/vlfeat');
    elseif exist('/Users/ang/workspace/toolboxes/vlfeat-0.9.20', 'dir')
		fprintf(1, 'imac+vlfeat\n');
        vlfeat_root = ('/Users/ang/workspace/toolboxes/vlfeat-0.9.20');
    elseif exist('/Users/ang/lib/vlfeat', 'dir')
        fprintf(1, 'macbook+vlfeat\n');
        vlfeat_root = ('/Users/ang/lib/vlfeat');
    end

mex('vl_gmm_get_posteriors.c', 'COPTIMFLAGS="-O3 -DNDEBUG"', ...
    ['-I', vlfeat_root], ['-I', fullfile(vlfeat_root, 'toolbox')], ...
    fullfile(vlfeat_root, 'toolbox', 'mex', 'mexa64', 'libvl.so'));
mex('gmm_get_posteriors.c', 'gmm_ang.c', 'COPTIMFLAGS="-O3 -DNDEBUG"', ...
    ['-I', vlfeat_root], ['-I', fullfile(vlfeat_root, 'toolbox')], ...
    fullfile(vlfeat_root, 'toolbox', 'mex', 'mexa64', 'libvl.so'));

mex('fv.c', 'fisher_ang.c', 'COPTIMFLAGS="-O3 -DNDEBUG"', ...
    ['-I', vlfeat_root], ['-I', fullfile(vlfeat_root, 'toolbox')], ...
    fullfile(vlfeat_root, 'toolbox', 'mex', 'mexa64', 'libvl.so'));

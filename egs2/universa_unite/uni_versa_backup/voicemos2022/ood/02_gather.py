# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import os
from hashlib import md5

f = open('gather.scp', 'r').readlines()
kl = [x.strip() for x in f]

print(' ===== Gathering samples ===== \n')

cmd = 'mkdir -p gathered'
os.system(cmd)
print('gathering data, please wait.....')

c = 0
for y in ['2019']:
    c += 1
    print(str(c) + '/1')
    
    kp = [k for k in kl if k.split('-')[0] == 'BC' + y]
    v = '1'
    bd = 'blizzard/blizzard_wavs_and_scores_' + y + '_release_version_' + v + '/'
    for k in kp:
        t = k.split('-')[1]
        g = k.split('-')[2].split('_')[0]
        uid = k.split('-')[-1]
        if y == '2019':
            wd = bd + t + '/submission_directory/' + y + '/celebrity/wav/'

        sid = 'BC' + y + '-' + t
        sh = md5(sid[::-1].encode()).hexdigest()[0:5][::-1]
        fwn0 = 'BC' + y + '-' + t + '-' + uid
        fwn = fwn0.split('.')[0]
        uh = md5(fwn[::-1].encode()).hexdigest()[4:11][::-1]
        hn = 'sys' + sh + '-utt' + uh + '.wav'
        
        cmd = 'cp ' + wd + uid + ' gathered/' + hn
        os.system(cmd)

print('done')

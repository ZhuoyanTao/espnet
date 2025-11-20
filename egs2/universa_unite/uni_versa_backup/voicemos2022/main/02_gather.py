# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import os
from hashlib import md5

lines = [x.strip() for x in open('gather.scp', 'r').readlines()]

print(' ===== Gathering samples ===== \n')

cmd = 'mkdir -p gathered'
os.system(cmd)
print('gathering data, please wait.....')

c = 0
for y in ['2008', '2009', '2010', '2011', '2013', '2016']:
    c += 1
    print(str(c) + '/6')
    
    kp = [l for l in lines if l.split('-')[0] == 'BC' + y]
    if y == '2013':
        v = '2'
    else:
        v = '1'
    bd = 'blizzard/blizzard_wavs_and_scores_' + y + '_release_version_' + v + '/'
    for k in kp:
        t = k.split('-')[1]
        g = k.split('-')[2].split('_')[0]
        uid = k.split('-')[-1]
        if y == '2008':
            wd = bd + t + '/submission_directory/english/full/' + y + '/' + g + '/'
        elif y in ['2009', '2010']:
            wd = bd + t + '/submission_directory/english/EH1/' + y + '/' + g + '/wavs/'
        elif y in ['2011', '2016']:
            if y == '2016':
                gd = 'audiobook'
            else:
                gd = g
            wd = bd + t + '/submission_directory/' + y + '/' + gd + '/wav/'
        elif y == '2013':
            if t == 'B':
                tk = 'EH2-English'
            else:
                tk = 'EH1-English'
            if g == 'booksent':
                gd = 'audiobook_sentences'
            else:
                gd = g            
            wd = bd + t + '/submission_directory/' + y + '/' + tk + '/' + gd + '/wav/'


        sid = 'BC' + y + '-' + t
        sh = md5(sid[::-1].encode()).hexdigest()[0:5][::-1]
        fwn0 = 'BC' + y + '-' + t + '-' + uid
        fwn = fwn0.split('.')[0]
        uh = md5(fwn[::-1].encode()).hexdigest()[4:11][::-1]
        hn = 'sys' + sh + '-utt' + uh + '.wav'
        
        cmd = 'cp ' + wd + uid + ' gathered/' + hn
        os.system(cmd)

print('done')

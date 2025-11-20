# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import os

trainlst = [x.split(',')[0] for x in open('DATA/sets/train_mos_list.txt', 'r').readlines()]
vallst = [x.split(',')[0] for x in open('DATA/sets/val_mos_list.txt', 'r').readlines()]
testlst = [x.split(',')[0] for x in open('DATA/sets/test_mos_list.txt', 'r').readlines()]
keeplst = trainlst + vallst + testlst

print('\n ===== Downsampling ===== \n')

cmd = 'mkdir -p downsampled'
os.system(cmd)

for w in os.listdir('gathered'):
    cmd = 'sox gathered/' + w + ' -r 16000 -V1 downsampled/' + w
    os.system(cmd)

print('\n ===== Normalizing ===== \n')

print('Please wait a moment....')

## normalize
sv56script = 'sv56scripts/batch_normRMSE.sh'
cmd = sv56script + ' downsampled'
os.system(cmd)

## sort and rename
cmd = 'mkdir -p normalized'
print(cmd)
os.system(cmd)
wavs = [x for x in os.listdir('downsampled') if x.split('_')[-1] == 'norm.wav']
for w in wavs:
    outname = '_'.join(w.split('_')[0:-1]) + '.wav'
    cmd = 'mv downsampled/' + w + ' ' + 'normalized/' + outname
    os.system(cmd)

## fix the one bad wav file
cmd = '\cp silence.wav normalized/sys4bafa-uttc2e86f6.wav'
print(cmd)
os.system(cmd)

## copy all to DATA
cmd = 'cp normalized/* DATA/wav/'
print(cmd)
os.system(cmd)

print('\n ===== Sanity check ===== \n')

print('Checking.....')

missing_count = 0
for k in keeplst:
    if not os.path.isfile('DATA/wav/' + k):
        if missing_count < 5:
            print('MISSING: ' + k)
        elif missing_count == 5:
            print('....')
        missing_count += 1        

if missing_count > 0:
    exit()
else:
    print('all good')

print('\n ===== Cleaning up ===== \n')
    
cmd = 'rm -rf gathered downsampled normalized'
print(cmd)
os.system(cmd)

print('\n ===== Done ===== \n')

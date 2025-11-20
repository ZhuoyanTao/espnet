# ==============================================================================
# Copyright (c) 2022, Yamagishi Laboratory, National Institute of Informatics
# Author: Erica Cooper (ecooper@nii.ac.jp)
# All rights reserved.
# ==============================================================================

import os
import subprocess

yn = str(input('\nBy participating in the MOS Prediction Challenge and downloading this Blizzard Challenge data, you agree to abide by the terms of use of this data.  This data may NOT be redistributed.  For more information, see https://www.cstr.ed.ac.uk/projects/blizzard/data.html. \n Do you agree?  y/n > '))

if yn not in ['y', 'Y', 'yes']:
    print('You must agree to the terms to download the data and participate in the challenge.  Exiting.')
    exit()

print(' ===== Starting download of Blizzard samples ===== \n')

cmd = 'wget -P blizzard --continue https://data.cstr.ed.ac.uk/blizzard/wavs_and_scores/blizzard_wavs_and_scores_2019_release_version_1.tar.bz2'
print(cmd)
os.system(cmd)

## next: check checksums
print(' ===== Checking checksums ===== \n')

sums = {'2019' : 'd99e6b7a8f6ec9219eec0e75d209de61', }

for year in ['2019']:
    ver = '1'
    result = subprocess.run(['md5sum', 'blizzard/blizzard_wavs_and_scores_' + year + '_release_version_' + ver + '.tar.bz2'], stdout=subprocess.PIPE)
    checksum = result.stdout.decode('utf-8').split()[0]
    if checksum != sums[year]:
        print('UH OH: file for ' + year + ' may be corrupted.  Please delete the corrupted file and retry downloading it.')
        exit()
    else:
        print('File for ' + year + ' downloaded ok')

print(' ===== Extracting data ===== \n')

for year in ['2019']:
    ver = '1'
    cmd = 'tar -xf blizzard/blizzard_wavs_and_scores_' + year + '_release_version_' + ver + '.tar.bz2 --directory blizzard'
    if not os.path.exists('blizzard/blizzard_wavs_and_scores_' + year + '_release_version_' + ver):
        print(cmd)
        os.system(cmd)
    else:
        print(year + ' already extracted')

print(' ===== Cleaning up ===== \n')
cmd = 'rm blizzard/*.tar.bz2'
print(cmd)
os.system(cmd)

print(' ===== Done ===== \n')


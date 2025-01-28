import os
import sys
import numpy
from pyAudioAnalysis.audioSegmentation import labels_to_segments
numpy.set_printoptions(threshold=sys.maxsize)

import sys
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)
from common_functions import diarize
sys.path.remove(parent_directory)

DATASET_PATH = '/path/to/datasets/folder/This American Life - Test Set/'
TEST_PATH = os.path.join(DATASET_PATH, 'wav')

TEST_LABELS_PATH = os.path.join(DATASET_PATH, 'rttm')
OUTPUT_PATH_BASELINE = '/research_data/diarization/pyAudioAnalysis/This American Life/results/baseline/'
OUTPUT_PATH_EXACT = '/research_data/diarization/pyAudioAnalysis/This American Life/results/exact/'

def diarization(data_path, output_path, exactNum = False, gt_path = ""):
        
    for file in os.listdir(data_path):
        print('-' * 20, file.center(5), '-' * 20, flush=True)
        
        # Skip if already diarized
        if os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')):
            print('Skipping', file.split('.')[0] + '.rttm', flush = True)
            continue
        else:
            try:
                result = diarize(data_path, file, exactNum, gt_path)
                if result == None:
                    print("====> # SPEAKERS < 2 -- CANNOT DIARIZE!", flush=True)
                else:
                    timestamps, spk_ids = labels_to_segments(result[0], 0.1)

                    rttm_lines = []
                    for ts, id in zip(timestamps, spk_ids):
                        start_time = ts[0]
                        end_time = ts[1]
                        dur = float(end_time) - float(start_time)
                        
                        rttm_lines.append("SPEAKER file 1 %s %s <NA> <NA> %s <NA>\n" 
                                    % (start_time, str(dur), id))
                    # Save results to existing directory or create new one
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    with open(
                        os.path.join(output_path, file.split('.')[0] + '.rttm'), 'w'
                        ) as rttm:
                        rttm.writelines(rttm_lines)
            except Exception as e:
                print("====> ", e, flush=True)
    print("DONE!", flush=True)

#---------------------------- BASELINE ----------------------------

print('-' * 20, "BASELINE".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_BASELINE + 'test/')

#---------------------------- EXACT ----------------------------
print('-' * 20, "EXACT".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_EXACT + 'test/', exactNum = True, 
            gt_path = TEST_LABELS_PATH)

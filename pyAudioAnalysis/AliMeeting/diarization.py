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


DATASET_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/'
TEST_FAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/audio_dir/mono/')
TEST_NEAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/audio_dir')

TEST_FAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/rttm_dir')
TEST_NEAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/rttm_dir')

OUTPUT_PATH_BASELINE = '/research_data/pyAudioAnalysis/AliMeeting/results/baseline/'
OUTPUT_PATH_EXACT = '/research_data/pyAudioAnalysis/AliMeeting/results/exact/'

def diarization(data_path, output_path, exactNum = False, gt_path = "", 
                isFar = False):
        
    for file in os.listdir(data_path):
        print('-' * 20, file.center(5), '-' * 20, flush=True)
        
        # Skip if already diarized
        if os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')):
            print('Skipping', file.split('.')[0] + '.rttm', flush = True)
            continue
        else:
            try:
                result = diarize(data_path, file, exactNum, gt_path, isFar)
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
print('-' * 20, "BASELINE FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + 'far/', isFar = True)

print('-' * 20, "BASELINE NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + 'near/')

#---------------------------- EXACT ----------------------------
print('-' * 20, "EXACT FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + 'far/', exactNum = True, 
            gt_path = TEST_FAR_LABELS_PATH, isFar = True)

print('-' * 20, "EXACT NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + 'near/', exactNum = True, 
            gt_path = TEST_NEAR_LABELS_PATH)

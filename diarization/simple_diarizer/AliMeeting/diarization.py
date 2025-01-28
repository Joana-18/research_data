import os
import torch
import sys
from simple_diarizer.diarizer import Diarizer

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)
from common_functions import diarize
sys.path.remove(parent_directory)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_NAME = 'AliMeeting'
DATASET_PATH = f'/path/to/datasets/folder/{DATASET_NAME}/Test_Ali/'
TEST_FAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/audio_dir')
TEST_NEAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/audio_dir')

TEST_FAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/rttm_dir')
TEST_NEAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/rttm_dir')

BASE_DIR = f'/research_data/diarization/simple_diarizer/{DATASET_NAME}/'
OUTPUT_PATH_BASELINE = f'{BASE_DIR}/results/baseline/'
OUTPUT_PATH_EXACT = f'{BASE_DIR}/results/exact/'

# Embedding models
XVEC = 'xvec'
ECAPA = 'ecapa'
# Clustering models
AHC = 'ahc'
SC = 'sc'
    
def diarization(data_path, output_path, exactNum = False, gt_path = "", 
                embed = '' , cluster = '', isFar = False):
    pipeline = Diarizer(
        embed_model = embed, # 'xvec' and 'ecapa' supported
        cluster_method = cluster # 'ahc' and 'sc' supported
        )
    output_path += "" + embed + "_" + cluster
    for file in os.listdir(data_path):
        print('-' * 20, file.center(5), '-' * 20, flush=True)
        
        # Skip if already diarized
        if (os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')) 
            or '_converted' in file):
            print('Skipping', file.split('.')[0] + '.rttm', flush = True)
            continue
        else:
            try:
                result = diarize(data_path, file, pipeline, exactNum, gt_path, isFar)
                rttm_lines = []
                for seg in result:
                    start_time = seg['start']
                    end_time = seg['end']
                    dur = float(end_time) - float(start_time)
                    spk = seg['label']
                    
                    rttm_lines.append("SPEAKER file 1 %s %s <NA> <NA> %s <NA>\n" 
                                % (start_time, str(dur), spk))
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
print('-' * 20,  f"BASELINE {XVEC} {SC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + 'far/', isFar = True, 
            embed = XVEC, cluster = SC)

print('-' * 20,  f"BASELINE {ECAPA} {SC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + 'far/', isFar = True,
            embed = ECAPA, cluster = SC)

print('-' * 20,  f"BASELINE {XVEC} {AHC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + 'far/', isFar = True, 
            embed = XVEC, cluster = AHC)

print('-' * 20,  f"BASELINE {ECAPA} {AHC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + 'far/', isFar = True, 
            embed = ECAPA, cluster = AHC)

print('-' * 20,  f"BASELINE {XVEC} {SC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + 'near/', embed = XVEC, cluster = SC)

print('-' * 20,  f"BASELINE {ECAPA} {SC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + 'near/', 
            embed = ECAPA, cluster = SC)

print('-' * 20,  f"BASELINE {XVEC} {AHC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + 'near/', 
            embed = XVEC, cluster = AHC)

print('-' * 20,  f"BASELINE {ECAPA} {AHC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + 'near/', 
            embed = ECAPA, cluster = AHC)

#---------------------------- EXACT ----------------------------
print('-' * 20, f"EXACT {XVEC} {SC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + 'far/', exactNum = True, 
            gt_path = TEST_FAR_LABELS_PATH, isFar = True, embed = XVEC, cluster = SC)

print('-' * 20, f"EXACT {XVEC} {AHC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + 'far/', exactNum = True, 
            gt_path = TEST_FAR_LABELS_PATH, isFar = True, embed = XVEC, cluster = AHC)

print('-' * 20, f"EXACT {ECAPA} {SC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + 'far/', exactNum = True, 
            gt_path = TEST_FAR_LABELS_PATH, isFar = True, embed = ECAPA, cluster = SC)

print('-' * 20, f"EXACT {ECAPA} {AHC} FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + 'far/', exactNum = True, 
            gt_path = TEST_FAR_LABELS_PATH, isFar = True, embed = ECAPA, cluster = AHC)

print('-' * 20, f"EXACT {XVEC} {SC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + 'near/', exactNum = True, 
            gt_path = TEST_NEAR_LABELS_PATH, embed = XVEC, cluster = SC)

print('-' * 20, f"EXACT {XVEC} {AHC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + 'near/', exactNum = True, 
            gt_path = TEST_NEAR_LABELS_PATH, embed = XVEC, cluster = AHC)

print('-' * 20, f"EXACT {ECAPA} {SC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + 'near/', exactNum = True, 
            gt_path = TEST_NEAR_LABELS_PATH, embed = ECAPA, cluster = SC)

print('-' * 20, f"EXACT {ECAPA} {AHC} NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + 'near/', exactNum = True, 
            gt_path = TEST_NEAR_LABELS_PATH, embed = ECAPA, cluster = AHC)


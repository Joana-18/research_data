import os
import torch
import sys
from simple_diarizer.diarizer import Diarizer

parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_directory)
from common_functions import diarize
sys.path.remove(parent_directory)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_NAME = 'MSDWILD'
DATASET_PATH = f'/path/to/datasets/folder/{DATASET_NAME}/'
TEST_PATH_FEW = os.path.join(DATASET_PATH, 'wav/few')
TEST_PATH_MANY = os.path.join(DATASET_PATH, 'wav/many')

TEST_LABELS_PATH_FEW = os.path.join(DATASET_PATH, 'rttm/few')
TEST_LABELS_PATH_MANY = os.path.join(DATASET_PATH, 'rttm/many')
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
                embed = '' , cluster = ''):
    pipeline = Diarizer(
        embed_model = embed, # 'xvec' and 'ecapa' supported
        cluster_method = cluster # 'ahc' and 'sc' supported
        )
    output_path += "" + embed + "_" + cluster
    for file in os.listdir(data_path):
        print('-' * 20, file.center(5), '-' * 20, flush=True)
        
        # Skip if already diarized
        if os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')):
            print('Skipping', file.split('.')[0] + '.rttm', flush = True)
            continue
        else:
            result = diarize(data_path, file, pipeline, exactNum, gt_path)
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
    print("DONE!", flush=True)

#---------------------------- BASELINE ----------------------------

print('-' * 20, f"BASELINE {XVEC} {SC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_BASELINE + 'few/', 
            embed = XVEC, cluster = SC)

print('-' * 20, f"BASELINE {ECAPA} {SC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_BASELINE + 'few/', 
            embed = ECAPA, cluster = SC)

print('-' * 20, f"BASELINE {XVEC} {AHC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_BASELINE + 'few/', 
            embed = XVEC, cluster = AHC)

print('-' * 20, f"BASELINE {ECAPA} {AHC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_BASELINE + 'few/', 
            embed = ECAPA, cluster = AHC)

print('-' * 20, f"BASELINE {XVEC} {SC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_BASELINE + 'many/', 
            embed = XVEC, cluster = SC)

print('-' * 20, f"BASELINE {ECAPA} {SC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_BASELINE + 'many/', 
            embed = ECAPA, cluster = SC)

print('-' * 20, f"BASELINE {XVEC} {AHC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_BASELINE + 'many/', 
            embed = XVEC, cluster = AHC)

print('-' * 20, f"BASELINE {ECAPA} {AHC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_BASELINE + 'many/', 
            embed = ECAPA, cluster = AHC)

#---------------------------- EXACT ----------------------------
print('-' * 20, f"EXACT {XVEC} {SC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EXACT + 'few/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_FEW, embed = XVEC, cluster = SC)

print('-' * 20,  f"EXACT {ECAPA} {SC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EXACT + 'few/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_FEW, embed = ECAPA, cluster = SC)

print('-' * 20, f"EXACT {XVEC} {AHC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EXACT + 'few/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_FEW, embed = XVEC, cluster = AHC)

print('-' * 20,  f"EXACT {ECAPA} {AHC} FEW".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EXACT + 'few/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_FEW, embed = ECAPA, cluster = AHC)

print('-' * 20, f"EXACT {XVEC} {SC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EXACT + 'many/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_MANY, embed = XVEC, cluster = SC)

print('-' * 20, f"EXACT {ECAPA} {SC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EXACT + 'many/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_MANY, embed = ECAPA, cluster = SC)

print('-' * 20, f"EXACT {XVEC} {AHC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EXACT + 'many/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_MANY, embed = XVEC, cluster = AHC)

print('-' * 20, f"EXACT {ECAPA} {AHC} MANY".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EXACT + 'many/', exactNum = True, 
            gt_path = TEST_LABELS_PATH_MANY, embed = ECAPA, cluster = AHC)




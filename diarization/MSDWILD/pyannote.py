import os
from pyannote.audio import Pipeline
import torch

AUTH_TOKEN = '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'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = '/path/to/datasets/folder/MSDWILD/'
TEST_PATH_FEW = os.path.join(DATASET_PATH, 'wav/few')
TEST_PATH_MANY = os.path.join(DATASET_PATH, 'wav/many')

TEST_LABELS_PATH_FEW = os.path.join(DATASET_PATH, 'rttm/few')
TEST_LABELS_PATH_MANY = os.path.join(DATASET_PATH, 'rttm/many')

OUTPUT_PATH_EP = '/research_data/pyannote/MSDWILD/results/global/'
OUTPUT_PATH_BASELINE = '/research_data/pyannote/MSDWILD/results/baseline/'
OUTPUT_PATH_EXACT = '/research_data/pyannote/MSDWILD/results/exact/'
PY2_1 =  "2.1/"
PY3_1 =  "3.1/"

v21 = "speaker-diarization@2.1"
v31 = "speaker-diarization-3.1"

def diarization(data_path, output_path, min_speakers = None, max_speakers = None, 
                version = v21, globalP = False, exactNum = False, gt_path = ""):
    # instantiate the pipeline
    pipeline = Pipeline.from_pretrained("pyannote/" + version,
        use_auth_token = AUTH_TOKEN)
    pipeline.to(DEVICE)
    
    for file in os.listdir(data_path):
        print('-' * 20, file.center(5), '-' * 20, flush=True)
        
        if os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')):
            print('Skipping', file.split('.')[0] + '.rttm', flush = True)
            continue
        else:
            file_path = os.path.join(data_path, file)
            if globalP:
                diarization = pipeline(file_path, 
                                       min_speakers = min_speakers, 
                                       max_speakers = max_speakers)
            elif exactNum:
                gt_file = file.split('.')[0] + '.rttm'
                num = getNumberSpeakers(gt_file, gt_path)
                diarization = pipeline(file_path, num_speakers = num)
            else:
                diarization = pipeline(file_path)

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(
                os.path.join(output_path, file.split('.')[0] + '.rttm'), 'w'
                ) as rttm:
                diarization.write_rttm(rttm)
    print("DONE!", flush=True)


def getNumberSpeakers(file, gt_path):
    speakers = []
    rttm_file_path = os.path.join(gt_path, file)
    
    with open(rttm_file_path, 'r') as rttm_file:
        for line in rttm_file:
            parts = line.strip().split()  # Split each line into parts
            
            speaker_label = parts[7]
            if speaker_label not in speakers:
                speakers.append(speaker_label)
    
    return len(speakers)
    
#---------------------------- GLOBAL ----------------------------
print('-' * 20, "GLOBAL FEW 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EP + PY2_1 + 'few/', 2, 4, v21, globalP = True)

print('-' * 20, "GLOBAL FEW 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EP + PY3_1 + 'few/', 2, 4, v31, globalP = True)

print('-' * 20, "GLOBAL MANY 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EP + PY2_1 + 'many/', 3, 9, v21, globalP = True)

print('-' * 20, "GLOBAL MANY 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EP + PY3_1 + 'many/', 3, 9, v31, globalP = True)


#---------------------------- BASELINE ----------------------------

print('-' * 20, "BASELINE FEW 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_BASELINE + PY2_1 + 'few/', version = v21)

print('-' * 20, "BASELINE FEW 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_BASELINE + PY3_1 + 'few/', version = v31)

print('-' * 20, "BASELINE MANY 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_BASELINE + PY2_1 + 'many/', version = v21)

print('-' * 20, "BASELINE MANY 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_BASELINE + PY3_1 + 'many/', version = v31)


#---------------------------- EXACT NUM ----------------------------
print('-' * 20, "EXACT NUM FEW 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EXACT + PY2_1 + 'few/', version = v21, exactNum = True, gt_path = TEST_LABELS_PATH_FEW)

print('-' * 20, "EXACT NUM FEW 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_FEW, OUTPUT_PATH_EXACT + PY3_1 + 'few/', version = v31, exactNum = True, gt_path = TEST_LABELS_PATH_FEW)

print('-' * 20, "EXACT NUM MANY 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EXACT + PY2_1 + 'many/', version = v21, exactNum = True, gt_path = TEST_LABELS_PATH_MANY)

print('-' * 20, "EXACT NUM MANY 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_PATH_MANY, OUTPUT_PATH_EXACT + PY3_1 + 'many/', version = v31, exactNum = True, gt_path = TEST_LABELS_PATH_MANY)

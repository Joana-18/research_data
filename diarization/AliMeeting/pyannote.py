import os
from pyannote.audio import Pipeline
import torch

AUTH_TOKEN = 'your_auth_token'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/'
TEST_FAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/audio_dir')
TEST_NEAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/audio_dir')

TEST_FAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/rttm_dir')
TEST_NEAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/rttm_dir')

OUTPUT_PATH_EP = '/research_data/pyannote/AliMeeting/results/global/'
OUTPUT_PATH_BASELINE = '/research_data/pyannote/AliMeeting/results/baseline/'
OUTPUT_PATH_EXACT = '/research_data/pyannote/AliMeeting/results/exact/'
PY2_1 =  "2.1/"
PY3_1 =  "3.1/"

v21 = "speaker-diarization@2.1"
v31 = "speaker-diarization-3.1"

def diarization(data_path, output_path, min_speakers = None, max_speakers = None, 
                version = v21, globalP = False, exactNum = False, gt_path = "", 
                isFar = True):
    # instantiate the pipeline
    pipeline = Pipeline.from_pretrained("pyannote/" + version,
        use_auth_token = AUTH_TOKEN)
    pipeline.to(DEVICE)
    for file in os.listdir(data_path):
        print('-' * 20, file.center(5), '-' * 20, flush=True)
        
        if (os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')) 
            or '_converted' in file):
            print('Skipping', file.split('.')[0] + '.rttm', flush = True)
            continue
        else:
            
            file_path = os.path.join(data_path, file)
            if globalP:
                diarization = pipeline(file_path, 
                                       min_speakers = min_speakers, 
                                       max_speakers = max_speakers)
            elif exactNum:
                if isFar:
                    file_name = file.split('_') 
                    gt_file = file_name[0] + "_" + file_name[1] + '.rttm'
                else:
                    file_name = file.split('.')[0] 
                    gt_file = file_name + '.rttm'
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
    
# ---------------------------- GLOBAL ----------------------------
print('-' * 20, "GLOBAL FAR 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EP + PY2_1 + 'far/', 2, 4, v21, 
            globalP = True)

print('-' * 20, "GLOBAL FAR 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EP + PY3_1 + 'far/', 2, 4, v31, 
            globalP = True)

print('-' * 20, "GLOBAL NEAR 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EP + PY2_1 + 'near/', 1, 1, v21, 
            globalP = True, isFar = False)

print('-' * 20, "GLOBAL NEAR 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EP + PY3_1 + 'near/', 1, 1, v31, 
            globalP = True, isFar = False)


#---------------------------- BASELINE ----------------------------

print('-' * 20, "BASELINE FAR 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + PY2_1 + 'far/', version = v21)

print('-' * 20, "BASELINE FAR 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + PY3_1 + 'far/', version =  v31)

print('-' * 20, "BASELINE NEAR 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + PY2_1 + 'near/', version = v21, 
            isFar = False)

print('-' * 20, "BASELINE NEAR 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + PY3_1 + 'near/', version = v31,
            isFar = False)


#---------------------------- EXACT NUM ----------------------------
print('-' * 20, "EXACT NUM FAR 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + PY2_1 + 'far/', version = v21, 
            exactNum = True, gt_path = TEST_FAR_LABELS_PATH)

print('-' * 20, "EXACT NUM FAR 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + PY3_1 + 'far/', version = v31, 
            exactNum = True, gt_path = TEST_FAR_LABELS_PATH)

print('-' * 20, "EXACT NUM NEAR 2.1".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + PY2_1 + 'near/', version = v21, 
            exactNum = True, gt_path = TEST_NEAR_LABELS_PATH, isFar = False)

print('-' * 20, "EXACT NUM NEAR 3.1".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + PY3_1 + 'near/', version = v31, 
            exactNum = True, gt_path = TEST_NEAR_LABELS_PATH, isFar = False)
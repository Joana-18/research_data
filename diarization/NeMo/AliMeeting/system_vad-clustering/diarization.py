import os
import torch
import sys
import numpy

import wget
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import time

import json
numpy.set_printoptions(threshold=sys.maxsize)

import sys
parent_directory = '/research_data/diarization/NeMo'
sys.path.append(parent_directory)
from common_functions import create_json_object
sys.path.remove(parent_directory)


# ==============================================================================
# ================================ GLOBAL PARAMS =============================== 
# ==============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_NAME = 'AliMeeting'
DATASET_PATH = f'/path/to/datasets/folder/{DATASET_NAME}/Test_Ali/'
TEST_FAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/audio_dir/mono/')
TEST_NEAR_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/audio_dir')

TEST_FAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_far/rttm_dir')
TEST_NEAR_LABELS_PATH = os.path.join(DATASET_PATH, 'Test_Ali_near/rttm_dir')

BASE_DIR = f'/research_data/diarization/NeMo/{DATASET_NAME}/'
CUR_DIR = f'{BASE_DIR}/system_vad-clustering'
OUTPUT_PATH_BASELINE = f'{CUR_DIR}/results/baseline/'
OUTPUT_PATH_EXACT = f'{CUR_DIR}/results/exact/'

# ==============================================================================
# ================================ MODEL CONFIG ================================ 
# ==============================================================================

# Source: https://github.com/NVIDIA/NeMo/blob/stable/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
DIR = '/research_data/diarization/NeMo'
DOMAIN_TYPE = "telephonic" # Can be meeting or telephonic based on domain type of the audio file
CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
MODEL_CONFIG = os.path.join(DIR, CONFIG_FILE_NAME)
if not os.path.exists(MODEL_CONFIG):
    config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = wget.download(config_url, DIR)

config = OmegaConf.load(MODEL_CONFIG)

pretrained_vad = 'vad_multilingual_marblenet'
pretrained_speaker_model = 'titanet_large'

config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config


config.diarizer.vad.model_path = pretrained_vad
config.verbose = False
config.diarizer.msdd_model.parameters.use_speaker_model_from_ckpt = False
config.device = 'cuda'



# ==============================================================================
# ==============================================================================



def diarization(data_path, output_path, exactNum = False, gt_path = "", 
                isFar = False, set_name = ''):
    if isFar:
        config.diarizer.clustering.parameters.max_num_speakers = 4
    else:
        config.diarizer.clustering.parameters.max_num_speakers = 1
    # Directory to store intermediate files and prediction outputs
    if exactNum:
        manifest_filepath = f'{BASE_DIR}/input_manifest_exact_{set_name}.json'
        config.diarizer.manifest_filepath =manifest_filepath
        config.diarizer.clustering.parameters.oracle_num_speakers=True
    else:
        manifest_filepath = f'{BASE_DIR}/input_manifest_baseline_{set_name}.json'
        config.diarizer.manifest_filepath = manifest_filepath
    config.diarizer.out_dir = output_path

    print(OmegaConf.to_yaml(config))
    start = time.time()
    system_vad_msdd_model = ClusteringDiarizer(cfg = config)   
    
    if not os.path.exists(manifest_filepath):
        for file in os.listdir(data_path):
            print('-' * 20, file.center(5), '-' * 20, flush=True)
            # Skip if already diarized
            if os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')):
                print('Skipping', file.split('.')[0] + '.rttm', flush = True)
                continue
            else:
                meta_info = create_json_object(data_path, file, gt_path, 
                                            exactNum, isFar)            
                
            with open(manifest_filepath, 'a+') as fp:
                json.dump(meta_info, fp)
                fp.write('\n')
    
    system_vad_msdd_model.diarize()
    print("DONE!", flush=True)
    end = time.time()
    print("ELAPSED TIME: ", end - start)


#---------------------------- BASELINE ----------------------------
print('-' * 20, "BASELINE FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_BASELINE + 'far_fixed/', isFar = True, set_name = 'far')

print('-' * 20, "BASELINE NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_BASELINE + 'near_fixed/', set_name = 'near')

#---------------------------- EXACT ----------------------------
print('-' * 20, "EXACT FAR".center(5), '-' * 20, flush=True)
diarization(TEST_FAR_PATH, OUTPUT_PATH_EXACT + 'far_fixed/', exactNum = True, 
            gt_path = TEST_FAR_LABELS_PATH, isFar = True, set_name = 'far')

print('-' * 20, "EXACT NEAR".center(5), '-' * 20, flush=True)
diarization(TEST_NEAR_PATH, OUTPUT_PATH_EXACT + 'near_fixed/', exactNum = True, 
            gt_path = TEST_NEAR_LABELS_PATH, set_name = 'near')
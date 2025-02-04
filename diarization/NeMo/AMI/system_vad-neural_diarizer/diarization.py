import os
import torch
import sys
import numpy

import wget
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

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

DATASET_NAME = 'AMI'
DATASET_PATH = f'/path/to/datasets/folder/{DATASET_NAME}/headset-mix/'
TEST_PATH = os.path.join(DATASET_PATH, 'test/')

TEST_LABELS_PATH = os.path.join(DATASET_PATH, 'Labels/test')
BASE_DIR = f'/research_data/diarization/NeMo/{DATASET_NAME}/'
CUR_DIR = f'{BASE_DIR}/system_vad-neural_diarizer'
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
config.diarizer.msdd_model.model_path = 'diar_msdd_telephonic'

config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0] # Evaluate with T=0.7 and T=1.0
config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
config.diarizer.clustering.parameters.max_num_speakers = 4

config.diarizer.vad.model_path = pretrained_vad

config.verbose = False
config.diarizer.msdd_model.parameters.use_speaker_model_from_ckpt = False
config.device = 'cuda'




# ==============================================================================
# ==============================================================================



def diarization(data_path, output_path, exactNum = False, gt_path = "", 
                isFar = False):
    # Directory to store intermediate files and prediction outputs
    if exactNum:
        manifest_filepath = f'{BASE_DIR}/input_manifest_exact.json'
        config.diarizer.manifest_filepath =manifest_filepath
        config.diarizer.clustering.parameters.oracle_num_speakers=True
    else:
        manifest_filepath = f'{BASE_DIR}/input_manifest_baseline.json'
        config.diarizer.manifest_filepath = manifest_filepath
    config.diarizer.out_dir = output_path

    print(OmegaConf.to_yaml(config))
    system_vad_msdd_model = NeuralDiarizer(cfg = config)   
    
    if not os.path.exists(manifest_filepath):
        for dir in os.listdir(data_path):
            dir_path = data_path + dir + '/audio/'
            for file in os.listdir(dir_path):
                print('-' * 20, file.center(5), '-' * 20, flush=True)
                # Skip if already diarized
                if os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')):
                    print('Skipping', file.split('.')[0] + '.rttm', flush = True)
                    continue
                else:
                    meta_info = create_json_object(dir_path, file, gt_path, 
                                                exactNum, isFar)            
                    
                with open(manifest_filepath, 'a+') as fp:
                    json.dump(meta_info, fp)
                    fp.write('\n')
    
    system_vad_msdd_model.diarize()
    print("DONE!", flush=True)


#---------------------------- BASELINE ----------------------------

print('-' * 20, "BASELINE".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_BASELINE + 'test_fixed/', 
            gt_path = TEST_LABELS_PATH)


#---------------------------- EXACT ----------------------------
print('-' * 20, "EXACT".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_EXACT + 'test_fixed/', exactNum = True, 
            gt_path = TEST_LABELS_PATH)
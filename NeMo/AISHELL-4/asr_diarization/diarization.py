import os
import torch
import sys
import numpy

import wget
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR

import json
numpy.set_printoptions(threshold=sys.maxsize)

import sys
parent_directory = '/research_data/NeMo'
sys.path.append(parent_directory)
from common_functions import create_json_object
sys.path.remove(parent_directory)


# ==============================================================================
# ================================ GLOBAL PARAMS =============================== 
# ==============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_NAME = 'AISHELL-4'
DATASET_PATH = f'/datasets/{DATASET_NAME}/'
TEST_PATH = f'/path/to/datasets/folder/{DATASET_NAME}_wav/mono/'

TEST_LABELS_PATH = os.path.join(DATASET_PATH, 'test/TextGrid')

BASE_DIR = f'/research_data/NeMo/{DATASET_NAME}/'
CUR_DIR = f'{BASE_DIR}/asr_diarization'
OUTPUT_PATH_BASELINE = f'{CUR_DIR}/results/baseline/'
OUTPUT_PATH_EXACT = f'{CUR_DIR}/results/exact/'

# ==============================================================================
# ================================ MODEL CONFIG ================================ 
# ==============================================================================

# Source: https://github.com/NVIDIA/NeMo/blob/stable/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb
DIR = '/research_data/NeMo'
DOMAIN_TYPE = "telephonic" # Can be meeting or telephonic based on domain type of the audio file
CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
MODEL_CONFIG = os.path.join(DIR, CONFIG_FILE_NAME)
if not os.path.exists(MODEL_CONFIG):
    config_url = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = wget.download(config_url, DIR)

config = OmegaConf.load(MODEL_CONFIG)

pretrained_speaker_model='titanet_large'
config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model

# Using Neural VAD and Conformer ASR 
config.diarizer.vad.model_path = 'vad_multilingual_marblenet'
config.diarizer.asr.model_path = 'stt_en_conformer_ctc_large'
config.diarizer.oracle_vad = False # ----> Not using oracle VAD 
config.diarizer.clustering.parameters.max_num_speakers = 7

config.verbose = False
config.diarizer.msdd_model.parameters.use_speaker_model_from_ckpt = False
config.device = 'cuda'

# ==============================================================================
# ==============================================================================



def diarization(data_path, output_path, exactNum = False, gt_path = "", 
                isFar = False, asr_based_ts = False):
    if asr_based_ts:
        config.diarizer.asr.parameters.asr_based_vad = True
    else:
        config.diarizer.asr.parameters.asr_based_vad = False
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
    asr_decoder_ts = ASRDecoderTimeStamps(config.diarizer)
    asr_model = asr_decoder_ts.set_asr_model()
    
    if not os.path.exists(manifest_filepath):
        for file in os.listdir(data_path):
            print('-' * 20, file.center(5), '-' * 20, flush=True)
            # Skip if already diarized
            if (os.path.isfile(os.path.join(output_path, file.split('.')[0] + '.rttm')) 
                or '_converted' in file):
                print('Skipping', file.split('.')[0] + '.rttm', flush = True)
                continue
            else:
                meta_info = create_json_object(data_path, file, gt_path, 
                                            exactNum, isFar)            
                
            with open(manifest_filepath, 'a+') as fp:
                json.dump(meta_info, fp)
                fp.write('\n')
    word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
    asr_diar_offline = OfflineDiarWithASR(config.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
    diar_hyp, diar_score = asr_diar_offline.run_diarization(config, word_ts_hyp)
    # print("Score: \n", diar_score)
    print("DONE!", flush=True)

#---------------------------- BASELINE ----------------------------

print('-' * 20, "BASELINE DEFAULT".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_BASELINE + 'test_fixed/', 
            gt_path = TEST_LABELS_PATH)

print('-' * 20, "BASELINE ASR-BASED TS".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_BASELINE + 'test_fixed_asr_based/', 
            gt_path = TEST_LABELS_PATH, asr_based_ts = True)


#---------------------------- EXACT ----------------------------
print('-' * 20, "EXACT DEFAULT".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_EXACT + 'test_fixed/', exactNum = True, 
            gt_path = TEST_LABELS_PATH)

print('-' * 20, "EXACT ASR-BASED TS".center(5), '-' * 20, flush=True)
diarization(TEST_PATH, OUTPUT_PATH_EXACT + 'test_fixed_asr_based/', exactNum = True, 
            gt_path = TEST_LABELS_PATH, asr_based_ts = True)
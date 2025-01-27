# Source code: https://github.com/m-bain/whisperX/blob/main/README.md
import whisperx
import torch
import os

# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

BATCH_SIZE = 16 # reduce if low on GPU mem
COMPUTE_TYPE = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
MODEL = "large-v2"

AUTH_TOKEN = 'your_auth_token'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PY2_1 =  "2.1/"
PY3_1 =  "3.1/"

v21 = "speaker-diarization@2.1"
v31 = "speaker-diarization-3.1"
OUTPUT_PATH = '/research_data/WhisperX'
DEFAULT_ALIGN_MODELS_HF = ["ja", "zh", "nl", "uk", "pt", "ar", "cs", "ru", "pl",
    "hu","fi", "fa", "el", "tr", "da", "he", "vi", "ko", "ur", "te", "hi", "en",
    "fr", "de", "es", "it"]

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================

MSD_DATASET_PATH = '/path/to/datasets/folder/MSDWILD/'
msd_TEST_PATH_FEW = os.path.join(MSD_DATASET_PATH, 'wav/few')
msd_TEST_PATH_MANY = os.path.join(MSD_DATASET_PATH, 'wav/many')

msd_TEST_LABELS_PATH_FEW = os.path.join(MSD_DATASET_PATH, 'rttm/few')
msd_TEST_LABELS_PATH_MANY = os.path.join(MSD_DATASET_PATH, 'rttm/many')

msd_OUTPUT_PATH_EP = os.path.join(OUTPUT_PATH, 'MSDWILD/results/global/')
msd_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_PATH, 'MSDWILD/results/baseline/')
msd_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_PATH, 'MSDWILD/results/exact/')
msd_ID = 5


# ===========================================================================
# ================================= METHODS =================================
# ===========================================================================

def diarization(dataset, data_path, output_path, min_speakers = None, max_speakers = None, 
                version = v21, globalP = False, exactNum = False, gt_path = "", isFar = False):
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(MODEL, DEVICE, compute_type = COMPUTE_TYPE)
    
    for name in os.listdir(data_path):
        
        if os.path.isfile(
            os.path.join(output_path, name.split('.')[0] + '.rttm')):
            
            print('Skipping', name.split('.')[0] + '.rttm', flush = True)
            continue
        else:
            print('-' * 20, name.center(5), '-' * 20, flush=True)
            
            file_path = os.path.join(data_path, name)
            
            res, lang = asr_diarization(model, file_path, min_speakers, 
                                max_speakers, globalP, version,
                                exactNum, gt_path, dataset, isFar)
            if lang is not None:
                print('Skipping', name.split('.')[0] + " Lang = " + lang, 
                        flush = True)
                continue
            else:
                # Save speaker diarization
                save2rttm(res["segments"], output_path, name)
    print("DONE!", flush=True)
            
def asr_diarization(model, file, min_speakers, max_speakers, globalP, version, 
                    exactNum, gt_path, dataset, isFar = False):
    
    print(file, flush=True)
    audio = whisperx.load_audio(file)
    result = model.transcribe(audio, batch_size = BATCH_SIZE)
    # 2. Align whisper output
    if result["language"] not in DEFAULT_ALIGN_MODELS_HF:
        return None, result["language"]
    
    model_a, metadata = whisperx.load_align_model(
        language_code = result["language"], device = DEVICE)
    
    result = whisperx.align(result["segments"], model_a, 
                            metadata, audio, DEVICE, 
                            return_char_alignments = False)

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(
        model_name = "pyannote/" + version, use_auth_token = AUTH_TOKEN, 
        device = DEVICE)
    
    
    if globalP:
        diarize_segments = diarize_model(audio, min_speakers = min_speakers, 
                      max_speakers = max_speakers)
    elif exactNum:
        audio_file = file.split('/')[-1]
        
        file_name = audio_file.split('.')[0] 
        gt_file = file_name + '.rttm'
        num = getNumberSpeakers(gt_file, gt_path)
        diarize_segments = diarize_model(audio, min_speakers = num, 
                      max_speakers = num)
    else:
        diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    return result, None

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
    
def save2rttm(segments, output_path, audio_name):
    lines = []
    
    for segment in segments: 
        start_time = float(segment['start'])
        end_time = float(segment['end'])
        if 'speaker' in segment.keys():
            spk_id = segment['speaker']
        else:
            spk_id = "N/A"
        lines.append(
            "SPEAKER meeting\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\n".format(
                start_time, end_time - start_time, spk_id))
                
    if not os.path.exists(output_path):
        os.makedirs(output_path)    
    with open(os.path.join(output_path, audio_name.split('.')[0] + '.rttm'), 'w') as rttm:
        for seg in lines:
            rttm.write(str(seg)) 
      

# # ===========================================================================
# # =============================== MSDWILD ===============================
# # ===========================================================================

# #---------------------------- GLOBAL ----------------------------

print('-' * 20, "MSD GLOBAL FEW 2.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_FEW, 
            output_path = msd_OUTPUT_PATH_EP + PY2_1 + 'few/', 
            min_speakers = 2, 
            max_speakers = 4, 
            version = v21,
            globalP = True)

print('-' * 20, "MSD GLOBAL FEW 3.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_FEW, 
            output_path = msd_OUTPUT_PATH_EP + PY3_1 + 'few/', 
            min_speakers = 2, 
            max_speakers = 4, 
            version = v31,
            globalP = True)

print('-' * 20, "MSD GLOBAL MANY 2.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_MANY, 
            output_path = msd_OUTPUT_PATH_EP + PY2_1 + 'many/', 
            min_speakers = 3, 
            max_speakers = 9, 
            version = v21,
            globalP = True)

print('-' * 20, "MSD GLOBAL MANY 3.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_MANY, 
            output_path = msd_OUTPUT_PATH_EP + PY3_1 + 'many/', 
            min_speakers = 3, 
            max_speakers = 9, 
            version = v31,
            globalP = True)


# #---------------------------- BASELINE ----------------------------


print('-' * 20, "MSD BASELINE FEW 2.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_FEW, 
            output_path = msd_OUTPUT_PATH_BASELINE + PY2_1 + 'few/', 
            min_speakers = None,
            max_speakers = None,
            version = v21,
            globalP = False, exactNum = False)

print('-' * 20, "MSD BASELINE FEW 3.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_FEW, 
            output_path = msd_OUTPUT_PATH_BASELINE + PY3_1 + 'few/', 
            min_speakers = None,
            max_speakers = None,
            version = v31,
            globalP = False, exactNum = False)

print('-' * 20, "MSD BASELINE MANY 2.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_MANY, 
            output_path = msd_OUTPUT_PATH_BASELINE + PY2_1 + 'many/', 
            min_speakers = None,
            max_speakers = None,
            version = v21,
            globalP = False, exactNum = False)

print('-' * 20, "MSD BASELINE MANY 3.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_MANY, 
            output_path = msd_OUTPUT_PATH_BASELINE + PY3_1 + 'many/', 
            min_speakers = None,
            max_speakers = None,
            version = v31,
            globalP = False, exactNum = False)


#---------------------------- EXACT ----------------------------

print('-' * 20, "MSD EXACT NUM FEW 2.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_FEW, 
            output_path = msd_OUTPUT_PATH_EXACT + PY2_1 + 'few/', 
            version = v21, 
            exactNum = True, 
            gt_path = msd_TEST_LABELS_PATH_FEW)

print('-' * 20, "MSD EXACT NUM FEW 3.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_FEW, 
            output_path = msd_OUTPUT_PATH_EXACT + PY3_1 + 'few/', 
            version = v31, 
            exactNum = True, 
            gt_path = msd_TEST_LABELS_PATH_FEW)

print('-' * 20, "MSD EXACT NUM MANY 2.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_MANY, 
            output_path = msd_OUTPUT_PATH_EXACT + PY2_1 + 'many/', 
            version = v21, 
            exactNum = True, 
            gt_path = msd_TEST_LABELS_PATH_MANY)

print('-' * 20, "MSD EXACT NUM MANY 3.1".center(5), '-' * 20, flush=True)
diarization(dataset = msd_ID, 
            data_path = msd_TEST_PATH_MANY, 
            output_path = msd_OUTPUT_PATH_EXACT + PY3_1 + 'many/', 
            version = v31, 
            exactNum = True, 
            gt_path = msd_TEST_LABELS_PATH_MANY)
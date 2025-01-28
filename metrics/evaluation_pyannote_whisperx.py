import os
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate, DiarizationPurity, DiarizationCoverage
from pyannote.core import Annotation, Segment
import torch
import re
import csv


# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

AUTH_TOKEN = 'your_auth_token'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PY2_1 =  "2.1/"
PY3_1 =  "3.1/"

OUTPUT_ASR_DIA_PATH = '/research_data/diarization/WhisperX/'
OUTPUT_DIA_PATH = '/research_data/diarization/pyannote'


# PYANNOTE
COMPARATIVE_ANALYSIS = False
WHISPERX = False

# WHISPERX
# COMPARATIVE_ANALYSIS = False
# WHISPERX = True

# ===========================================================================
# ================================ AISHELL-4 ================================
# ===========================================================================
if WHISPERX:
    ai_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'AISHELL-4/results/global/')
    ai_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'AISHELL-4/results/baseline/')
    ai_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'AISHELL-4/results/exact/')
else:
    ai_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'AISHELL-4/results/global/')
    ai_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'AISHELL-4/results/baseline/')
    ai_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'AISHELL-4/results/exact/')

ai_TEST_GT_PATH = '/path/to/datasets/folder/AISHELL-4/test/TextGrid'


# ===========================================================================
# =================================== AMI ===================================
# ===========================================================================
if WHISPERX:
    ami_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'AMI/results/global/')
    ami_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'AMI/results/baseline/')
    ami_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'AMI/results/exact/')
else:
    ami_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'AMI/results/global/')
    ami_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'AMI/results/baseline/')
    ami_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'AMI/results/exact/')

AMI_PATH = '/path/to/datasets/folder/AMI/headset-mix/Labels/'
ami_TEST_GT_PATH = AMI_PATH + 'test/'


# ===========================================================================
# =============================== VoxConverse ===============================
# ===========================================================================
if WHISPERX:
    vox_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'VoxConverse/results/global/')
    vox_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'VoxConverse/results/baseline/')
    vox_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'VoxConverse/results/exact/')
else: 
    vox_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'VoxConverse/results/global/')
    vox_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'VoxConverse/results/baseline/')
    vox_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'VoxConverse/results/exact/')

vox_TEST_GT_PATH = '/path/to/datasets/folder/VoxConverse/GroundTruth/test'


# ===========================================================================
# =============================== TAM ===============================
# ===========================================================================
if WHISPERX:
    tam_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'This American Life/results/global/')
    tam_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'This American Life/results/baseline/')
    tam_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'This American Life/results/exact/')
else: 
    tam_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'This American Life/results/global/')
    tam_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'This American Life/results/baseline/')
    tam_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'This American Life/results/exact/')

tam_TEST_GT_PATH = '/path/to/datasets/folder/TAL/rttm'


# ===========================================================================
# =============================== RAMC ===============================
# ===========================================================================
if WHISPERX:
    ramc_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'RAMC/results/global/')
    ramc_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'RAMC/results/baseline/')
    ramc_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'RAMC/results/exact/')
else: 
    ramc_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'RAMC/results/global/')
    ramc_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'RAMC/results/baseline/')
    ramc_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'RAMC/results/exact/')

ramc_TEST_GT_PATH = '/path/to/datasets/folder/RAMC/MDT2021S003/rttm/test'

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================
if WHISPERX:
    msdwild_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'MSDWILD/results/global/')
    msdwild_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'MSDWILD/results/baseline/')
    msdwild_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'MSDWILD/results/exact/')
else: 
    msdwild_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'MSDWILD/results/global/')
    msdwild_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'MSDWILD/results/baseline/')
    msdwild_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'MSDWILD/results/exact/')

msdwild_few_TEST_GT_PATH = '/path/to/datasets/folder/MSDWILD/rttm/few'
msdwild_many_TEST_GT_PATH = '/path/to/datasets/folder/MSDWILD/rttm/many'

# ===========================================================================
# =============================== EARNINGS-21 ===============================
# ===========================================================================
if WHISPERX:
    earn_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'Earnings-21/results/global/')
    earn_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'Earnings-21/results/baseline/')
    earn_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'Earnings-21/results/exact/')
else: 
    earn_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'Earnings-21/results/global/')
    earn_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'Earnings-21/results/baseline/')
    earn_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'Earnings-21/results/exact/')

earn_TEST_GT_PATH = '/path/to/datasets/folder/Earnings-21/earnings-21/earnings21/rttms'

# ===========================================================================
# =============================== ALI ===============================
# ===========================================================================
if WHISPERX:
    ali_OUTPUT_PATH_EP = os.path.join(OUTPUT_ASR_DIA_PATH, 'AliMeeting/results/global/')
    ali_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_ASR_DIA_PATH, 'AliMeeting/results/baseline/')
    ali_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_ASR_DIA_PATH, 'AliMeeting/results/exact/')
else: 
    ali_OUTPUT_PATH_EP = os.path.join(OUTPUT_DIA_PATH, 'AliMeeting/results/global/')
    ali_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'AliMeeting/results/baseline/')
    ali_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'AliMeeting/results/exact/')

ali_far_TEST_GT_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/Test_Ali_far/rttm_dir'
ali_near_TEST_GT_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/Test_Ali_near/rttm_dir'


if WHISPERX:
    print('-' * 20, "WHISPERX".center(5), '-' * 20, flush=True)
else:
    print('-' * 20, "PYANNOTE".center(5), '-' * 20, flush=True)


def metrics(GT_path, output_path, dataset, subset, version, approach, isFar = None):
    rows = []    
    method = ''
    if WHISPERX:
        method = 'WhisperX'
    else:
        method = 'pyannote'
        
        
    print("OUTPUT PATH: ", output_path, flush=True)
    for file in os.listdir(GT_path):

        added_h = False
        # Create reference for ground-truth
        if file.split('.')[1] == 'rttm':
            
            gt_file_path = os.path.join(GT_path, file)
            if isFar == True:
                for diarization in os.listdir(output_path):
                    print(diarization, flush=True)
                    if file.split('.')[0] in diarization:
                        file = diarization
                        
            ref = Annotation()
            if os.path.isfile(os.path.join(output_path, file)):
                with open(gt_file_path, 'r') as rttm_file:
                    for line in rttm_file:
                        parts = line.strip().split()  # Split each line into parts
                        
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        end_time = start_time + duration
                        speaker_label = parts[7]
                        if 'h' in speaker_label:
                            added_h = True
                            break
                        ref[Segment(start_time, end_time)] = re.findall(r'\d+', speaker_label)[0]
                if added_h:
                    continue
                # Create reference for diarization result
                hyp_file_path = os.path.join(output_path, file)
                hyp = Annotation()
                with open(hyp_file_path, 'r') as rttm_file:
                    for line in rttm_file:
                        parts = line.strip().split()  # Split each line into parts
                        
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        end_time = start_time + duration
                        speaker_label = parts[7]
                        if speaker_label != "N/A":
                            hyp[Segment(start_time, end_time)] = re.findall(r'\d+', speaker_label)[0]
            
                # Calculate per-audio DER
                der = DiarizationErrorRate()
                der_value = der(ref, hyp)
                
                # Calculate per-audio JER
                jer = JaccardErrorRate()
                jer_value = jer(ref, hyp)
                
                # Calculate per-audio Purity
                pur = DiarizationPurity()
                pur_value = pur(ref, hyp)
                
                # Calculate per-audio Coverage
                cov = DiarizationCoverage()
                cov_value = cov(ref, hyp)

                rows.append([dataset,subset,version,approach,"DER", round(der_value, 5), method, file.split('.')[0]])
                rows.append([dataset,subset,version,approach,"JER", round(jer_value, 5), method, file.split('.')[0]])
                rows.append([dataset,subset,version,approach,"Purity", round(pur_value, 5), method, file.split('.')[0]])
                rows.append([dataset,subset,version,approach,"Coverage", round(cov_value, 5), method, file.split('.')[0]])

   
    name = method + '_baseline.csv'
        
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open(name, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        #write multiple rows
        writer.writerows(rows)


header = ['Dataset', 'Set', 'Version', 'Approach', 'Metric', 'Value', 'Method', 'File']

if WHISPERX:
    name = 'WhisperX_baseline.csv'
else:
    name = 'pyannote_baseline.csv'

writeHeader = True
if os.path.isfile(os.path.join(name)):
    writeHeader = False
# Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
with open(name, 'a+', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    if writeHeader:
        writer.writerow(header)

# ===========================================================================
# ================================ AISHELL-4 ================================
# ===========================================================================
      
#---------------------------- GLOBAL ----------------------------

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EP + PY2_1 + 'test/', "AISHELL-4", "", 
        "2.1", "Global")

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EP + PY3_1 + 'test/', "AISHELL-4", "", 
        "3.1", "Global") 
 

#---------------------------- BASELINE ----------------------------
metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_BASELINE + PY2_1 + 'test/', "AISHELL-4", 
        "", "2.1", "Baseline")

    
metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_BASELINE + PY3_1 + 'test/', "AISHELL-4", 
        "", "3.1", "Baseline") 

#---------------------------- EXACT ----------------------------

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EXACT + PY2_1 + 'test/', "AISHELL-4", 
        "", "2.1", "Exact")

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EXACT + PY3_1 + 'test/', "AISHELL-4", 
        "", "3.1", "Exact") 

# ===========================================================================
# =================================== AMI ===================================
# ===========================================================================

#------------------------------ PYANNOTE ------------------------------
#---------------------------- GLOBAL ----------------------------

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EP + PY2_1 + 'test/', "AMI", "", "2.1", 
        "Global")

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EP + PY3_1 + 'test/', "AMI", "", "3.1", 
        "Global")  


#---------------------------- BASELINE ----------------------------

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_BASELINE + PY2_1 + 'test/', "AMI", "", 
        "2.1", "Baseline")

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_BASELINE + PY3_1 + 'test/', "AMI", "", 
        "3.1", "Baseline")  

# #---------------------------- EXACT ----------------------------

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EXACT + PY2_1 + 'test/', "AMI", "", 
        "2.1", "Exact")

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EXACT + PY3_1 + 'test/', "AMI", "", 
        "3.1", "Exact") 

# ===========================================================================
# =============================== VoxConverse ===============================
# ===========================================================================

#------------------------------ PYANNOTE ------------------------------
#---------------------------- GLOBAL ----------------------------
metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_EP + PY2_1 + 'test/', "VoxConverse", 
        "", "2.1", "Global")

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_EP + PY3_1 + 'test/', "VoxConverse", 
        "", "3.1", "Global") 


#---------------------------- BASELINE ----------------------------

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + PY2_1 + 'test/', 
        "VoxConverse", "", "2.1", "Baseline")

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + PY3_1 + 'test/', 
        "VoxConverse", "", "3.1", "Baseline") 


#------------------------------- EXACT -------------------------------


metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_EXACT + PY2_1 + 'test/', 
        "VoxConverse", "", "2.1", "Exact")

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_EXACT + PY3_1 + 'test/', 
        "VoxConverse", "", "3.1", "Exact")  


# ===========================================================================
# =============================== TAL ===============================
# ===========================================================================

# ------------------------------ PYANNOTE ------------------------------
# ---------------------------- GLOBAL ----------------------------
metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EP + PY2_1 + 'test/', 
        "ThisAmericanLife", "", "2.1", "Global")

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EP + PY3_1 + 'test/', 
        "ThisAmericanLife", "", "3.1", "Global") 


#---------------------------- BASELINE ----------------------------

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_BASELINE + PY2_1 + 'test/', 
        "ThisAmericanLife", "", "2.1", "Baseline")

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_BASELINE + PY3_1 + 'test/', 
        "ThisAmericanLife", "", "3.1", "Baseline")  


#------------------------------- EXACT -------------------------------


metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EXACT + PY2_1 + 'test/', 
        "ThisAmericanLife", "", "2.1", "Exact")

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EXACT + PY3_1 + 'test/', 
        "ThisAmericanLife", "", "3.1", "Exact")  

# ===========================================================================
# =============================== RAMC ===============================
# ===========================================================================

# ------------------------------ PYANNOTE ------------------------------
# ---------------------------- GLOBAL ----------------------------
metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EP + PY2_1 + 'test/', 
        "RAMC", "", "2.1", "Global")

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EP + PY3_1 + 'test/', 
        "RAMC", "", "3.1", "Global") 


#---------------------------- BASELINE ----------------------------

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_BASELINE + PY2_1 + 'test/', "RAMC", 
        "", "2.1", "Baseline")

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_BASELINE + PY3_1 + 'test/', "RAMC", 
        "", "3.1", "Baseline")  


#------------------------------- EXACT -------------------------------


metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EXACT + PY2_1 + 'test/', "RAMC", 
        "", "2.1", "Exact")

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EXACT + PY3_1 + 'test/', "RAMC", 
        "", "3.1", "Exact")  

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================

# ------------------------------ PYANNOTE ------------------------------
# ---------------------------- GLOBAL ----------------------------
metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EP + PY2_1 + 'few/', 
        "MSDWILD", "Few", "2.1", "Global")

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EP + PY3_1 + 'few/', 
        "MSDWILD", "Few", "3.1", "Global") 

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EP + PY2_1 + 'many/', 
        "MSDWILD", "Many", "2.1", "Global")

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EP + PY3_1 + 'many/', 
        "MSDWILD", "Many", "3.1", "Global") 

#---------------------------- BASELINE ----------------------------

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY2_1 + 'few/', 
        "MSDWILD", "Few", "2.1", "Baseline")

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY3_1 + 'few/', 
        "MSDWILD", "Few", "3.1", "Baseline")  

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY2_1 + 'few/', 
                 "MSDWILD", "Few", "2.1", "Baseline")

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY3_1 + 'few/', 
                 "MSDWILD", "Few", "3.1", "Baseline")  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY2_1 + 'many/', 
        "MSDWILD", "Many", "2.1", "Baseline")

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY3_1 + 'many/', 
        "MSDWILD", "Many", "3.1", "Baseline")  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY2_1 + 'many/', 
                 "MSDWILD", "Many", "2.1", "Baseline")

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + PY3_1 + 'many/', 
                 "MSDWILD", "Many", "3.1", "Baseline")  

#------------------------------- EXACT -------------------------------


metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + PY2_1 + 'few/', 
        "MSDWILD", "Few", "2.1", "Exact")

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + PY3_1 + 'few/',
         "MSDWILD", "Few", "3.1", "Exact")  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + PY2_1 + 'many/', 
        "MSDWILD", "Many", "2.1", "Exact")

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + PY3_1 + 'many/', 
        "MSDWILD", "Many", "3.1", "Exact")  

# ===========================================================================
# =============================== EARNINGS-21 ===============================
# ===========================================================================

# ------------------------------ PYANNOTE ------------------------------
# ---------------------------- GLOBAL ----------------------------
metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EP + PY2_1 + 'test/', "Earnings-21", 
        "", "2.1", "Global")

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EP + PY3_1 + 'test/', "Earnings-21", 
        "", "3.1", "Global") 


#---------------------------- BASELINE ----------------------------

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_BASELINE + PY2_1 + 'test/', 
        "Earnings-21", "", "2.1", "Baseline")

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_BASELINE + PY3_1 + 'test/', 
        "Earnings-21", "", "3.1", "Baseline")  


#------------------------------- EXACT -------------------------------


metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EXACT + PY2_1 + 'test/', 
        "Earnings-21", "", "2.1", "Exact")

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EXACT + PY3_1 + 'test/', 
        "Earnings-21", "", "3.1", "Exact")  

# # ===========================================================================
# # =============================== ALIMEETING ===============================
# # ===========================================================================

# #------------------------------ PYANNOTE ------------------------------
# #---------------------------- GLOBAL ----------------------------
metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_EP + PY2_1 + 'far/', "AliMeeting", 
        "Far", "2.1", "Global", True)

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_EP + PY3_1 + 'far/', "AliMeeting", 
        "Far", "3.1", "Global", True) 

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_EP + PY2_1 + 'near/', "AliMeeting", 
        "Near", "2.1", "Global", False)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_EP + PY3_1 + 'near/', "AliMeeting", 
        "Near", "3.1", "Global", False) 


#---------------------------- BASELINE ----------------------------

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + PY2_1 + 'far/', 
        "AliMeeting", "Far", "2.1", "Baseline", True)

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + PY3_1 + 'far/', 
        "AliMeeting", "Far", "3.1", "Baseline", True)  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + PY2_1 + 'near/', 
        "AliMeeting", "Near", "2.1", "Baseline", False)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + PY3_1 + 'near/', 
        "AliMeeting", "Near", "3.1", "Baseline", False)  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + PY2_1 + 'near/', 
                 "AliMeeting", "Near", "2.1", "Baseline")

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + PY3_1 + 'near/', 
                 "AliMeeting", "Near", "3.1", "Baseline")  


# ------------------------------- EXACT -------------------------------


metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_EXACT + PY2_1 + 'far/', 
        "AliMeeting", "Far", "2.1", "Exact", True)

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_EXACT + PY3_1 + 'far/', 
        "AliMeeting", "Far", "3.1", "Exact", True)  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_EXACT + PY2_1 + 'near/', 
        "AliMeeting", "Near", "2.1", "Exact", False)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_EXACT + PY3_1 + 'near/', 
        "AliMeeting", "Near", "3.1", "Exact", False)  

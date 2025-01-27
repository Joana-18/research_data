import os
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate, DiarizationPurity, DiarizationCoverage
from pyannote.core import Annotation, Segment
import torch
import re
import csv


# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

OUTPUT_DIA_PATH = '/research_data/pyAudioAnalysis'

# ===========================================================================
# ================================ AISHELL-4 ================================
# ===========================================================================
ai_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'AISHELL-4/results/baseline/')
ai_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'AISHELL-4/results/exact/')
ai_TEST_GT_PATH = '/path/to/datasets/folder/AISHELL-4/test/TextGrid'


# ===========================================================================
# =================================== AMI ===================================
# ===========================================================================

ami_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'AMI/results/baseline/')
ami_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'AMI/results/exact/')
AMI_PATH = '/path/to/datasets/folder/AMI/headset-mix/Labels/'
ami_TEST_GT_PATH = AMI_PATH + 'test/'


# ===========================================================================
# =============================== VoxConverse ===============================
# ===========================================================================
vox_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'VoxConverse/results/baseline/')
vox_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'VoxConverse/results/exact/')
vox_TEST_GT_PATH = '/path/to/datasets/folder/VoxConverse/GroundTruth/test'


# ===========================================================================
# =============================== TAM ===============================
# ===========================================================================

tam_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'This American Life/results/baseline/')
tam_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'This American Life/results/exact/')
tam_TEST_GT_PATH = '/path/to/datasets/folder/TAL/rttm'


# ===========================================================================
# =============================== RAMC ===============================
# ===========================================================================
ramc_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'RAMC/results/baseline/')
ramc_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'RAMC/results/exact/')
ramc_TEST_GT_PATH = '/path/to/datasets/folder/RAMC/MDT2021S003/rttm/test'

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================
msdwild_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'MSDWILD/results/baseline/')
msdwild_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'MSDWILD/results/exact/')
msdwild_few_TEST_GT_PATH = '/path/to/datasets/folder/MSDWILD/rttm/few'
msdwild_many_TEST_GT_PATH = '/path/to/datasets/folder/MSDWILD/rttm/many'

# ===========================================================================
# =============================== EARNINGS-21 ===============================
# ===========================================================================
earn_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'Earnings-21/results/baseline/')
earn_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'Earnings-21/results/exact/')
earn_TEST_GT_PATH = '/path/to/datasets/folder/Earnings-21/earnings-21/earnings21/rttms'

# ===========================================================================
# =============================== ALI ===============================
# ===========================================================================
ali_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'AliMeeting/results/baseline/')
ali_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'AliMeeting/results/exact/')
ali_far_TEST_GT_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/Test_Ali_far/rttm_dir'
ali_near_TEST_GT_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/Test_Ali_near/rttm_dir'


def metrics(GT_path, output_path, dataset, subset, approach, isFar = None):
    rows = []    
    method = ''
                
    for file in os.listdir(GT_path):

        # Create reference for ground-truth
        if file.split('.')[1] == 'rttm':
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

                rows.append([dataset,subset,approach,"DER", round(der_value, 5), method, file.split('.')[0]])
                rows.append([dataset,subset,approach,"JER", round(jer_value, 5), method, file.split('.')[0]])
                rows.append([dataset,subset,approach,"Purity", round(pur_value, 5), method, file.split('.')[0]])
                rows.append([dataset,subset,approach,"Coverage", round(cov_value, 5), method, file.split('.')[0]])


    name = 'pyAudioAnalysis.csv'
    
        
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open(name, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        #write multiple rows
        writer.writerows(rows)



header = ['Dataset', 'Set', 'Approach', 'Metric', 'Value', 'Method', 'File']

name  = 'pyAudioAnalysis.csv'

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

#---------------------------- BASELINE ----------------------------
metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_BASELINE + 'test/', "AISHELL-4", "", "Baseline")

#---------------------------- EXACT ----------------------------

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EXACT + 'test/', "AISHELL-4", "", "Exact")

# # ===========================================================================
# # =================================== AMI ===================================
# # ===========================================================================

# #---------------------------- BASELINE ----------------------------

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_BASELINE + 'test/', "AMI", "", "Baseline")

# #---------------------------- EXACT ----------------------------

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EXACT + 'test/', "AMI", "", "Exact")

# # ===========================================================================
# # =============================== VoxConverse ===============================
# # ===========================================================================

# #---------------------------- BASELINE ----------------------------

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/', "VoxConverse", "",  "Baseline") 

# # #---------------------------- EXACT ----------------------------

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/', "VoxConverse", "", "Exact")


# ===========================================================================
# =============================== TAL ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_BASELINE + 'test/', "ThisAmericanLife", "", "Baseline")  


#------------------------------- EXACT -------------------------------

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EXACT + 'test/', "ThisAmericanLife", "", "Exact")  

# ===========================================================================
# =============================== RAMC ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_BASELINE + 'test/', "RAMC", "", "Baseline")

#------------------------------- EXACT -------------------------------

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EXACT + 'test/', "RAMC", "", "Exact")  

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'few/', "MSDWILD", "Few", "Baseline")

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'many/', "MSDWILD", "Many", "Baseline")

#------------------------------- EXACT -------------------------------

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'few/', "MSDWILD", "Few", "Exact")  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'many/', "MSDWILD", "Many", "Exact")  

# ===========================================================================
# =============================== EARNINGS-21 ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_BASELINE + 'test/', "Earnings-21", "", "Baseline")  


#------------------------------- EXACT -------------------------------

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EXACT + 'test/', "Earnings-21", "", "Exact")  

# # ===========================================================================
# # =============================== ALIMEETING ===============================
# # ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/', "AliMeeting", "Far", "Baseline", True)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/', "AliMeeting", "Near", "Baseline", False)


#------------------------------- EXACT -------------------------------

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/', "AliMeeting", "Far", "", "Exact")  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/', "AliMeeting", "Near", "", "Exact")  
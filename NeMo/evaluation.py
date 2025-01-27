import os
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate, DiarizationPurity, DiarizationCoverage
from pyannote.core import Annotation, Segment
import torch
import re
import csv


# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

OUTPUT_DIA_PATH = 'research_data/NeMo'
SYS_VAD_CLUS = 'system_vad-clustering'
SYS_VAD_NEUR = 'system_vad-neural_diarizer'
JOINT = 'asr_diarization'

# ===========================================================================
# ================================ AISHELL-4 ================================
# ===========================================================================

ai_DATASET = 'AISHELL-4'

ai_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{ai_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
ai_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{ai_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
ai_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{ai_DATASET}/{JOINT}/results/baseline/'

ai_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{ai_DATASET}/{SYS_VAD_CLUS}/results/exact/'
ai_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{ai_DATASET}/{SYS_VAD_NEUR}/results/exact/'
ai_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{ai_DATASET}/{JOINT}/results/exact/'
ai_TEST_GT_PATH = F'/path/to/datasets/folder/{ai_DATASET}/test/TextGrid'


# ===========================================================================
# =================================== AMI ===================================
# ===========================================================================

ami_DATASET = 'AMI'
ami_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{ami_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
ami_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{ami_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
ami_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{ami_DATASET}/{JOINT}/results/baseline/'

ami_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{ami_DATASET}/{SYS_VAD_CLUS}/results/exact/'
ami_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{ami_DATASET}/{SYS_VAD_NEUR}/results/exact/'
ami_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{ami_DATASET}/{JOINT}/results/exact/'

AMI_PATH = f'/path/to/path/to/datasets/folder/folder/{ami_DATASET}/headset-mix/Labels/'
ami_TEST_GT_PATH = AMI_PATH + 'test/'


# ===========================================================================
# =============================== VoxConverse ===============================
# ===========================================================================

vox_DATASET = 'VoxConverse'
vox_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{vox_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
vox_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{vox_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
vox_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{vox_DATASET}/{JOINT}/results/baseline/'

vox_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{vox_DATASET}/{SYS_VAD_CLUS}/results/exact/'
vox_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{vox_DATASET}/{SYS_VAD_NEUR}/results/exact/'
vox_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{vox_DATASET}/{JOINT}/results/exact/'

vox_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/{vox_DATASET}/GroundTruth/test'


# ===========================================================================
# =============================== TAL ===============================
# ===========================================================================

tal_DATASET = 'This American Life'
tal_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{tal_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
tal_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{tal_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
tal_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{tal_DATASET}/{JOINT}/results/baseline/'

tal_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{tal_DATASET}/{SYS_VAD_CLUS}/results/exact/'
tal_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{tal_DATASET}/{SYS_VAD_NEUR}/results/exact/'
tal_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{tal_DATASET}/{JOINT}/results/exact/'

tam_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/TAL/rttm'


# ===========================================================================
# =============================== RAMC ===============================
# ===========================================================================

ramc_DATASET = 'RAMC'
ramc_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{ramc_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
ramc_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{ramc_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
ramc_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{ramc_DATASET}/{JOINT}/results/baseline/'

ramc_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{ramc_DATASET}/{SYS_VAD_CLUS}/results/exact/'
ramc_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{ramc_DATASET}/{SYS_VAD_NEUR}/results/exact/'
ramc_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{ramc_DATASET}/{JOINT}/results/exact/'

ramc_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/{ramc_DATASET}/MDT2021S003/rttm/test'

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================

msd_DATASET = 'MSDWILD'
msdwild_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{msd_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
msdwild_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{msd_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
msdwild_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{msd_DATASET}/{JOINT}/results/baseline/'

msdwild_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{msd_DATASET}/{SYS_VAD_CLUS}/results/exact/'
msdwild_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{msd_DATASET}/{SYS_VAD_NEUR}/results/exact/'
msdwild_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{msd_DATASET}/{JOINT}/results/exact/'

msdwild_few_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/{msd_DATASET}/rttm/few'
msdwild_many_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/{msd_DATASET}/rttm/many'

# ===========================================================================
# =============================== EARNINGS-21 ===============================
# ===========================================================================

earn_DATASET = 'Earnings-21'
earn_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{earn_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
earn_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{earn_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
earn_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{earn_DATASET}/{JOINT}/results/baseline/'

earn_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{earn_DATASET}/{SYS_VAD_CLUS}/results/exact/'
earn_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{earn_DATASET}/{SYS_VAD_NEUR}/results/exact/'
earn_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{earn_DATASET}/{JOINT}/results/exact/'

earn_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/{earn_DATASET}/earnings-21/earnings21/rttms'

# ===========================================================================
# =============================== ALI ===============================
# ===========================================================================

ali_DATASET = 'AliMeeting'
ali_BASELINE_CLUS = f'{OUTPUT_DIA_PATH}/{ali_DATASET}/{SYS_VAD_CLUS}/results/baseline/'
ali_BASELINE_NEUR = f'{OUTPUT_DIA_PATH}/{ali_DATASET}/{SYS_VAD_NEUR}/results/baseline/'
ali_BASELINE_ASR = f'{OUTPUT_DIA_PATH}/{ali_DATASET}/{JOINT}/results/baseline/'

ali_EXACT_CLUS = f'{OUTPUT_DIA_PATH}/{ali_DATASET}/{SYS_VAD_CLUS}/results/exact/'
ali_EXACT_NEUR = f'{OUTPUT_DIA_PATH}/{ali_DATASET}/{SYS_VAD_NEUR}/results/exact/'
ali_EXACT_ASR = f'{OUTPUT_DIA_PATH}/{ali_DATASET}/{JOINT}/results/exact/'

ali_far_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/{ali_DATASET}/Test_Ali/Test_Ali_far/rttm_dir'
ali_near_TEST_GT_PATH = f'/path/to/path/to/datasets/folder/folder/{ali_DATASET}/Test_Ali/Test_Ali_near/rttm_dir'


def metrics(GT_path, output_path, dataset, subset, approach, method, isFar = None):
    rows = []    
                
    print("OUTPUT PATH: ", output_path, flush=True)
    h_files = []
    
    output_path = f'{output_path}/pred_rttms'
    if os.path.exists(output_path):
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
                                h_files.append(file)
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

    name = 'NeMo.csv'
    
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open(name, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        #write multiple rows
        writer.writerows(rows)



header = ['Dataset', 'Set', 'Approach', 'Metric', 'Value', 'Method', 'File']

name  = 'NeMo.csv'

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
metrics(ai_TEST_GT_PATH, ai_BASELINE_CLUS + 'test/', "AISHELL-4", "",
        "Baseline", 'Clustering')

metrics(ai_TEST_GT_PATH, ai_BASELINE_NEUR + 'test/', "AISHELL-4", "", 
        "Baseline", 'Neural')

metrics(ai_TEST_GT_PATH, ai_BASELINE_ASR + 'test/', "AISHELL-4", "", 
        "Baseline", 'Joint')

#---------------------------- EXACT ----------------------------

metrics(ai_TEST_GT_PATH, ai_EXACT_CLUS + 'test/', "AISHELL-4", "", 
        "Exact", 'Clustering')

metrics(ai_TEST_GT_PATH, ai_EXACT_NEUR + 'test/', "AISHELL-4", "", 
        "Exact", 'Neural')

metrics(ai_TEST_GT_PATH, ai_EXACT_ASR + 'test/', "AISHELL-4", "", 
        "Exact", 'Joint')

# # ===========================================================================
# # =================================== AMI ===================================
# # ===========================================================================

# #---------------------------- BASELINE ----------------------------

metrics(ami_TEST_GT_PATH, ami_BASELINE_CLUS + 'test/', "AMI", "", 
        "Baseline", 'Clustering')

metrics(ami_TEST_GT_PATH, ami_BASELINE_NEUR + 'test/', "AMI", "", 
        "Baseline", 'Neural')

metrics(ami_TEST_GT_PATH, ami_BASELINE_ASR + 'test/', "AMI", "", 
        "Baseline", 'Joint')

# #---------------------------- EXACT ----------------------------

metrics(ami_TEST_GT_PATH, ami_EXACT_CLUS + 'test/', "AMI", "", 
        "Exact", 'Clustering')

metrics(ami_TEST_GT_PATH, ami_EXACT_NEUR + 'test/', "AMI", "", 
        "Exact", 'Neural')

metrics(ami_TEST_GT_PATH, ami_EXACT_ASR + 'test/', "AMI", "", 
        "Exact", 'Joint')


# # ===========================================================================
# # =============================== VoxConverse ===============================
# # ===========================================================================

# #---------------------------- BASELINE ----------------------------

metrics(vox_TEST_GT_PATH, vox_BASELINE_CLUS + 'test/', "VoxConverse", "",
        "Baseline", 'Clustering') 

metrics(vox_TEST_GT_PATH, vox_BASELINE_NEUR + 'test/', "VoxConverse", "",
        "Baseline", 'Neural') 

metrics(vox_TEST_GT_PATH, vox_BASELINE_ASR + 'test/', "VoxConverse", "",
        "Baseline", 'Joint') 

# # #---------------------------- EXACT ----------------------------

metrics(vox_TEST_GT_PATH, vox_EXACT_CLUS + 'test/', "VoxConverse", "", 
        "Exact", 'Clustering')

metrics(vox_TEST_GT_PATH, vox_EXACT_NEUR + 'test/', "VoxConverse", "", 
        "Exact", 'Neural')

metrics(vox_TEST_GT_PATH, vox_EXACT_ASR + 'test/', "VoxConverse", "", 
        "Exact", 'Joint')


# ===========================================================================
# =============================== TAL ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(tam_TEST_GT_PATH, tal_BASELINE_CLUS + 'test/', "ThisAmericanLife", "", 
        "Baseline", 'Clustering')  

metrics(tam_TEST_GT_PATH, tal_BASELINE_NEUR + 'test/', "ThisAmericanLife", "", 
        "Baseline", 'Neural')  

metrics(tam_TEST_GT_PATH, tal_BASELINE_ASR + 'test/', "ThisAmericanLife", "", 
        "Baseline", 'Joint')  


#------------------------------- EXACT -------------------------------

metrics(tam_TEST_GT_PATH, tal_EXACT_CLUS + 'test/', "ThisAmericanLife", "", 
        "Exact", 'Clustering')  

metrics(tam_TEST_GT_PATH, tal_EXACT_NEUR + 'test/', "ThisAmericanLife", "", 
        "Exact", 'Neural')  

metrics(tam_TEST_GT_PATH, tal_EXACT_ASR + 'test/', "ThisAmericanLife", "", 
        "Exact", 'Joint')  

# ===========================================================================
# =============================== RAMC ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(ramc_TEST_GT_PATH, ramc_BASELINE_CLUS + 'test/', "RAMC", "", 
        "Baseline", 'Clustering')

metrics(ramc_TEST_GT_PATH, ramc_BASELINE_NEUR + 'test/', "RAMC", "", 
        "Baseline", 'Neural')

metrics(ramc_TEST_GT_PATH, ramc_BASELINE_ASR + 'test/', "RAMC", "", 
        "Baseline", 'Joint')

#------------------------------- EXACT -------------------------------

metrics(ramc_TEST_GT_PATH, ramc_EXACT_CLUS + 'test/', "RAMC", "", 
        "Exact", 'Clustering')  

metrics(ramc_TEST_GT_PATH, ramc_EXACT_NEUR + 'test/', "RAMC", "", 
        "Exact", 'Neural')  

metrics(ramc_TEST_GT_PATH, ramc_EXACT_ASR + 'test/', "RAMC", "", 
        "Exact", 'Joint')  

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(msdwild_few_TEST_GT_PATH, msdwild_BASELINE_CLUS + 'few/', "MSDWILD", 
        "Few", "Baseline", 'Clustering')

metrics(msdwild_few_TEST_GT_PATH, msdwild_BASELINE_NEUR + 'few/', "MSDWILD", 
        "Few", "Baseline", 'Neural')

metrics(msdwild_few_TEST_GT_PATH, msdwild_BASELINE_ASR + 'few/', "MSDWILD", 
        "Few", "Baseline", 'Joint')

metrics(msdwild_many_TEST_GT_PATH, msdwild_BASELINE_CLUS + 'many/', "MSDWILD", 
        "Many", "Baseline", 'Clustering')

metrics(msdwild_many_TEST_GT_PATH, msdwild_BASELINE_NEUR + 'many/', "MSDWILD", 
        "Many", "Baseline", 'Neural')

metrics(msdwild_many_TEST_GT_PATH, msdwild_BASELINE_ASR + 'many/', "MSDWILD", 
        "Many", "Baseline", 'Joint')

#------------------------------- EXACT -------------------------------

metrics(msdwild_few_TEST_GT_PATH, msdwild_EXACT_CLUS + 'few/', "MSDWILD", 
        "Few", "Exact", 'Clustering')  

metrics(msdwild_few_TEST_GT_PATH, msdwild_EXACT_NEUR + 'few/', "MSDWILD", 
        "Few", "Exact", 'Neural')  

metrics(msdwild_few_TEST_GT_PATH, msdwild_EXACT_ASR + 'few/', "MSDWILD", 
        "Few", "Exact", 'Joint')  

metrics(msdwild_many_TEST_GT_PATH, msdwild_EXACT_CLUS + 'many/', "MSDWILD",
         "Many", "Exact", 'Clustering')  

metrics(msdwild_many_TEST_GT_PATH, msdwild_EXACT_NEUR + 'many/', "MSDWILD",
         "Many", "Exact", 'Neural')  

metrics(msdwild_many_TEST_GT_PATH, msdwild_EXACT_ASR + 'many/', "MSDWILD",
         "Many", "Exact", 'Joint')  

# ===========================================================================
# =============================== EARNINGS-21 ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(earn_TEST_GT_PATH, earn_BASELINE_CLUS + 'test/', "Earnings-21", "", 
        "Baseline", 'Clustering')  

metrics(earn_TEST_GT_PATH, earn_BASELINE_NEUR + 'test/', "Earnings-21", "", 
        "Baseline", 'Neural')  

metrics(earn_TEST_GT_PATH, earn_BASELINE_ASR + 'test/', "Earnings-21", "", 
        "Baseline", 'Joint')  


#------------------------------- EXACT -------------------------------

metrics(earn_TEST_GT_PATH, earn_EXACT_CLUS + 'test/', "Earnings-21", "", 
        "Exact", 'Clustering')  

metrics(earn_TEST_GT_PATH, earn_EXACT_NEUR + 'test/', "Earnings-21", "", 
        "Exact", 'Neural')  

metrics(earn_TEST_GT_PATH, earn_EXACT_ASR + 'test/', "Earnings-21", "", 
        "Exact", 'Joint')  


# # ===========================================================================
# # =============================== ALIMEETING ===============================
# # ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(ali_far_TEST_GT_PATH, ali_BASELINE_CLUS + 'far/', "AliMeeting", 
        "Far", "Baseline", 'Clustering', True)

metrics(ali_near_TEST_GT_PATH, ali_BASELINE_CLUS + 'near/', "AliMeeting", 
        "Near", "Baseline", 'Clustering', False)

metrics(ali_far_TEST_GT_PATH, ali_BASELINE_NEUR + 'far/', "AliMeeting", 
        "Far", "Baseline", 'Neural', True)

metrics(ali_near_TEST_GT_PATH, ali_BASELINE_CLUS + 'near/', "AliMeeting", 
        "Near", "Baseline", 'Neural', False)

metrics(ali_far_TEST_GT_PATH, ali_BASELINE_ASR + 'far/', "AliMeeting", 
        "Far", "Baseline", 'Joint', True)

metrics(ali_near_TEST_GT_PATH, ali_BASELINE_ASR + 'near/', "AliMeeting", 
        "Near", "Baseline", 'Joint', False)



#------------------------------- EXACT -------------------------------

metrics(ali_far_TEST_GT_PATH, ali_EXACT_CLUS + 'far/', "AliMeeting", 
        "Far", "Exact", 'Clustering', True)  

metrics(ali_near_TEST_GT_PATH, ali_EXACT_CLUS + 'near/', "AliMeeting", 
        "Near", "Exact", 'Clustering')  

metrics(ali_far_TEST_GT_PATH, ali_EXACT_NEUR + 'far/', "AliMeeting", 
        "Far", "Exact", 'Neural', True)  

metrics(ali_near_TEST_GT_PATH, ali_EXACT_NEUR + 'near/', "AliMeeting", 
        "Near", "Exact", 'Neural')  

metrics(ali_far_TEST_GT_PATH, ali_EXACT_ASR + 'far/', "AliMeeting", 
        "Far", "Exact", 'Joint', True)  

metrics(ali_near_TEST_GT_PATH, ali_EXACT_ASR + 'near/', "AliMeeting", 
        "Near", "Exact", 'Joint')    
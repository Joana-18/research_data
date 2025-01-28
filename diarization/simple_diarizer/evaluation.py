import os
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate, DiarizationPurity, DiarizationCoverage
from pyannote.core import Annotation, Segment
import torch
import re
import csv


# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

OUTPUT_DIA_PATH = '/research_data/diarization/simple_diarizer'

ECAPA_AHC = 'ecapa_ahc'
ECAPA_SC = 'ecapa_sc'
XVEC_AHC = 'xvec_ahc'
XVEC_SC = 'xvec_sc'

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
# =============================== TAL ===============================
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
# =============================== ALIMEETING ===============================
# ===========================================================================
ali_OUTPUT_PATH_BASELINE = os.path.join(OUTPUT_DIA_PATH, 'AliMeeting/results/baseline/')
ali_OUTPUT_PATH_EXACT = os.path.join(OUTPUT_DIA_PATH, 'AliMeeting/results/exact/')
ali_far_TEST_GT_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/Test_Ali_far/rttm_dir'
ali_near_TEST_GT_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/Test_Ali_near/rttm_dir'


def metrics(GT_path, output_path, dataset, subset, approach, specs, isFar = None):
    rows = []    
    method = ''
                
    print("OUTPUT PATH: ", output_path, flush=True)

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

                rows.append([dataset,subset,approach,"DER", round(der_value, 5), 
                             method, file.split('.')[0], specs])
                rows.append([dataset,subset,approach,"JER", round(jer_value, 5),
                              method, file.split('.')[0], specs])
                rows.append([dataset,subset,approach,"Purity", round(pur_value, 5), 
                             method, file.split('.')[0], specs])
                rows.append([dataset,subset,approach,"Coverage", round(cov_value, 5), 
                             method, file.split('.')[0], specs])


    name = 'simple_diarizer.csv'
    
        
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open(name, 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        #write multiple rows
        writer.writerows(rows)



header = ['Dataset', 'Set', 'Approach', 'Metric', 'Value', 'Method', 'File', 'Specs']

name  = 'simple_diarizer.csv'

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

# ---------------------------- BASELINE ----------------------------
metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_AHC, 
        "AISHELL-4", "", "Baseline", ECAPA_AHC)

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_SC, 
        "AISHELL-4", "", "Baseline", ECAPA_SC)

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_BASELINE + 'test/' + XVEC_AHC, 
        "AISHELL-4", "", "Baseline", XVEC_AHC)

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_BASELINE + 'test/' + XVEC_SC, 
        "AISHELL-4", "", "Baseline", XVEC_SC)
#---------------------------- EXACT ----------------------------

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EXACT + 'test/' + ECAPA_AHC, "AISHELL-4", 
        "", "Exact", ECAPA_AHC)

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EXACT + 'test/' + ECAPA_SC, "AISHELL-4", 
        "", "Exact", ECAPA_SC)

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EXACT + 'test/' + XVEC_AHC, "AISHELL-4", 
        "", "Exact", XVEC_AHC)

metrics(ai_TEST_GT_PATH, ai_OUTPUT_PATH_EXACT + 'test/' + XVEC_SC, "AISHELL-4", 
        "", "Exact", XVEC_SC)

# ===========================================================================
# =================================== AMI ===================================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_AHC, "AMI",
         "", "Baseline", ECAPA_AHC)

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_SC, "AMI", 
        "", "Baseline", ECAPA_SC)

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_BASELINE + 'test/' + XVEC_AHC, "AMI", 
        "", "Baseline", XVEC_AHC)

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_BASELINE + 'test/' + XVEC_SC, "AMI", 
        "", "Baseline", XVEC_SC)

#---------------------------- EXACT ----------------------------

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EXACT + 'test/' + ECAPA_AHC, "AMI", "", 
        "Exact", ECAPA_AHC)

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EXACT + 'test/' + ECAPA_SC, "AMI", "", 
        "Exact", ECAPA_SC)

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EXACT + 'test/' + XVEC_AHC, "AMI", "", 
        "Exact", XVEC_AHC)

metrics(ami_TEST_GT_PATH, ami_OUTPUT_PATH_EXACT + 'test/' + XVEC_SC, "AMI", "", 
        "Exact", XVEC_SC)

# ===========================================================================
# =============================== VoxConverse ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_AHC, 
        "VoxConverse", "",  "Baseline", ECAPA_AHC) 

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_SC, 
        "VoxConverse", "",  "Baseline", ECAPA_SC) 

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + XVEC_AHC, 
        "VoxConverse", "",  "Baseline", XVEC_AHC) 

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + XVEC_SC, 
        "VoxConverse", "",  "Baseline", XVEC_SC) 

# # #---------------------------- EXACT ----------------------------

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_AHC, 
        "VoxConverse", "", "Exact", ECAPA_AHC)

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_SC, 
        "VoxConverse", "", "Exact", ECAPA_SC)

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + XVEC_AHC, 
        "VoxConverse", "", "Exact", XVEC_AHC)

metrics(vox_TEST_GT_PATH, vox_OUTPUT_PATH_BASELINE + 'test/' + XVEC_SC, 
        "VoxConverse", "", "Exact", XVEC_SC)


# ===========================================================================
# =============================== TAL ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_AHC, 
        "ThisAmericanLife", "", "Baseline", ECAPA_AHC)  

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_SC, 
        "ThisAmericanLife", "", "Baseline", ECAPA_SC)  

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_BASELINE + 'test/' + XVEC_AHC, 
        "ThisAmericanLife", "", "Baseline", XVEC_AHC)  

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_BASELINE + 'test/' + XVEC_SC, 
        "ThisAmericanLife", "", "Baseline", XVEC_SC)  


#------------------------------- EXACT -------------------------------

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EXACT + 'test/' + ECAPA_AHC, 
        "ThisAmericanLife", "", "Exact", ECAPA_AHC)  

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EXACT + 'test/' + ECAPA_SC, 
        "ThisAmericanLife", "", "Exact", ECAPA_SC)  

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EXACT + 'test/' + XVEC_AHC, 
        "ThisAmericanLife", "", "Exact", XVEC_AHC)  

metrics(tam_TEST_GT_PATH, tam_OUTPUT_PATH_EXACT + 'test/' + XVEC_SC, 
        "ThisAmericanLife", "", "Exact", XVEC_SC)  

# ===========================================================================
# =============================== RAMC ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_AHC, 
        "RAMC", "", "Baseline", ECAPA_AHC)

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_SC, 
        "RAMC", "", "Baseline", ECAPA_SC)

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_BASELINE + 'test/' + XVEC_AHC, 
        "RAMC", "", "Baseline", XVEC_AHC)

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_BASELINE + 'test/' + XVEC_SC, 
        "RAMC", "", "Baseline", XVEC_SC)

#------------------------------- EXACT -------------------------------

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EXACT + 'test/' + ECAPA_AHC, 
        "RAMC", "", "Exact", ECAPA_AHC)  

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EXACT + 'test/' + ECAPA_SC, 
        "RAMC", "", "Exact", ECAPA_SC)  

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EXACT + 'test/' + XVEC_AHC, 
        "RAMC", "", "Exact", XVEC_AHC)  

metrics(ramc_TEST_GT_PATH, ramc_OUTPUT_PATH_EXACT + 'test/' + XVEC_SC, 
        "RAMC", "", "Exact", XVEC_SC)  

# ===========================================================================
# =============================== MSDWILD ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'few/' + ECAPA_AHC, 
        "MSDWILD", "Few", "Baseline", ECAPA_AHC)

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'many/' + ECAPA_AHC, 
        "MSDWILD", "Many", "Baseline", ECAPA_AHC)

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'few/' + ECAPA_SC, 
        "MSDWILD", "Few", "Baseline", ECAPA_SC)

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'many/' + ECAPA_SC, 
        "MSDWILD", "Many", "Baseline", ECAPA_SC)

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'few/' + XVEC_AHC, 
        "MSDWILD", "Few", "Baseline", XVEC_AHC)

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'many/' + XVEC_AHC, 
        "MSDWILD", "Many", "Baseline", XVEC_AHC)

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'few/' + XVEC_SC, 
        "MSDWILD", "Few", "Baseline", XVEC_SC)

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_BASELINE + 'many/' + XVEC_SC, 
        "MSDWILD", "Many", "Baseline", XVEC_SC)

#------------------------------- EXACT -------------------------------

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'few/' + ECAPA_AHC, 
        "MSDWILD", "Few", "Exact", ECAPA_AHC)  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'many/' + ECAPA_AHC, 
        "MSDWILD", "Many", "Exact", ECAPA_AHC)  

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'few/' + ECAPA_SC,
        "MSDWILD", "Few", "Exact", ECAPA_SC)  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'many/' + ECAPA_SC, 
        "MSDWILD", "Many", "Exact", ECAPA_SC)  

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'few/' + XVEC_AHC, 
        "MSDWILD", "Few", "Exact", XVEC_AHC)  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'many/' + XVEC_AHC, 
        "MSDWILD", "Many", "Exact", XVEC_AHC)  

metrics(msdwild_few_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'few/' + XVEC_SC, 
        "MSDWILD", "Few", "Exact", XVEC_SC)  

metrics(msdwild_many_TEST_GT_PATH, msdwild_OUTPUT_PATH_EXACT + 'many/' + XVEC_SC, 
        "MSDWILD", "Many", "Exact", XVEC_SC)  

# ===========================================================================
# =============================== EARNINGS-21 ===============================
# ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_AHC,
         "Earnings-21", "", "Baseline", ECAPA_AHC)  

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_BASELINE + 'test/' + ECAPA_SC, 
        "Earnings-21", "", "Baseline", ECAPA_SC)  

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_BASELINE + 'test/' + XVEC_AHC, 
        "Earnings-21", "", "Baseline", XVEC_AHC)  

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_BASELINE + 'test/' + XVEC_SC, 
        "Earnings-21", "", "Baseline", XVEC_SC)  


#------------------------------- EXACT -------------------------------

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EXACT + 'test/' + ECAPA_AHC, 
        "Earnings-21", "", "Exact", ECAPA_AHC)  

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EXACT + 'test/' + ECAPA_SC, 
        "Earnings-21", "", "Exact", ECAPA_SC)  

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EXACT + 'test/' + XVEC_AHC, 
        "Earnings-21", "", "Exact", XVEC_AHC)  

metrics(earn_TEST_GT_PATH, earn_OUTPUT_PATH_EXACT + 'test/' + XVEC_SC, 
        "Earnings-21", "", "Exact", XVEC_SC)  

# # ===========================================================================
# # =============================== ALIMEETING ===============================
# # ===========================================================================

#---------------------------- BASELINE ----------------------------

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + ECAPA_AHC, 
        "AliMeeting", "Far", "Baseline", ECAPA_AHC, True)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + ECAPA_AHC, 
        "AliMeeting", "Near", "Baseline", ECAPA_AHC, False)

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + ECAPA_SC, 
        "AliMeeting", "Far", "Baseline", ECAPA_SC, True)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + ECAPA_SC, 
        "AliMeeting", "Near", "Baseline", ECAPA_SC, False)

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + XVEC_AHC, 
        "AliMeeting", "Far", "Baseline", XVEC_AHC, True)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + XVEC_AHC, 
        "AliMeeting", "Near", "Baseline", XVEC_AHC, False)

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + XVEC_SC, 
        "AliMeeting", "Far", "Baseline", XVEC_SC, True)

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + XVEC_SC, 
        "AliMeeting", "Near", "Baseline", XVEC_SC, False)


#------------------------------- EXACT -------------------------------

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + ECAPA_AHC, 
        "AliMeeting", "Far", "", "Exact", ECAPA_AHC)  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + ECAPA_AHC, 
        "AliMeeting", "Near", "", "Exact", ECAPA_AHC)  

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + ECAPA_SC, 
        "AliMeeting", "Far", "", "Exact", ECAPA_SC)  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + ECAPA_SC, 
        "AliMeeting", "Near", "", "Exact", ECAPA_SC) 

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + XVEC_AHC, 
        "AliMeeting", "Far", "", "Exact", XVEC_AHC)  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + XVEC_AHC, 
        "AliMeeting", "Near", "", "Exact", XVEC_AHC) 

metrics(ali_far_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'far/' + XVEC_SC, 
        "AliMeeting", "Far", "", "Exact", XVEC_SC)  

metrics(ali_near_TEST_GT_PATH, ali_OUTPUT_PATH_BASELINE + 'near/' + XVEC_SC, 
        "AliMeeting", "Near", "", "Exact", XVEC_SC) 
# Source: https://github.com/aclew/utils/blob/master/textgrid2rttm.py
import os
import re

#================================= ALIMEETING ==================================
DATASET_PATH = '/path/to/datasets/folder/RAMC/MDT2021S003/'
  
def convert_to_rttm_ALI(path):
    subfolder = os.path.join(path, 'TXT')
    for file in os.listdir(subfolder):
        txt_file_path = os.path.join(subfolder, file)
        audio_name = file.split('.')[0]
        output_file_name = os.path.join(path, "rttm/", audio_name + '.rttm')
        
        spk_dir = {}
        spk_id = 0
        with open(txt_file_path, 'r') as txt:
            with open(output_file_name, 'w') as rttm:
                for line in txt:
                    interval, id, _, _ = line.rstrip().split('\t')
                    if id in spk_dir.keys():
                        spk = spk_dir[id]
                    else:
                        spk_dir[id] = spk_id
                        spk = spk_id
                        spk_id += 1


                    start_time = re.findall(r'\[.*?\,', interval)
                    start_time = start_time[0].replace("[","")
                    start_time = start_time.replace(",","")

                    end_time = re.findall(r'\,.*?\]', interval)
                    end_time = end_time[0].replace("]","")
                    end_time = end_time.replace(",","")

                    dur = float(end_time) - float(start_time)
                    
                    rttm.write("SPEAKER %s 1 %s %s <NA> <NA> %s <NA>\n" 
                               % (audio_name, start_time, str(dur), spk))
  


print("RAMC -- TEST SET", flush = True)
convert_to_rttm_ALI(DATASET_PATH)


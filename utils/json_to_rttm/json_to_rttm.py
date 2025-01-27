# Source: https://github.com/aclew/utils/blob/master/textgrid2rttm.py
import os
import re
import json

FILE_PATH = '/path/to/datasets/folder/This American Life - Test Set/test-transcripts-aligned.json'
OUTPUT_PATH = '/path/to/output/folder/This American Life - Test Set/rttm'

def convert_to_rttm_ALI(path):
    file = open(path)
    jsonFile = json.load(file)
    for episode in jsonFile:
        spk_dir = {}
        spk_id = 0

        episode_num = episode.split('-')[1]
        output_file_name = os.path.join(OUTPUT_PATH, episode_num + '.rttm')
        with open(output_file_name, 'w') as rttm:
            for content in jsonFile[episode]:
                speaker = content['speaker']
                start_time = content['utterance_start']
                duration = content['duration']

                if speaker in spk_dir.keys():
                    spk = spk_dir[speaker]
                else:
                    spk_dir[speaker] = spk_id
                    spk = spk_id
                    spk_id += 1

                rttm.write("SPEAKER %s 1 %s %s <NA> <NA> %s <NA>\n" 
                            % (episode_num, start_time, duration, spk))

convert_to_rttm_ALI(FILE_PATH)


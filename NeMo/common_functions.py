import os

def create_json_object(data_path, file, gt_path, exactNum = False, isFar = False):
    if isFar:
        file_name = file.split('_') 
        gt_file = file_name[0] + "_" + file_name[1] + '.rttm'
    else:
        file_name = file.split('.')[0] 
        gt_file = file_name + '.rttm'

        
    file_path = os.path.join(data_path, file)
    rttm_path = os.path.join(gt_path, gt_file)

    speakers = None
    if exactNum:
        print("IN exactNum", flush=True)        
        speakers = getNumberSpeakers(rttm_path)

    meta = {
        'audio_filepath': file_path, 
        'offset': 0, 
        'duration': None, 
        'label': 'infer', 
        'text': '-', 
        'num_speakers': speakers, 
        'rttm_filepath': rttm_path, 
        'uem_filepath' : None
    }
    return meta

def getNumberSpeakers(rttm_path):
    speakers = []
    
    with open(rttm_path, 'r') as rttm_file:
        for line in rttm_file:
            parts = line.strip().split()  # Split each line into parts
            
            speaker_label = parts[7]
            if speaker_label not in speakers:
                speakers.append(speaker_label)
    
    return len(speakers)
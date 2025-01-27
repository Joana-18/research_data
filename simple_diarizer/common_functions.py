import os
from pydub import AudioSegment
import io

def diarize(data_path, file, pipeline, exactNum = False, gt_path = '', 
            isFar = False):
    file_path = os.path.join(data_path, file)
    sound = file_path
    
    if file.split('.')[1] == 'flac':
        sound = AudioSegment.from_file(file_path, format='flac') 
        stream = io.BytesIO()
        sound.export(stream, format="wav")  
        
    if exactNum:
        if isFar:
            file_name = file.split('_') 
            gt_file = file_name[0] + "_" + file_name[1] + '.rttm'
        else:
            file_name = file.split('.')[0] 
            gt_file = file_name + '.rttm'
        num = getNumberSpeakers(gt_file, gt_path)
        diarization_result = pipeline.diarize(sound, num_speakers = num)

    else:
        diarization_result = pipeline.diarize(sound)
        
    return diarization_result

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
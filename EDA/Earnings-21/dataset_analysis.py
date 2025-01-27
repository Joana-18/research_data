import os
import librosa 
import librosa.display
import torch

from pyannote.audio import Pipeline
from pyannote.audio import Model
from pyannote.audio import Inference
import csv

AUTH_TOKEN = 'replace_with_your_auth_token'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = '/path/to/datasets/folder/Earnings-21/'
TEST_PATH = os.path.join(DATASET_PATH, 'wav/')

TEST_LABELS_PATH = os.path.join(DATASET_PATH, 'earnings-21/earnings21/rttms')
   
    
def create_csv(path, labels_path):
    header = ['Dataset', 'Duration', 'Speakers', 'Overlapping Speech', 'SNR', 'File']
    rows = []
    
    # Overlapping speech pipeline
    os_pipeline = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
                                        use_auth_token = AUTH_TOKEN)
    # SNR pipeline
    snr_model = Model.from_pretrained("pyannote/brouhaha", 
                                  use_auth_token = AUTH_TOKEN)
    snr_inference = Inference(snr_model)
    
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        
        
        audio, sample_rate = librosa.load(file_path)
        # Duration
        duration =  librosa.get_duration(y = audio, sr = sample_rate)
        
        # Number of Speakers
        gt_file_path = os.path.join(labels_path, file.split('.')[0] + '.rttm')
        
        speakers_audio = []
        num_speakers = 0
        with open(gt_file_path, 'r') as rttm_file:
            for line in rttm_file:
                parts = line.strip().split()  # Split each line into parts
                speaker_label = parts[7]
                if speaker_label not in speakers_audio:
                    speakers_audio.append(speaker_label)
            
            num_speakers = len(set(speakers_audio))
            
            
        # SNR
        output = snr_inference(file_path) 
        sum_snr = 0
        count = 0
        # iterate over each frame
        for _, (_, snr, _) in output:
            count += 1 
            sum_snr += snr
        mean_snr = round(sum_snr / count)  
        
        # Overlapping Speech
        output = os_pipeline(file_path)
        overlapping_time = 0
        for speech in output.get_timeline().support():
            overlapping_time += speech.duration
        percentage = round((overlapping_time / duration) * 100)
        
        
        rows.append(["Earnings", round(duration/60, 3), num_speakers, round(percentage, 3), round(mean_snr, 3), file.split('.')[0]])
       
    
    # Source https://www.pythontutorial.net/python-basics/python-write-csv-file/
    with open("Earnings_Statistics.csv", 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write multiple rows
        writer.writerows(rows)   
    
create_csv(TEST_PATH, TEST_LABELS_PATH)
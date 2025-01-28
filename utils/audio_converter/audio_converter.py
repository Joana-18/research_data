# Source: https://github.com/aclew/utils/blob/master/textgrid2rttm.py
import os
from pydub import AudioSegment
from sphfile import SPHFile

DATASET_PATH = '/path/to/datasets/folder/'
OUTPUT = '/path/to/output/folder/'

def flac_to_wav(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        sound = AudioSegment.from_file(file_path, format='flac')
    
        output_file_path = os.path.join(OUTPUT, file.split('.')[0] + '.wav')
        sound.export(output_file_path, format="wav")  

def mp3_to_wav(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        sound = AudioSegment.from_mp3(file_path) 
        
        output_file_path = os.path.join(OUTPUT, file.split('.')[0] + '.wav')
        sound.export(output_file_path, format="wav")  

def sph_to_wav(path):

    for folder in os.listdir(path):
        subfolder = os.path.join(path, folder)
        for file in os.listdir(subfolder):
            file_path = os.path.join(subfolder, file)
            
            output_dir = os.path.join(OUTPUT, folder)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file_path = os.path.join(output_dir, file.split('.')[0] + '.wav')

            sph = SPHFile(file_path)
            print(sph.format)
            sph.write_wav(output_file_path)

mp3_to_wav(DATASET_PATH)
flac_to_wav(DATASET_PATH)
sph_to_wav(DATASET_PATH)


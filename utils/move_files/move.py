import os

#================================= ALIMEETING ==================================
FILE_PATH_FEW = '/path/to/datasets/folder/MSDWILD/rttm/few'
FILE_PATH_MANY = '/path/to/datasets/folder/MSDWILD/rttm/many'
WAV_PATH = '/path/to/datasets/folder/MSDWILD/wav'
OUTPUT_PATH_FEW = '/path/to/datasets/folder/MSDWILD/wav/few'
OUTPUT_PATH_MANY = '/path/to/datasets/folder/MSDWILD/wav/many'
            
def move_wavs(path, output_path):
    for file in os.listdir(path):
        
        audio_name = file.split('.')[0]
        file_path = os.path.join(WAV_PATH, audio_name + '.wav')
        new_file_path = os.path.join(output_path, audio_name + '.wav')
        os.rename(file_path, new_file_path)

print("MSDWILD FEW -- TEST SET", flush = True)
move_wavs(FILE_PATH_FEW, OUTPUT_PATH_FEW)
print("MSDWILD MANY -- TEST SET", flush = True)
move_wavs(FILE_PATH_MANY, OUTPUT_PATH_MANY)


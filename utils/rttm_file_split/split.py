import os

#================================= ALIMEETING ==================================
FILE_PATH_FEW = '/path/to/datasets/folder/MSDWILD/MSDWILD/rttms/few.val.rttm'
FILE_PATH_MANY = '/path/to/datasets/folder/MSDWILD/MSDWILD/rttms/many.val.rttm'
OUTPUT_PATH_FEW = '/path/to/datasets/folder/MSDWILD/rttm/few'
OUTPUT_PATH_MANY = '/path/to/datasets/folder/MSDWILD/rttm/many'

def split_rttm(path, output_path):
    with open(path, 'r') as rttm_file:
        for line in rttm_file:
            
            parts = line.strip().split()  # Split each line into parts
            file = parts[1]
            output_file_name = os.path.join(output_path, file + '.rttm')

            start_time = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            
            with open(output_file_name, 'a+') as rttm:
                rttm.write("SPEAKER %s 1 %s %s <NA> <NA> %s <NA>\n" 
                            % (file, start_time, duration, speaker))
            
print("MSDWILD FEW -- TEST SET", flush = True)
split_rttm(FILE_PATH_FEW, OUTPUT_PATH_FEW)
print("MSDWILD MANY -- TEST SET", flush = True)
split_rttm(FILE_PATH_MANY, OUTPUT_PATH_MANY)


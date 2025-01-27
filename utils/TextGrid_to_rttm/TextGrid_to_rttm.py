# Source: https://github.com/aclew/utils/blob/master/textgrid2rttm.py
import os
import tgt

#================================= ALIMEETING ==================================
DATASET_PATH = '/path/to/datasets/folder/AliMeeting/Test_Ali/'
  
def convert_to_rttm_ALI(path):
    for folder in os.listdir(path):
        subfolder = os.path.join(path, folder, 'textgrid_dir/')
        for textgrid_file in os.listdir(subfolder):
            spk_id = 0
            textgrid_file_path = os.path.join(subfolder, textgrid_file)
            
            # init output
            rttm_out = dict()

            # open textgrid
            tg = tgt.read_textgrid(textgrid_file_path)

            # loop over all speakers in this text grid
            for spkr in tg.get_tier_names():
                spkr_timestamps = []

                # loop over all annotations for this speaker
                for _interval in tg.get_tiers_by_name(spkr):
                    for interval in _interval:

                        start_time, end_time = interval.start_time,\
                                                interval.end_time

                        spkr_timestamps.append((start_time,
                                    end_time - start_time))

                # add list of onsets, durations for each speakers
                rttm_out[spk_id] = spkr_timestamps    
                spk_id += 1                     

            audio_name = textgrid_file.split('.')[0]
            output_file_name = os.path.join(path, folder, "rttm_dir/", audio_name + '.rttm')
            with open(output_file_name, 'w') as fout:
                for spkr in rttm_out:
                    for bg, dur in rttm_out[spkr]:
                        fout.write(u'SPEAKER {} 1 {} {} <NA> <NA> {} <NA>\n'.format(
                                    output_file_name.split('/')[-1], bg, dur, spkr))    


print("ALIMEETING -- TEST SET", flush = True)
convert_to_rttm_ALI(DATASET_PATH)


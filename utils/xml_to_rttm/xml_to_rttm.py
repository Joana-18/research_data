#Base source code: https://github.com/cunnie/bin/blob/26eecbc292fc9066be0554447fe69afcafd2295c/nite_xml_to_rttm.py
import os
import xml.etree.ElementTree as ET

#===================================== AMI =====================================
DATASET_PATH = '/path/to/datasets/folder/AMI/'
TEST_PATH = os.path.join(DATASET_PATH, 'test')
LABELS_PATH = os.path.join(DATASET_PATH, 'Labels/manual/segments/')
TEST_LABELS_PATH = os.path.join(DATASET_PATH, 'Labels/test/')

def convert_xml_to_rttm_AMI(path, output_path):
    for audio_name in os.listdir(path):
        lines = []
        xml_trees = []
        
        for xml_file in os.listdir(LABELS_PATH):
            xml_file_indices = xml_file.split('.')
            
            # Find all corresponding segment files 
            if xml_file_indices[0] == audio_name:
                xml_file_index = xml_file_indices[1] # Get speaker ID
                
                if xml_file_index == 'A':
                    spk_id = 0
                elif xml_file_index == 'B':
                    spk_id = 1
                elif xml_file_index == 'C':
                    spk_id = 2
                elif xml_file_index == 'D':
                    spk_id = 3
                elif xml_file_index == 'E':
                    spk_id = 4
                xml_file_path = os.path.join(LABELS_PATH, xml_file)
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                xml_trees.append(root)
                
                start_time = end_time = None
                for element in root:
                    if element.tag == 'segment' and 'transcriber_start' in element.attrib and 'transcriber_end' in element.attrib:
                        start_time = float(element.attrib['transcriber_start'])
                        end_time = float(element.attrib['transcriber_end'])
                        lines.append("SPEAKER meeting\t1\t{}\t{}\t<NA>\t<NA>\tSpeaker_{}\t<NA>\n".format(
                            start_time, end_time - start_time, spk_id))
                            
        with open(os.path.join(output_path, audio_name + '.rttm'), 'w') as rttm:
            for seg in lines:
                rttm.write(str(seg))        

print("AMI -- TEST SET", flush = True)
convert_xml_to_rttm_AMI(TEST_PATH, TEST_LABELS_PATH)

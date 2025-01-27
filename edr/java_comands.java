/////////////////////////////
java caren 'research_data\edr\data\rules\Baseline_DER__pyaudio.csv' 0.1 0.5 -Att -s',' -Dist -imp'0.001' -GDER -classDER -ocs'research_data\edr\edrs\Baseline_DER_pyaudio'
java caren 'research_data\edr\data\rules\Baseline_Coverage__pyaudio.csv' 0.1 0.5 -Att -s',' -Dist -imp'0.001' -GCoverage -classCoverage -ocs'research_data\edr\edrs\Baseline_Coverage_pyaudio'


////////////////////////////// Global comparison - top 3 methods

java caren 'research_data\edr\data\rules\Baseline_DER_3.1_pyannote.csv' 0.10 0.5 -Att -s',' -Dist -imp'0.001' -GDER -classDER -ocs'research_data\edr\edrs\Baseline_DER_3.1_pyannote'
java caren 'research_data\edr\data\rules\Baseline_JER_3.1_pyannote.csv' 0.10 0.5 -Att -s',' -Dist -imp'0.001' -GJER -classJER -ocs'research_data\edr\edrs\Baseline_JER_3.1_pyannote'
java caren 'research_data\edr\data\rules\Baseline_DER__nemo_Clustering.csv' 0.10 0.5 -Att -s',' -Dist -imp'0.001' -GDER -classDER -ocs'research_data\edr\edrs\Baseline_DER_nemo_Clustering'
java caren 'research_data\edr\data\rules\Baseline_JER__nemo_Clustering.csv' 0.10 0.5 -Att -s',' -Dist -imp'0.001' -GJER -classJER -ocs'research_data\edr\edrs\Baseline_JER_nemo_Clustering'
java caren 'research_data\edr\data\rules\Baseline_DER__nemo_Joint.csv' 0.10 0.5 -Att -s',' -Dist -imp'0.001' -GDER -classDER -ocs'research_data\edr\edrs\Baseline_DER_nemo_Joint'
java caren 'research_data\edr\data\rules\Baseline_JER__nemo_Joint.csv' 0.10 0.5 -Att -s',' -Dist -imp'0.001' -GJER -classJER -ocs'research_data\edr\edrs\Baseline_JER_nemo_Joint'

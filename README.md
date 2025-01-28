# research_data
 
The data used in our work was retrieved from the following sources:
- AMI: https://groups.inf.ed.ac.uk/ami/corpus/
- AISHELL-4: https://openslr.org/111/
- AliMeeting: https://openslr.org/119/
- VoxConverse: https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
- This American Life: https://www.kaggle.com/datasets/shuyangli94/this-american-life-podcast-transcriptsalignments?select=download_page_snapshot.html
- RAMC: https://github.com/MagicHub-io/MagicData-RAMC
- MSDWild: https://github.com/X-LANCE/MSDWILD/tree/master
- Earnings-21: https://github.com/revdotcom/speech-datasets/tree/main/earnings21

Caren software:
- https://www.di.uminho.pt/~pja/class/caren.html

This repository includes all code necessary to:
- perform an exploratory data analysis on SNR, audio duration, SNR, and overlapping speech
- diarize the data sets using NeMo, pyannote, pyAudioAnalysis, simple_diarizer, and WhisperX
- evaluate the results on all data sets and plot them
- apply error distribution rules to the obtained results distributions
- perform significance tests
- convert audio to mono when necessary
- convert the audio files from flac, mp3, and sph to wav
- convert the groundtruth files from textgrid, json, txt, and xml to rttm
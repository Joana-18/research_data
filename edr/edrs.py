import pandas as pd
import re

# WHISPER_PYA = True
WHISPER_PYA = False
APPROACH = 'Baseline'
# METRICS = ['DER']
METRICS = ['JER']
# METRICS = ['Coverage']
# METHOD = 'pyannote'
# METHOD = 'WhisperX'
# METHOD = 'pyaudio'
METHOD = 'nemo'
DIARIZER = 'Joint'
# DIARIZER = 'Clustering'


# ===============================================================
# ==================== Statistics ANALYSIS =====================
# ===============================================================
# If analysing WhisperX or pyannote, apply a threshold of 70% to DER
if WHISPER_PYA:
	MIN_DER = 70.0
	FILE_NAME = ''
else:
	MIN_DER = 0
	if METHOD == 'pyaudio':
		FILE_NAME = ''
	else:
		FILE_NAME = '_rq3'
PATH = '\\research_data\\edr\\data\\'
if METHOD == 'WhisperX':
	FILE = '\\research_data\\metrics\\WhisperX_for_edrs.csv'
else:
	if METHOD == 'PyAnnote':
		FILE = '\\research_data\\metrics\\pyannote_for_edrs.csv'
	if METHOD == 'pyaudio':
		FILE = '\\research_data\\metrics\\pyAudioAnalysis_for_edrs.csv'
	if METHOD == 'nemo':
		FILE = '\\research_data\\metrics\\NeMo_for_edrs.csv'

# Data sets' statistics
AISHELL = pd.read_csv(f"{PATH}\\AISHELL_Statistics.csv",
					  dtype={
    'File': 'string'
})
ALI_FAR = pd.read_csv(f"{PATH}\\Ali_Far_Statistics.csv",
					  dtype={
    'File': 'string'
})
ALI_NEAR = pd.read_csv(f"{PATH}\\Ali_Near_Statistics.csv",
					  dtype={
    'File': 'string'
})
AMI = pd.read_csv(f"{PATH}\\AMI_Statistics.csv",
					  dtype={
    'File': 'string'
})
EARNINGS = pd.read_csv(f"{PATH}\\Earnings_Statistics.csv",
					  dtype={
    'File': 'string'
})
MSD_FEW = pd.read_csv(f"{PATH}\\MSD_Few_Statistics.csv",
					  dtype={
    'File': 'string'
})
MSD_MANY = pd.read_csv(f"{PATH}\\MSD_Many_Statistics.csv",
					  dtype={
    'File': 'string'
})
RAMC = pd.read_csv(f"{PATH}\\RAMC_Statistics.csv",
					  dtype={
    'File': 'string'
})
TAL = pd.read_csv(f"{PATH}\\TAL_Statistics.csv",
					  dtype={
    'File': 'string'
})
VOXCONVERSE = pd.read_csv(f"{PATH}\\VOX_Statistics.csv",
					  dtype={
    'File': 'string'
})


metrics_all = pd.read_csv(
	FILE,
	dtype={
		'Method': 'string',
		'Metric': 'string',
		'Version': 'float',
		'File': 'string'
})
if METHOD == 'WhisperX' or METHOD == 'PyAnnote':
	VERSION = 3.1
	metrics_all = metrics_all[['Data Set', 'Version', 'Approach', 'Method', 'Metric', 'Value', 'File']]
	metrics_all = metrics_all[metrics_all['Version'] == VERSION]
	DIARIZER = ''
else:
	VERSION = ''
	if METHOD == 'SD':
		metrics_all = metrics_all[['Data Set', 'Approach', 'Method', 'Metric', 'Value', 'File', 'Specs']]
		metrics_all = metrics_all[metrics_all['Specs'] == 'ecapa_ahc']
		DIARIZER = ''
	else:
		metrics_all = metrics_all[['Data Set', 'Approach', 'Method', 'Metric', 'Value', 'File']]
		if METHOD == 'nemo':
			if DIARIZER == 'Joint':
				metrics_all = metrics_all[metrics_all['Method'] == 'Joint (ASR-based TS)']
			else:
				metrics_all = metrics_all[metrics_all['Method'] == 'Clustering']
			print("\n", metrics_all['Method'])
metrics_all['Value'] = round(metrics_all['Value'] * 100, 3)
print("METRICS pre filtering\n", metrics_all)

metrics = metrics_all[(metrics_all['Value'] >= MIN_DER) & 
					  (metrics_all['Metric'] == METRICS[0]) &
					  (metrics_all['Approach'] == APPROACH)]

print("METRICS post filtering\n", metrics['Data Set'].value_counts())
files =  metrics['File']

print("METRICS post filtering - DATASETS", metrics['Data Set'].unique())

datasets = pd.concat([AISHELL, AMI, VOXCONVERSE, ALI_FAR, ALI_NEAR, EARNINGS,
					  MSD_FEW, MSD_MANY, RAMC, TAL]).reset_index(drop=True)

for metric in METRICS:

	performance_metrics_df = datasets.merge(
		metrics.loc[
			(metrics['Approach'] == APPROACH) & 
			(metrics['Metric'] == metric) 
			],
		how = 'left',
		on = ['Data Set', 'File']
		)[['Duration', 'Speakers', 'Overlapping Speech', 'SNR', 'Value']].rename(
			columns = {'Value' : metric, 'Duration': 'D',
						'Speakers': 'Spk', 
						'Overlapping Speech': 'OS'})

	print("AFTER MERGE\n", performance_metrics_df)
	# Count NA
	print("NA SUM\n", performance_metrics_df[['D', 'Spk', 'OS', 'SNR', metric]].isna().sum())
	# drop NA
	performance_metrics_df.dropna(inplace = True)

	print("AFTER NA REMOVAL\n", performance_metrics_df)

	# Discretize numeric features using quantiles
	for feature in ['D', 'Spk', 'OS', 'SNR']:
		performance_metrics_df[feature] = pd.qcut(performance_metrics_df[feature], 10, 
												duplicates = 'drop').astype(str)

		interval_list = []
		for interval in performance_metrics_df[feature].values:
			interval = interval.replace("(", "")
			interval = interval.replace("]", "")

			num_1 = re.search('(.+?),', interval)
			if num_1 is not None:
				num_1 = num_1.group(0).replace(",", "")
				num_1 = float(num_1)

			num_2 = re.search(', (.*)', interval)
			if num_2 is not None:
				num_2 = num_2.group(0).replace(", ", "")
				num_2 = float(num_2)

			if feature == 'Spk' and num_2 is not None and num_1 is not None:
				if num_1 < 1:
					new_interval = "[" + str(round(num_1)) + ", " +  str(round(num_2)) + "]"
				else:
					new_interval = "(" + str(round(num_1)) + ", " +  str(round(num_2)) + "]"
			else:
				if num_1 is not None and num_1 < 0 and num_2 is not None:
					new_interval = "[0.0, " +  str(num_2) + "]"
				else:
					if num_2 is not None and num_1 is not None:
						new_interval = "(" + str(num_1) + ", " +  str(num_2) + "]"
			interval_list.append(new_interval)

		performance_metrics_df[feature] = interval_list
		performance_metrics_df[feature] = performance_metrics_df[feature].apply(lambda x : x.replace(', ', ' - '))


	# To csv
	performance_metrics_df.to_csv(
		f'\\research_data\\edr\\data\\rules\\{APPROACH}_{metric}_{VERSION}_{METHOD}_{DIARIZER}_{FILE_NAME}.csv', 
		index = False)

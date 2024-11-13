import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


manual_counts = pd.read_csv('../../one_microphone_recordings/manual_count_one_mic_recs.txt', sep='\t')


algorithm_results = pd.read_csv('../../one_microphone_recordings/main_stopping_breathing3.txt', sep=',')

algorithm_results['file_name'] = algorithm_results['file_name'].str.replace('.wav', '', regex=False)


data = pd.merge(
    algorithm_results,
    manual_counts,
    left_on='file_name',
    right_on='name_of_file',
    suffixes=('_algorithm', '_manual')
)


numerical_cols = ['window_size', 'sigma', 'zero_threshold', 'breath_threshold_factor',
                  'inhale_count_algorithm', 'exhale_count_algorithm',
                  'inhale_count_manual', 'exhale_count_manual']
for col in numerical_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')


if data[numerical_cols].isnull().values.any():
    print("Uwaga: Wystąpiły wartości nienumeryczne w kolumnach liczbowych. Proszę sprawdzić dane wejściowe.")
    
    data[numerical_cols] = data[numerical_cols].fillna(0)


data['inhale_error'] = abs(data['inhale_count_algorithm'] - data['inhale_count_manual'])
data['exhale_error'] = abs(data['exhale_count_algorithm'] - data['exhale_count_manual'])
data['total_error'] = data['inhale_error'] + data['exhale_error']


grouped = data.groupby(['window_size', 'sigma', 'zero_threshold', 'breath_threshold_factor'], as_index=False)[['inhale_error', 'exhale_error', 'total_error']].mean()


pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 20)       


top_combinations = grouped.sort_values(by=['exhale_error','total_error' ]).head(108)
print("Najlepsze kombinacje parametrów według total_error i exhale_error:\n", top_combinations)




param_counts = {
    'window_size': top_combinations['window_size'].value_counts(),
    'sigma': top_combinations['sigma'].value_counts(),
    'zero_threshold': top_combinations['zero_threshold'].value_counts(),
    'breath_threshold_factor': top_combinations['breath_threshold_factor'].value_counts()
}


for param, counts in param_counts.items():
    print(f"\nCzęstotliwość dla parametru '{param}':")
    print(counts)



pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')

optimal_params = grouped.loc[grouped['total_error'].idxmin()]
print("Optymalne parametry:")
print(optimal_params)


plt.figure(figsize=(10, 6))
plt.plot(grouped['window_size'], grouped['exhale_error'], marker='o')
plt.xlabel("Window Size")
plt.ylabel("Total Error")
plt.title("Total Error vs Window Size")
plt.grid(True)
plt.savefig("window_size_to_exhale_error")
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(grouped['sigma'], grouped['total_error'], marker='o', color='orange')
plt.xlabel("Sigma")
plt.ylabel("Total Error")
plt.title("Total Error vs Sigma")
plt.grid(True)
plt.savefig("sigma_to_total_error")
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(grouped['zero_threshold'], grouped['total_error'], marker='o', color='green')
plt.xlabel("Zero Threshold")
plt.ylabel("Total Error")
plt.title("Total Error vs Zero Threshold")
plt.grid(True)
plt.savefig("zero_threshold_to_total_error")
plt.show()



plt.figure(figsize=(10, 6))
plt.plot(grouped['breath_threshold_factor'], grouped['total_error'], marker='o', color='red')
plt.xlabel("Breath Threshold Factor")
plt.ylabel("Total Error")
plt.title("Total Error vs Breath Threshold Factor")
plt.grid(True)
plt.savefig("breath_threshold_factor_to_total_error")
plt.show()

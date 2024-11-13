import csv
from collections import Counter


input_file = '../parameters_checker/best_results_video_in_background_all_atributes_values.txt'  
output_file = '../parameters_checker/usage_of_parameters_video_in_background_all_atributes_values.txt'

window_size_counter = Counter()
sigma_counter = Counter()
zero_threshold_counter = Counter()
breath_threshold_factor_counter = Counter()


with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        
        window_size_counter[row['window_size']] += 1
        sigma_counter[row['sigma']] += 1
        zero_threshold_counter[row['zero_threshold']] += 1
        breath_threshold_factor_counter[row['breath_threshold_factor']] += 1


print("Najczęstsze wartości parametrów w najlepszych wynikach:")
print("Window Size:", window_size_counter.most_common())
print("Sigma:", sigma_counter.most_common())
print("Zero Threshold:", zero_threshold_counter.most_common())
print("Breath Threshold Factor:", breath_threshold_factor_counter.most_common())
with open(output_file, 'w') as f:
    f.write(f"Most usages in best results:\n")
    f.write(f"Window Size: {window_size_counter.most_common()}\n")
    f.write(f"Sigma: {sigma_counter.most_common()}\n")
    f.write(f"Zero Threshold: {zero_threshold_counter.most_common()}\n")
    f.write(f"Breath Threshold Factor: {breath_threshold_factor_counter.most_common()}\n")

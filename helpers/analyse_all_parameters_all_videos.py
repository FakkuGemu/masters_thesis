import os
import csv


results_folder = 'parameters_results'  
actual_counts_file = '../one_microphone_recordings/manual_count_one_mic_recs.txt'  
output_file = 'best_parameters_summary.txt'


actual_counts = {}
with open(actual_counts_file, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  
    for row in reader:

        file_name = row[0]
        print(file_name)
        inhale_count = int(row[1])
        exhale_count = int(row[2])
        actual_counts[file_name] = (inhale_count, exhale_count)


best_parameters = []


for result_file in os.listdir(results_folder):
    if result_file.endswith('.txt'):
        file_path = os.path.join(results_folder, result_file)

        
        file_name = result_file.replace('.txt', '')
        print(file_name)
        
        if file_name in actual_counts:
            actual_inhale, actual_exhale = actual_counts[file_name]
            print(actual_exhale, actual_inhale)
            min_error = float('inf')
            best_params = None

            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    window_size = row['window_size']
                    sigma = row['sigma']
                    zero_threshold = row['zero_threshold']
                    breath_threshold_factor = row['breath_threshold_factor']
                    inhale_count = int(row['inhale_count'])
                    exhale_count = int(row['exhale_count'])

                    
                    inhale_diff = abs(inhale_count - actual_inhale)
                    exhale_diff = abs(exhale_count - actual_exhale)
                    total_error = inhale_diff + exhale_diff

                    
                    if total_error < min_error:
                        min_error = total_error
                        best_params = {
                            'file_name': file_name,
                            'window_size': window_size,
                            'sigma': sigma,
                            'zero_threshold': zero_threshold,
                            'breath_threshold_factor': breath_threshold_factor,
                            'inhale_count': inhale_count,
                            'exhale_count': exhale_count,
                            'error': total_error
                        }

            
            if best_params:
                best_parameters.append(best_params)


with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_name', 'window_size', 'sigma', 'zero_threshold', 'breath_threshold_factor', 'inhale_count',
                     'exhale_count', 'error'])

    for params in best_parameters:
        writer.writerow([
            params['file_name'], params['window_size'], params['sigma'],
            params['zero_threshold'], params['breath_threshold_factor'],
            params['inhale_count'], params['exhale_count'], params['error']
        ])

print(f"Najlepsze parametry zapisano do pliku {output_file}.")

import csv


input_file = '../parameters_checker/breathing_analysis_results_fast_breathing.txt'
output_file = 'parameters_checker\\best_results_fast_breathing_inhale_exhale.txt'


target_inhale_count = 10
target_exhale_count = 10


correct_exhale_results = []
min_inhale_difference = float('inf')


with open(input_file, 'r') as f:
    reader = csv.DictReader(f)

    for row in reader:
        inhale_count = int(row['inhale_count'])
        exhale_count = int(row['exhale_count'])


        if exhale_count == target_exhale_count:

            inhale_difference = abs(inhale_count - target_inhale_count)


            if inhale_difference < min_inhale_difference:
                min_inhale_difference = inhale_difference
                correct_exhale_results = [
                    row]
            elif inhale_difference == min_inhale_difference:
                correct_exhale_results.append(row)


with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['file_name', 'window_size', 'sigma', 'zero_threshold', 'breath_threshold_factor', 'inhale_count',
                     'exhale_count'])

    for result in correct_exhale_results:
        writer.writerow([
            result['file_name'], result['window_size'], result['sigma'],
            result['zero_threshold'], result['breath_threshold_factor'],
            result['inhale_count'], result['exhale_count']
        ])

print(
    f"Wyniki z najbliższymi wartościami do {target_inhale_count} wdechów i {target_exhale_count} wydechów zostały zapisane do pliku {output_file}.")

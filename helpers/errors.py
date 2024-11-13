import numpy as np


real_inhale = np.array([6,
7,
5,
1,
5,
0,
2,
15,
                        25,
                        10])
real_exhale = np.array([7,
7,
8,
5,
7,
6,
15,
22,
                        26,
                        10])
detected_inhale = np.array([6,
7,
6,
3,
4,
1,
9,
10,
                            12,
                            7])
detected_exhale = np.array([7,
7,
8,
5,
7,
6,
15,
22,
                            25,
                            10])


inhale_diff = real_inhale - detected_inhale
exhale_diff = real_exhale - detected_exhale



mae_inhale = inhale_diff.mean()
mae_exhale = exhale_diff.mean()


combined_mae = np.concatenate((inhale_diff, exhale_diff)).mean()


std_inhale = inhale_diff.std()
std_exhale = exhale_diff.std()


combined_std = np.concatenate((inhale_diff, exhale_diff)).std()

print("MAE Inhale:", mae_inhale)
print("MAE Exhale:", mae_exhale)
print("Combined MAE:", combined_mae)
print("STD Inhale:", std_inhale)
print("STD Exhale:", std_exhale)
print("Combined STD:", combined_std)

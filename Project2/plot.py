import os
import matplotlib.pyplot as plt

DATASET_ROOT = os.path.join(os.getcwd(), "prep_dataset")
subset_paths = [os.path.join(DATASET_ROOT, subset) for subset in os.listdir(DATASET_ROOT)]

class_folders = os.listdir(subset_paths[0])

class_counts: dict[str, int] = {
    name: 0
    for name in class_folders
}

for subset_path in subset_paths:
    class_counts = {
        name: count + len(os.listdir(os.path.join(subset_path, name)))
        for name, count in class_counts.items()
    }

UNKNOWN_SUBCLASSES = [
    'bed',
    'bird',
    'cat',
    'dog',
    'happy',
    'house',
    'marvin',
    'sheila',
    'tree',
    'wow',
]
SILENCE_CLASSES = [
    "doing_the_dishes", "dude_miaowing", "exercise_bike", "pink_noise", "running_tap", "white_noise",
]

silence_total = 0
for name in SILENCE_CLASSES:
    silence_total += class_counts[name]
    del class_counts[name]

class_counts['silence'] = silence_total

colors = [
    'tab:blue' if name not in UNKNOWN_SUBCLASSES
    else 'tab:purple'
    for name in class_counts.keys()
]

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(list(class_counts.keys()), list(class_counts.values()), color=colors)

ax.set_ylabel('Number of distances', fontsize=18)
ax.set_title('Class distribution of Speech Command Dataset with generated $silence$ class', fontsize=18, pad=20)
ax.tick_params(axis='x', labelrotation=45, labelsize=10)

plt.show()

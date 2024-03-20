from datasets import load_dataset
from datasets import DatasetDict

# Load the original dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# Shuffle the dataset to ensure random selection of samples
dataset = dataset.shuffle(seed=42)

# Select 100 samples for the test set
test_dataset = dataset.select(range(100))

# Select the rest for the train set
train_dataset = dataset.select(range(100, len(dataset)))

# Create a DatasetDict to organize our new splits
new_splits = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Save the new splits to disk for later use
new_splits.save_to_disk('./alpaca_data')


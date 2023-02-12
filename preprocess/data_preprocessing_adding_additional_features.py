####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Data Preprocessing - Add other important data from dataset1 to pe_files_dataset ####

import csv

dataset1_path = "/home/nimna/Documents/MyProjects/DatasetMalDitectist/DatasetCreationPEFiles/dataset1.csv"
pe_files_dataset_path = "/home/nimna/Documents/MyProjects/DatasetMalDitectist/DatasetCreationPEFiles/pe_files_dataset.csv"
combined_dataset_path = "/home/nimna/Documents/MyProjects/DatasetMalDitectist/DatasetCreationPEFiles/combined_dataset.csv"

# Read the first dataset into memory
with open(dataset1_path) as f:
    reader = csv.DictReader(f)
    data1 = {row["id"]: row for row in reader}

# Read the second dataset into memory
with open(pe_files_dataset_path) as f:
    reader = csv.DictReader(f)
    data2 = list(reader)

# Add data from the first dataset to the second dataset
header_names = list(data2[0].keys()) + list(list(data1.values())[0].keys())
combined_data = []
processed_files = 0

for row in data2:
    try:
        file_data = data1[row["file_name"]]
        row.update(file_data)
        processed_files += 1
        print(f"Processed {processed_files} files")
        combined_data.append(row)
    except KeyError:
        print(f"No data found for file {row['file_name']}")

# Write the combined data to a new file
with open(combined_dataset_path, "w") as f:
    writer = csv.DictWriter(f, fieldnames=header_names)
    writer.writeheader()
    writer.writerows(combined_data)

# Print the number of processed files
print(f"Processed {processed_files} files. Dataset Creation Successfully Completed!!!")



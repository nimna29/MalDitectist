####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Data Pre-processing PE Files ####

import os
import pefile
import pandas as pd

# Create an empty DataFrame to store the header information
columns = ["FileName", "SizeOfImage"]

header_data = pd.DataFrame(columns=columns)
# header_data -> PE Header File Data

# Loop over all of the PE files in the directory
directory = "/media/nimna/New Volume1/Malware_Dataset"
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    # Try to extract the header information from the file
    try:
        pe = pefile.PE(file_path)
        
        size_of_image = pe.OPTIONAL_HEADER.SizeOfImage
        
        # Store the header information in the DataFrame
        header_data = header_data.append({"FileName": filename, 
                                          "SizeOfImage": size_of_image}, ignore_index=True)
        
        print(f"Successful extract data from PE file: {file_path}")
    except Exception as e:
        print(f"Error processing file: {file_path}")
        print(e)
        
# Save the header information to a CSV file
header_data.to_csv("header_data.csv", index=False)


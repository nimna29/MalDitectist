####### MalDitectist ########
## Developed by Nimna Niwarthana ##
#### Data Extraction From PE Files ####

import pefile
import os
import pandas as pd

directory = "/media/nimna/New Volume1/Malware_Dataset/"
features_list = []
error_files = []
batch_count = 0
batch_size = 1500

for filename in os.listdir(directory):
    full_path = os.path.join(directory, filename)
    try:
        with open(full_path, 'rb') as f:
            pe = pefile.PE(full_path)
            features = {
                'file_name': filename,
                'machine_type': pe.FILE_HEADER.Machine,
                'number_of_sections': pe.FILE_HEADER.NumberOfSections,
                'timestamp': pe.FILE_HEADER.TimeDateStamp,
                'pointer_to_symbol_table': pe.FILE_HEADER.PointerToSymbolTable,
                'number_of_symbols': pe.FILE_HEADER.NumberOfSymbols,
                'size_of_optional_header': pe.FILE_HEADER.SizeOfOptionalHeader,
                'characteristics': pe.FILE_HEADER.Characteristics,
                'iat_rva': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IAT']].VirtualAddress,
                'major_version': pe.OPTIONAL_HEADER.MajorLinkerVersion,
                'minor_version': pe.OPTIONAL_HEADER.MinorLinkerVersion,
                'check_sum': pe.OPTIONAL_HEADER.CheckSum,
                'compile_date': pe.FILE_HEADER.TimeDateStamp,
                'datadir_IMAGE_DIRECTORY_ENTRY_BASERELOC_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_BASERELOC']].Size,
                'datadir_IMAGE_DIRECTORY_ENTRY_EXPORT_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].Size,
                'datadir_IMAGE_DIRECTORY_ENTRY_IAT_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IAT']].Size,
                'datadir_IMAGE_DIRECTORY_ENTRY_IMPORT_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].Size,
                'debug_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DEBUG']].Size,
                'export_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']].Size,
                'size_of_code': pe.OPTIONAL_HEADER.SizeOfCode,
                'size_of_initialized_data': pe.OPTIONAL_HEADER.SizeOfInitializedData,
                'size_of_uninitialized_data': pe.OPTIONAL_HEADER.SizeOfUninitializedData,
                'size_of_image': pe.OPTIONAL_HEADER.SizeOfImage,
                'size_of_headers': pe.OPTIONAL_HEADER.SizeOfHeaders,
                'subsystem': pe.OPTIONAL_HEADER.Subsystem,
                'major_operating_system_version': pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
                'minor_operating_system_version': pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
                'number_of_rva_and_sizes': pe.OPTIONAL_HEADER.NumberOfRvaAndSizes,
                'base_of_code': pe.OPTIONAL_HEADER.BaseOfCode,
                'entry_point_rva': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
                'resource_size': pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']].Size,
                'size_of_heap_commit': pe.OPTIONAL_HEADER.SizeOfHeapCommit,
                'size_of_heap_reserve': pe.OPTIONAL_HEADER.SizeOfHeapReserve,
                'size_of_stack_commit': pe.OPTIONAL_HEADER.SizeOfStackCommit,
                'size_of_stack_reserve': pe.OPTIONAL_HEADER.SizeOfStackReserve,
                'status': pe.OPTIONAL_HEADER.DllCharacteristics,
            }
            print(f"Data Collection Successfully: {filename}")
            features_list.append(features)
            if len(features_list) == batch_size:
                df = pd.DataFrame(features_list)
                df.to_csv(f"/media/nimna/New Volume1/MalDitectist_dataset/PE_files_dataset/pe_files_dataset_{batch_count}.csv", index=False)
                features_list = []
                batch_count += 1
                print(f"Dataset Batch {batch_count} Created!!!!")
    except pefile.PEFormatError as e:
        print(f"Error while processing {filename}: {e}")
    except Exception as e:
        print(f"Error while processing file: {filename}")
        print(f"Error message: {str(e)}")
        error_files.append(filename)
        continue

if features_list:
    df = pd.DataFrame(features_list)
    df.to_csv(f"/media/nimna/New Volume1/MalDitectist_dataset/PE_files_dataset/pe_files_dataset_{batch_count}.csv", index=False)

# Combine all batch files into a single file
file_list = [f"/media/nimna/New Volume1/MalDitectist_dataset/PE_files_dataset/pe_files_dataset_{i}.csv" for i in range(batch_count + 1)]
df = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)
df.to_csv("/media/nimna/New Volume1/MalDitectist_dataset/PE_files_dataset/pe_files_dataset.csv", index=False)

# Saving Error Files data to a csv file
error_files_df = pd.DataFrame(error_files, columns=['Error_files'])
error_files_df.to_csv("/media/nimna/New Volume1/MalDitectist_dataset/PE_files_dataset/pe_error_files.csv", index=False)

print("PE Files Data Collection Successfully Completed!!!")



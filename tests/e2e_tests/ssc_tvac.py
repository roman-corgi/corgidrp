'''Script Kevin used to help gather and organize SSC TVAC data for use in CGI DRP end-to-end tests.'''

from astropy.io import fits
import numpy as np
import os
import re
import csv
import corgidrp.data as data
output_base_dir = '/Users/kevinludwick/Documents/ssc_tvac_test/E2E_Test_Data2'
tvac_dir = '/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/Working_Folder/'
ssc_dir = '/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/ssc_331'

if __name__ == "__main__":
    if False:
        dest_dir = '/Users/kevinludwick/Documents/ssc_tvac_test/E2E_Test_Data3'
        dest_kgain = os.path.join(dest_dir, 'TV-20_EXCAM_noise_characterization', 'nonlin', 'kgain')
        dest_nonlin = os.path.join(dest_dir, 'TV-20_EXCAM_noise_characterization', 'nonlin')
        dest_darkmap = os.path.join(dest_dir, 'TV-20_EXCAM_noise_characterization', 'darkmap')
        dest_noisemap = os.path.join(dest_dir, 'TV-20_EXCAM_noise_characterization', 'noisemap_test_data', 'test_l1_data')
        source_kgain = os.path.join(output_base_dir, 'TV-20_EXCAM_noise_characterization', 'nonlin', 'kgain')
        source_nonlin = os.path.join(output_base_dir, 'TV-20_EXCAM_noise_characterization', 'nonlin')
        source_darkmap = os.path.join(output_base_dir, 'TV-20_EXCAM_noise_characterization', 'darkmap')
        source_noisemap = os.path.join(output_base_dir, 'TV-20_EXCAM_noise_characterization', 'noisemap_test_data', 'test_l1_data')
        for dest in [(dest_kgain,source_kgain), (dest_nonlin,source_nonlin), (dest_darkmap,source_darkmap), (dest_noisemap,source_noisemap)]:
            for file in os.listdir(dest[1]):
                if not file.lower().endswith('.fits'):
                    continue
                ssc_filepath = os.path.join(ssc_dir, file.split('_')[1], file.lower())
                t_indices = [m.start() for m in re.finditer('t', ssc_filepath)]
                ssc_filepath = ssc_filepath[:t_indices[-2]] + 'T' + ssc_filepath[t_indices[-2]+1:]
                dest_filepath = os.path.join(dest[0], file.lower())
                with fits.open(ssc_filepath) as hdul:
                    hdul.writeto(dest_filepath, overwrite=True)
                    
    if False:
        datetime_list = []
        
        kgain_dir = os.path.join(output_base_dir, 'TV-20_EXCAM_noise_characterization', 'nonlin', 'kgain')
        kgain_list = [os.path.join(kgain_dir, f) for f in os.listdir(kgain_dir) if f.lower().endswith('.fits')]
        kgain_dataset = data.Dataset(kgain_list)
        sets, unique_vals = kgain_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        for Set in sets: 
            for file in Set:
                if file.ext_hdr['DATETIME'] in datetime_list:
                    os.remove(file.filepath)
                else:
                    datetime_list.append(file.ext_hdr['DATETIME'])
        #redo now that we've removed files with duplicate DATETIME values
        # keep only sets that are at least 5 files long
        datetime_list = []
        kgain_list = [os.path.join(kgain_dir, f) for f in os.listdir(kgain_dir) if f.lower().endswith('.fits')]
        sets, unique_vals = kgain_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        for Set in sets:
            if len(Set) < 5:
                for file in Set:
                    os.remove(file.filepath)
            else:
                for file in Set:
                    datetime_list.append(file.ext_hdr['DATETIME'])
            

        nonlin_dir = os.path.join(output_base_dir, 'TV-20_EXCAM_noise_characterization', 'nonlin')
        nonlin_list = [os.path.join(nonlin_dir, f) for f in os.listdir(nonlin_dir) if f.lower().endswith('.fits')]
        nonlin_dataset = data.Dataset(nonlin_list)
        sets, unique_vals = nonlin_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        for Set in sets:
            for file in Set:
                if file.ext_hdr['DATETIME'] in datetime_list:
                    os.remove(file.filepath)
                else:
                    datetime_list.append(file.ext_hdr['DATETIME'])
        #redo now that we've removed files with duplicate DATETIME values
        # keep only sets that are at least 5 files long
        # nonlin_list = [os.path.join(nonlin_dir, f) for f in os.listdir(nonlin_dir) if f.lower().endswith('.fits')]
        # nonlin_dataset = data.Dataset(nonlin_list)
        # sets, unique_vals = nonlin_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        # for Set in sets:
        #     if len(Set) < 5:
        #         for file in Set:
        #             os.remove(file.filepath)

        #### trying to pare down number of files some 
        sets, unique_vals = kgain_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        if not os.path.exists(kgain_dir+'/extra_kgain/'):
                        os.makedirs(kgain_dir+'/extra_kgain/')
        for i in range(len(sets)):
            if i in np.append(np.arange(13, 22),np.arange(23,43)):
                for file in sets[i]:
                    dest_path = os.path.join(kgain_dir, 'extra_kgain', os.path.basename(file.filepath))
                    os.rename(file.filepath, dest_path)

    if False:
        kgain_list = []
        nonlin_list = []
        datetime_list = []
        # getting decent frames for kgian/nonlin, comparable to what was there with original TVAC files (before SSC's updated headers)
        for ssc_root, ssc_dirs, ssc_files in os.walk(ssc_dir):
            # if len(kgain_list) > 50 and len(nonlin_list) > 50:
            #     break
            if not any(ssc_root.endswith(suffix) for suffix in ['27', '31', '32']):
                continue
            for ssc_file in ssc_files:
                # if len(kgain_list) > 50 and len(nonlin_list) > 50:
                #     break
                if not ssc_file.lower().endswith('.fits'):
                    continue
                with fits.open(os.path.join(ssc_root, ssc_file)) as hdul:
                    if 'CFAMNAME' not in hdul[1].header:
                        continue
                    # looks like EMGAIN_A is not implemented yet and is simply 1 for all files
                    if (hdul[1].header['EMGAIN_C'] == 1) and hdul[1].header['CFAMNAME'] == 'CLEAR':
                        file_path = file_path = os.path.join(ssc_root, ssc_file) 
                        kgain_list.append(file_path)
                        hdul[1].header['EMGAIN_A'] = -1
                        if hdul[1].header['DATETIME'] in datetime_list:
                            continue
                        datetime_list.append(hdul[1].header['DATETIME'])
                        file_path = '/Users/kevinludwick/Documents/ssc_tvac_test/TV-20_EXCAM_noise_characterization/nonlin/kgain/'+ssc_file.lower().split('.')[0]+'.fits'
                        hdul.writeto(file_path, overwrite=True)
                    elif (hdul[1].header['EMGAIN_C'] > 1) and hdul[1].header['CFAMNAME'] == 'CLEAR':
                        file_path = file_path = os.path.join(ssc_root, ssc_file) 
                        nonlin_list.append(file_path)
                        hdul[1].header['EMGAIN_A'] = -1
                        if hdul[1].header['DATETIME'] in datetime_list:
                            continue
                        datetime_list.append(hdul[1].header['DATETIME'])
                        file_path = '/Users/kevinludwick/Documents/ssc_tvac_test/TV-20_EXCAM_noise_characterization/nonlin/'+ssc_file.lower().split('.')[0]+'.fits'
                        hdul.writeto(file_path, overwrite=True)
            

        nonlin_dataset = data.Dataset(nonlin_list)
        sets, unique_vals = nonlin_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        for i in range(len(sets)):
            print('index: ', i, 'length of set: ',len(sets[i]), 'unique values for set: ',unique_vals[i] )

        kgain_dataset = data.Dataset(kgain_list)
        sets, unique_vals = kgain_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
        for i in range(len(sets)):
            print('index: ', i, 'length of set: ',len(sets[i]), 'unique values for set: ',unique_vals[i] )
    ############################################################################
    if False:
        dark_list = []
        # getting decent frames for darkmap, comparable to what was there with original TVAC files (before SSC's updated headers)
        for ssc_root, ssc_dirs, ssc_files in os.walk(ssc_dir):
            # if len(dark_list) > 50:
            #     break
            if not any(ssc_root.endswith(suffix) for suffix in ['27', '31', '32']):
                continue
            for ssc_file in ssc_files:
                if not ssc_file.lower().endswith('.fits'):
                    continue
                with fits.open(os.path.join(ssc_root, ssc_file)) as hdul:
                    if 'CFAMNAME' not in hdul[1].header:
                        continue
                    if hdul[1].header['CFAMNAME'] == 'DARK':
                        file_path = file_path = os.path.join(ssc_root, ssc_file) 
                        dark_list.append(file_path)
                        if hdul[1].header['EXPTIME'] == 100 and hdul[1].header['EMGAIN_C'] == 1.34 and hdul[1].header['KGAINPAR'] == 6.0:
                            file_path = '/Users/kevinludwick/Documents/ssc_tvac_test/TV-20_EXCAM_noise_characterization/darkmap/'+ssc_file.upper().split('.')[0]+'.fits'
                            hdul.writeto(file_path, overwrite=True)

        dark_dataset = data.Dataset(dark_list)
        sets, unique_vals = dark_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
   
    ############################################################################
    # get DARK frames for synthesized noise maps e2e test
    if True:
        dark_list = []
        # getting decent frames for darkmap, comparable to what was there with original TVAC files (before SSC's updated headers)
        ssc_root = '/Users/kevinludwick/Library/CloudStorage/Box-Box/CGI_TVAC_Data/ssc_331/0089001001001001027'
        for ssc_file in os.listdir(ssc_root):
            if not ssc_file.lower().endswith('.fits'):
                continue
            with fits.open(os.path.join(ssc_root, ssc_file)) as hdul:
                if 'CFAMNAME' not in hdul[1].header:
                    continue
                if hdul[1].header['CFAMNAME'] == 'DARK':
                    file_path = file_path = os.path.join(ssc_root, ssc_file) 
                    dark_list.append(file_path)
                    file_path = '/Users/kevinludwick/Documents/ssc_tvac_test/E2E_Test_Data2/TV-20_EXCAM_noise_characterization/noisemap_test_data/'+ssc_file
                    hdul.writeto(file_path, overwrite=True)

        dark_dataset = data.Dataset(dark_list)
        sets, unique_vals = dark_dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])

    ############################################################################
    # manually getting a string of 30 frames for mean frame
    if False:
        counter = 0
        mnframe_path = os.path.join(ssc_dir, '0089001001001001027')
        for filename in os.listdir(mnframe_path):
            if not filename.lower().endswith('.fits'):
                continue
            filepath = os.path.join(mnframe_path, filename)
            with fits.open(filepath) as ssc_hdul:
                exptime = ssc_hdul[1].header.get('EXPTIME')
                cmdgain = ssc_hdul[1].header.get('EMGAIN_C')
                dpamname = ssc_hdul[1].header.get('DPAMNAME')
                if exptime == 5 and cmdgain == 1 and dpamname == 'PUPIL,PUPIL_FFT':
                    counter += 1
                    # Write the SSC file to the new location
                    ssc_hdul_copy = fits.HDUList([hdu.copy() for hdu in ssc_hdul])
                    ssc_hdul_copy.writeto(os.path.join(output_base_dir, 'TV-20_EXCAM_noise_characterization', 'nonlin', 'kgain', filename), overwrite=True)
                    if counter > 30:
                        break
        

    ############################################################################
    # Recursively find all .fits files in tvac_dir and its subdirectories and try to match 1-to-1 with updated SSC files
    # based on the 'DATETIME' value in the extension header.
    # If a match is found, copy the SSC file to a new location with the same relative path as the original TVAC file.
    # If no match is found, log the unmatched.
    unmatched_files = []
    match_dict = {}
    for root, dirs, files in os.walk(tvac_dir):
        if 'TV-20_' not in root and 'TV-36_' not in root:
            continue
        if 'TV-20_' in root and not any(name in root for name in ['nonlin', 'kgain', 'darkmap', 'noisemap_test_data']): 
            continue
        if 'TV-36_' in root and not any(name in root for name in ['L1']):
            continue
        for file in files:
            if file.lower().endswith('.fits'):
                if not any(char.isdigit() for char in file):
                    continue
                if not re.search(r'\d{5,}', file):
                    continue
                # Open the FITS file and get the 'DATETIME' value from the extension header
                with fits.open(os.path.join(root, file)) as hdul:
                    if 'DATETIME' not in hdul[1].header:
                        continue
                    datetime_val = hdul[1].header.get('DATETIME')
                    # obsname_val = hdul[1].header.get('OBSNAME')
                    # pri_hdr = hdul[0].header
                    # ext_hdr = hdul[1].header

                # Search for a .fits file in ssc_dir with a matching 'DATETIME' value
                matched = False
                for ssc_root, ssc_dirs, ssc_files in os.walk(ssc_dir):
                    if 'TV-20_' in root:
                            if not any(ssc_root.endswith(suffix) for suffix in ['27', '31', '32']):
                                continue 
                    if 'TV-36_' in root:
                        if not any(ssc_root.endswith(suffix) for suffix in ['80', '88', '109']):
                            continue
                    for ssc_file in ssc_files:
                        if ssc_file.lower().endswith('.csv'):
                            csv_filepath = os.path.join(ssc_root, ssc_file)
                            # Read the CSV file as a numpy array
                            #csv_data = np.genfromtxt(csv_filepath, delimiter=',', names=True, dtype=None, encoding='utf-8')
                            # Check if any value in the DATETIME column of the .csv file matches datetime_val
                            with open(csv_filepath, 'r', newline='') as csvfile_check:
                                reader = csv.DictReader(csvfile_check)
                                for row in reader:
                                    if 'DATETIME' in row and row['DATETIME'] == datetime_val:
                                        matched = True
                                        year=row['DATETIME'][:4]
                                        month=row['DATETIME'][5:7]
                                        day=row['DATETIME'][8:10]
                                        hour= row['DATETIME'][11:13]
                                        minute= row['DATETIME'][14:16]
                                        seconds= row['DATETIME'][17:19]
                                        tenth = int(np.round(row['DATETIME'][19:20], 1))
                                        first_part = ssc_root.split(os.sep)[-1] # get the last part of the path
                                        filename = 'cgi_{0}_{1}{2}{3}T{4}{5}{6}{7}_l1_.fits'.format(first_part, year, month, day, hour, minute, seconds, tenth)
                                        match_dict[os.path.join(root, file)] = filename
                                        # Determine relative path from tvac_dir
                                        rel_path = os.path.relpath(os.path.join(root, file), tvac_dir)
                                        # Construct new output path in a new directory
                                        output_path = os.path.join(output_base_dir, rel_path)
                                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                        # Write the SSC file to the new location
                                        with fits.open(os.path.join(ssc_root, ssc_file)) as ssc_hdul:
                                            ssc_hdul_copy = fits.HDUList([hdu.copy() for hdu in ssc_hdul])
                                            ssc_hdul_copy.writeto(os.path.join(os.path.split(output_path)[0], filename), overwrite=True)
                                        break
                        if matched:
                            break
                    if matched:
                        break

                    #     if ssc_file.lower().endswith('.fits'):
                    #         if not any(char.isdigit() for char in ssc_file):
                    #             continue
                    #         if not re.search(r'\d{5,}', ssc_file):
                    #             continue
                    #         with fits.open(os.path.join(ssc_root, ssc_file)) as ssc_hdul:
                    #             if 'DATETIME' not in ssc_hdul[1].header:
                    #                 continue
                    #             ssc_datetime = ssc_hdul[1].header.get('DATETIME')
                    #             if ssc_datetime == datetime_val:
                    #                 matched = True
                    #                 match_dict[file] = ssc_file
                    #                 # ssc_pri_hdr = ssc_hdul[0].header
                    #                 # ssc_ext_hdr = ssc_hdul[1].header
                    #                 # ssc_ext_hdr['OBSNAME'] = obsname_val
                    #                 # Determine relative path from tvac_dir
                    #                 rel_path = os.path.relpath(os.path.join(root, file), tvac_dir)
                    #                 # Construct new output path in a new directory
                    #                 output_path = os.path.join(output_base_dir, rel_path)
                    #                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    #                 # Write the SSC file to the new location
                    #                 ssc_hdul_copy = fits.HDUList([hdu.copy() for hdu in ssc_hdul])
                    #                 ssc_hdul_copy.writeto(os.path.join(os.path.split(output_path)[0], ssc_file.split('.')[0].lower()+'.fits'), overwrite=True)
                    #                 break
                    #     if matched:
                    #         break
                    # if matched:
                    #     break

                if not matched:
                    unmatched_files.append(os.path.join(root, file))
    # Print unmatched files
    if unmatched_files:
        print("Unmatched files:")
        for file in unmatched_files:
            print(file)
        # Save unmatched files to a CSV file
        with open(os.path.join(output_base_dir,'noisemap_unmatched_files.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Unmatched File'])
            for file in unmatched_files:
                with fits.open(file) as hdul:
                    datetime_val = hdul[1].header.get('DATETIME', '')
                writer.writerow([file, datetime_val])
    else:
        print("All files matched successfully.")
    # Print matched files
    if match_dict:
        print("Matched files:")
        for tvac_file, ssc_file in match_dict.items():
            print(f"{tvac_file} -> {ssc_file}")
            with open(os.path.join(output_base_dir, 'noisemap_matched_files.csv'), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # with fits.open(os.path.join(ssc_dir, ssc_file)) as ssc_hdul:
                #     ssc_datetime = ssc_hdul[1].header.get('DATETIME', '')
                writer.writerow([tvac_file, ssc_file])
    else:
        print("No files matched.")


import os
import json
import astropy.time as time
import warnings
import corgidrp
import corgidrp.astrom
import corgidrp.bad_pixel_calibration
import corgidrp.calibrate_kgain
import corgidrp.combine
import corgidrp.data as data
import corgidrp.caldb as caldb
import corgidrp.l1_to_l2a
import corgidrp.l2a_to_l2b
import corgidrp.l2b_to_l3
import corgidrp.l3_to_l4
import corgidrp.nd_filter_calibration
import corgidrp.photon_counting
import corgidrp.pump_trap_calibration
import corgidrp.calibrate_nonlin
import corgidrp.detector
import corgidrp.flat
import corgidrp.darks
import corgidrp.sorting
import corgidrp.fluxcal
import corgidrp.spec

all_steps = {
    "prescan_biassub" : corgidrp.l1_to_l2a.prescan_biassub,
    "detect_cosmic_rays" : corgidrp.l1_to_l2a.detect_cosmic_rays,
    "calibrate_nonlin": corgidrp.calibrate_nonlin.calibrate_nonlin,
    "correct_nonlinearity" : corgidrp.l1_to_l2a.correct_nonlinearity,
    "update_to_l2a" : corgidrp.l1_to_l2a.update_to_l2a,
    "add_shot_noise_to_err" : corgidrp.l2a_to_l2b.add_shot_noise_to_err,
    "dark_subtraction" : corgidrp.l2a_to_l2b.dark_subtraction,
    "flat_division" : corgidrp.l2a_to_l2b.flat_division,
    "frame_select" : corgidrp.l2a_to_l2b.frame_select,
    "convert_to_electrons" : corgidrp.l2a_to_l2b.convert_to_electrons,
    "em_gain_division" : corgidrp.l2a_to_l2b.em_gain_division,
    "cti_correction" : corgidrp.l2a_to_l2b.cti_correction,
    "correct_bad_pixels" : corgidrp.l2a_to_l2b.correct_bad_pixels,
    "desmear" : corgidrp.l2a_to_l2b.desmear,
    "update_to_l2b" : corgidrp.l2a_to_l2b.update_to_l2b,
    "boresight_calibration": corgidrp.astrom.boresight_calibration,
    "calibrate_trap_pump": corgidrp.pump_trap_calibration.tpump_analysis,
    "create_bad_pixel_map" : corgidrp.bad_pixel_calibration.create_bad_pixel_map,
    "calibrate_kgain" : corgidrp.calibrate_kgain.calibrate_kgain,
    "calibrate_darks" : corgidrp.darks.calibrate_darks_lsq,
    "create_onsky_flatfield" : corgidrp.flat.create_onsky_flatfield,
    "combine_subexposures" : corgidrp.combine.combine_subexposures,
    "build_trad_dark" : corgidrp.darks.build_trad_dark,
    "sort_pupilimg_frames" : corgidrp.sorting.sort_pupilimg_frames,
    "get_pc_mean" : corgidrp.photon_counting.get_pc_mean,
    "divide_by_exptime" : corgidrp.l2b_to_l3.divide_by_exptime,
    "crop" : corgidrp.l2b_to_l3.crop,
    "northup" : corgidrp.l3_to_l4.northup,
    "calibrate_fluxcal_aper": corgidrp.fluxcal.calibrate_fluxcal_aper,
    "calibrate_pol_fluxcal_aper": corgidrp.fluxcal.calibrate_pol_fluxcal_aper,
    "update_to_l3": corgidrp.l2b_to_l3.update_to_l3,
    "create_wcs": corgidrp.l2b_to_l3.create_wcs,
    "replace_bad_pixels": corgidrp.l3_to_l4.replace_bad_pixels,
    "distortion_correction": corgidrp.l3_to_l4.distortion_correction,
    "find_star": corgidrp.l3_to_l4.find_star,
    "do_psf_subtraction": corgidrp.l3_to_l4.do_psf_subtraction,
    "determine_wave_zeropoint": corgidrp.l3_to_l4.determine_wave_zeropoint,
    "add_wavelength_map": corgidrp.l3_to_l4.add_wavelength_map,
    "update_to_l4": corgidrp.l3_to_l4.update_to_l4,
    "generate_ct_cal": corgidrp.corethroughput.generate_ct_cal,
    "create_ct_map": corgidrp.corethroughput.create_ct_map,
    "create_nd_filter_cal": corgidrp.nd_filter_calibration.create_nd_filter_cal,
    "compute_psf_centroid": corgidrp.spec.compute_psf_centroid,
    "calibrate_dispersion_model": corgidrp.spec.calibrate_dispersion_model,
    "fit_line_spread_function": corgidrp.spec.fit_line_spread_function,
}

recipe_dir = os.path.join(os.path.dirname(__file__), "recipe_templates")

def walk_corgidrp(filelist, CPGS_XML_filepath, outputdir, template=None):
    """
    Automatically create a recipe and process the input filelist.
    Does both the `autogen_recipe` and `run_recipe` steps.

    Args:
        filelist (list of str): list of filepaths to files
        CPGS_XML_filepath (str): path to CPGS XML file for this set of files in filelist
        outputdir (str): output directory folderpath
        template (str or json): custom template. It can be one of three things
                                  * the full json object, 
                                  * a filename of a template that's already in the recipe_templates folder
                                  * a filepath to a template on disk somewhere
                                

    Returns:
        json or list: the JSON recipe (or list of JSON recipes) that was used for processing
    """
    if isinstance(template, str):
        if os.path.sep not in template:
            # this is just a template name in the recipe_templates folder
            recipe_filepath = os.path.join(recipe_dir, template)
        else:
            recipe_filepath = template
        
        template = json.load(open(recipe_filepath, 'r'))

    # generate recipe
    recipes = autogen_recipe(filelist, outputdir, template=template)


    if not isinstance(recipes, list):
        recipes = [recipes]
    
    # process recipes
    output_filelist = None
    for i, recipe in enumerate(recipes):
        # check for recipe chaining
        if i > 0 and  len(recipe['inputs']) == 0:
            for filename in output_filelist:
                recipe["inputs"].append(filename)

        output_filelist = run_recipe(recipe)

    # return just the recipe if there was only one
    if len(recipes) == 1:
        return recipes[0]
    else:
        return recipes

def autogen_recipe(filelist, outputdir, template=None):
    """
    Automatically creates a recipe (or recipes) by identifyng and populating a template.
    Returns a single recipe unless there are multiple recipes that should be produced.

    Args:
        filelist (list of str): list of filepaths to files
        outputdir (str): output directory folderpath
        template (json): enables passing in of custom template, if desired

    Returns:
        json list: the JSON recipe (or list of recipes) that the input filelist will be processed with
    """
    # Handle the case where filelist is empty
    if not filelist:
        print("Input filelist is empty, using default handling to create recipe.")
        first_frame = None
    else:
        # load the data to check what kind of recipe it is
        dataset = data.Dataset(filelist)
        first_frame = dataset[0]

    # if user didn't pass in template
    if template is None:
        recipe_filename, chained = guess_template(dataset)

        # handle it as a list moving forward
        if isinstance(recipe_filename, list):
            recipe_filename_list = recipe_filename
        else:
            recipe_filename_list = [recipe_filename]

        recipe_template_list = []
        for recipe_filename in recipe_filename_list:
            # load the template recipe
            recipe_filepath = os.path.join(recipe_dir, recipe_filename)
            template = json.load(open(recipe_filepath, 'r'))
            recipe_template_list.append(template)
    else:
        # user passed in a single template
        recipe_template_list = [template]
        chained = False

    recipe_list = []
    for i, template in enumerate(recipe_template_list):
        # create the personalized recipe
        recipe = template.copy()
        recipe["template"] = False

        # for chained recipes, don't put the input in yet since we don't know it
        if i > 0 and chained:
            pass
        else:
            for filename in filelist:
                recipe["inputs"].append(filename)


        recipe["outputdir"] = outputdir

        ## Populate default values
        ## This includes calibration files that need to be automatically determined
        ## This also includes the dark subtraction outputdir for synthetic darks
        this_caldb = caldb.CalDB()
        for step in recipe["steps"]:
            # by default, identify all the calibration files needed, unless jit setting is turned on
            # two cases where we should be identifying the calibration recipes now
            if "jit_calib_id" in recipe['drpconfig'] and (not recipe['drpconfig']["jit_calib_id"]):
                _fill_in_calib_files(step, this_caldb, first_frame)
            elif ("jit_calib_id" not in recipe['drpconfig']) and (not corgidrp.jit_calib_id):
                _fill_in_calib_files(step, this_caldb, first_frame)

            if step["name"].lower() == "dark_subtraction":
                if step["keywords"]["outputdir"].upper() == "AUTOMATIC":
                    step["keywords"]["outputdir"] = recipe["outputdir"]

        recipe_list.append(recipe)
    
    # if only a single recipe, return the recipe. otherwise return list
    if len(recipe_list) > 1:
        return recipe_list
    else:
        return recipe_list[0]

def _fill_in_calib_files(step, this_caldb, ref_frame):
    """
    Fills in calibration files defined as "AUTOMATIC" in a recipe
    
    By default, throws an error if there are no available cal files of a certian type.
    Exceptional case is when the pipeline setting `skip_missing_cal_steps = True` is set:
    in this case, it will mark this step to be skipped, but continue processing the recipe.

    Args:
        step (dict): the portion of a recipe for this step
        this_caldb (corgidrp.CalDB): calibration database conection
        ref_frame (corgidrp.Image): a reference frame to use to determine the optimal calibration
    
    Returns:
        dict: the step, but with calibration files filled in
    """
    if "calibs" not in step:
        return step # don't have to do anything if no calibrations
    
    for calib in step["calibs"]:
        # order matters, so only one calibration file per dictionary

        if "AUTOMATIC" in step["calibs"][calib].upper():
            calib_dtype = data.datatypes[calib]

            # try to look up the best calibration, but it could raise an error
            try:
                best_cal_file = this_caldb.get_calib(ref_frame, calib_dtype)
                best_cal_filepath = best_cal_file.filepath
            except ValueError as e:
                if "OPTIONAL" in step["calibs"][calib].upper():
                    # couldn't find a good cal but this one is optional, so we are going to put nothing in there
                    # this means the step function can run without this calibration file
                    best_cal_filepath = None
                elif corgidrp.skip_missing_cal_steps:
                    step["skip"] = True # skip this step but continue
                    step["calibs"][calib] = None
                    warnings.warn("Skipping {0} because no {1} in caldb and skip_missing_cal_steps is True".format(step['name'], calib))
                    continue # continue on the for loop
                else:
                    raise # reraise exception

            # set calibration file to this one
            step["calibs"][calib] = best_cal_filepath

    return step

def guess_template(dataset):
    """
    Guesses what template should be used to process a specific image

    Args:
        dataset (corgidrp.data.Dataset): a Dataset to process

    Returns:
        str or list: the best template filename or a list of multiple template filenames
        bool: whether multiple recipes are chained together. If True, the output of the first recipe
              should be used as the input to the second recipe. If False, the same input should be used
              for all recipes. This keyworkd is irrelevant if only a single recipe is returned.
    """
    image = dataset[0] # first image for convenience

    chained = False # whether multiiple recipes are chained together
    # L1 -> L2a data processing
    if image.ext_hdr['DATALVL'] == "L1":
        if 'VISTYPE' not in image.pri_hdr:
            # this is probably IIT test data. Do generic processing
            recipe_filename = "l1_to_l2b.json"
        elif image.pri_hdr['VISTYPE'][:3] == "ENG":
            # first three letters are ENG
            # for either ENGPUPIL or ENGIMGAGE
            recipe_filename = "l1_to_l2a_eng.json"
        elif image.pri_hdr['VISTYPE'] == "BORESITE":
            recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", 'l2b_to_boresight.json'] #"l1_to_boresight.json"
            chained = True
        elif image.pri_hdr['VISTYPE'] == "FFIELD":
            recipe_filename = "l1_flat_and_bp.json"
        elif image.pri_hdr['VISTYPE'] == "DARK":
            _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
            if image.ext_hdr['ISPC']:
                recipe_filename = "l1_to_l2b_pc_dark.json"
            elif len(unique_vals) > 1: # darks for noisemap creation
                recipe_filename = "l1_to_l2a_noisemap.json"
            else: # then len(unique_vals) is 1 and not PC: traditional darks
                recipe_filename = "build_trad_dark_image.json"
        elif image.pri_hdr['VISTYPE'] == "CGIVST_CAL_PUPIL_IMAGING":
            recipe_filename = ["l1_to_l2a_nonlin.json", "l1_to_kgain.json"]
        elif image.pri_hdr['VISTYPE'] in ("ABSFLXFT", "ABSFLXBT"):
            _, fsm_unique = dataset.split_dataset(exthdr_keywords=['FSMX', 'FSMY'])
            if len(fsm_unique) > 1:
                recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", "l2b_to_nd_filter.json"]
                chained = True
            else:
                recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", "l2b_to_fluxcal_factor.json"]
                chained = True
        elif image.pri_hdr['VISTYPE'] == 'CORETPUT':
            recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", 'l2b_to_corethroughput.json']
            chained = True
        else:
            recipe_filename = "l1_to_l2a_basic.json"  # science data and all else (including photon counting)
    # L2a -> L2b data processing
    elif image.ext_hdr['DATALVL'] == "L2a":
        if image.pri_hdr['VISTYPE'] == "DARK":
            _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
            if image.ext_hdr['ISPC']:
                recipe_filename = "l2a_to_l2b_pc_dark.json"
            elif len(unique_vals) > 1: # darks for noisemap creation
                recipe_filename = "l2a_to_l2a_noisemap.json"
            else: # then len(unique_vals) is 1 and not PC: traditional darks
                recipe_filename = "l2a_build_trad_dark_image.json"
        else:
            if image.ext_hdr['ISPC']:
                recipe_filename = "l2a_to_l2b_pc.json"
            else:
                recipe_filename = "l2a_to_l2b.json"  # science data and all else
    # L2b -> L3 data processing
    elif image.ext_hdr['DATALVL'] == "L2b":
        if image.pri_hdr['VISTYPE'] in ("ABSFLXFT", "ABSFLXBT"):
            _, fsm_unique = dataset.split_dataset(exthdr_keywords=['FSMX', 'FSMY'])
            if len(fsm_unique) > 1:
                recipe_filename = "l2b_to_nd_filter.json"
            else:
                if image.ext_hdr['DPAMNAME'] == 'POL0' or image.ext_hdr['DPAMNAME'] == 'POL45':
                    recipe_filename = 'l2b_to_fluxcal_factor_pol.json'
                else:
                    recipe_filename = "l2b_to_fluxcal_factor.json"
        elif image.pri_hdr['VISTYPE'] == 'CORETPUT':
            recipe_filename = 'l2b_to_corethroughput.json'
        else:
            recipe_filename = "l2b_to_l3.json"
    # L3 -> L4 data processing
    elif image.ext_hdr['DATALVL'] == "L3":
        if image.ext_hdr['FSMLOS'] == 1:
            # coronagraphic obs - PSF subtraction
            recipe_filename = "l3_to_l4.json"
        else:
            # noncorongrpahic obs - no PSF subtraction
            recipe_filename = "l3_to_l4_nopsfsub.json"
            
    else:
        raise NotImplementedError("Cannot automatically guess the input dataset with 'DATALVL' = {0}".format(image.ext_hdr['DATALVL']))

    return recipe_filename, chained


def save_data(dataset_or_image, outputdir, suffix=""):
    """
    Saves the dataset or image that has currently been outputted by the last step function.
    Records calibration frames into the caldb during the process

    Args:
        dataset_or_image (corgidrp.data.Dataset or corgidrp.data.Image): data to save
        outputdir (str): path to directory where files should be saved
        suffix (str): optional suffix to tack onto the filename. 
                      E.g.: `test.fits` with `suffix="dark"` becomes `test_dark.fits`
    """
    # convert everything to dataset to make life easier
    if isinstance(dataset_or_image, data.Image):
        dataset = data.Dataset([dataset_or_image])
    else:
        dataset = dataset_or_image

    # add suffix to ending if necessary
    if len(suffix) > 0:
        filenames = []

        suffix = suffix.strip("_") # user doesn't need to pass underscores
        for image in dataset:
            # grab everything before .FITS
            fits_index = image.filename.lower().rfind(".fits")
            filename_base = image.filename[:fits_index]
            new_filename = "{0}_{1}.fits".format(filename_base, suffix)
            filenames.append(new_filename)
    else:
        filenames = None

    # save!
    dataset.save(filedir=outputdir, filenames=filenames)

    # add calibration data to caldb as necessary
    for image in dataset:
        if type(image) in caldb.labels:
            # this is a calibration frame!
            this_caldb = caldb.CalDB()
            this_caldb.create_entry(image)



def run_recipe(recipe, save_recipe_file=True):
    """
    Run the specified recipe

    Args:
        recipe (dict or str): either the filepath to the recipe or the already loaded in recipe
        save_recipe_file (bool): saves the recipe as a JSON file in the outputdir (true by default)

    Returns:
        list: list of filepaths to the saved files, or None if no files were saved
    """
    if isinstance(recipe, str):
        # need to load in
        recipe = json.load(open(recipe, "r"))

    # configure pipeline as needed
    for setting in recipe['drpconfig']:
        # equivalent to corgidrp.setting = recipe['drpconfig'][setting]
        setattr(corgidrp, setting, recipe['drpconfig'][setting])

    # read in data, if not doing bp map
    if recipe["inputs"]:
        filelist = recipe["inputs"]
        curr_dataset = data.Dataset(filelist)
        # write the recipe into the image extension header
        for frame in curr_dataset:
            frame.ext_hdr["RECIPE"] = json.dumps(recipe)
    else:
        curr_dataset = []

    # save recipe before running recipe
    if save_recipe_file:
        recipe_filename = "{0}_{1}_recipe.json".format(recipe["name"], time.Time.now().isot)
        recipe_filename = recipe_filename.replace(":", ".")  # replace colons with periods for compatibility with Windows machines
        recipe_filepath = os.path.join(recipe["outputdir"], recipe_filename)
        with open(recipe_filepath, "w") as json_file:
            json.dump(recipe, json_file, indent=4)

    tot_steps = len(recipe["steps"])
    save_step = False

    # execute each pipeline step
    for i, step in enumerate(recipe["steps"]):
        print("Walker step {0}/{1}: {2}".format(i+1, tot_steps, step["name"]))
        if step["name"].lower() == "save":
            # special save instruction
            
            # see if suffix is specified as a keyword
            if "keywords" in step and "suffix" in step["keywords"]:
                suffix =  step["keywords"]["suffix"]
            else:
                suffix = ''
                
            save_data(curr_dataset, recipe["outputdir"], suffix=suffix)
            save_step = True

        else:
            step_func = all_steps[step["name"]]

            # edge case if this step has been specified to be skipped
            if "skip" in step and step["skip"]:
                continue

            other_args = ()
            if "calibs" in step:
                # if JIT calibration resolving is toggled, figure out the calibrations here
                # by default, this is false
                if (corgidrp.jit_calib_id and ("jit_calib_id" not in recipe['drpconfig'])) or (("jit_calib_id" in recipe['drpconfig']) and recipe['drpconfig']["jit_calib_id"]) :
                    this_caldb = caldb.CalDB()
                    # dataset may have turned into a single image. handle this case. 
                    if isinstance(curr_dataset, data.Dataset):
                        ref_image = curr_dataset[0]
                        list_of_frames = curr_dataset
                    else:
                        ref_image = curr_dataset
                        list_of_frames = [curr_dataset]
                    _fill_in_calib_files(step, this_caldb, ref_image)

                    # also update the recipe we used in the headers
                    for frame in list_of_frames:
                        frame.ext_hdr["RECIPE"] = json.dumps(recipe)


                # load the calibration files in from disk
                for calib in step["calibs"]:
                    calib_dtype = data.datatypes[calib]
                    if step["calibs"][calib] is not None:
                        cal_file = calib_dtype(step["calibs"][calib])
                    else:
                        cal_file = None
                    other_args += (cal_file,)


            if "keywords" in step:
                kwargs = step["keywords"]
            else:
                kwargs = {}

            # run the step!
            curr_dataset = step_func(curr_dataset, *other_args, **kwargs)

    output_filepaths = None
    if save_step:
        if isinstance(curr_dataset, data.Dataset):
            output_filepaths = [frame.filepath for frame in curr_dataset]
        else:
            output_filepaths = [curr_dataset.filepath]
    
    return output_filepaths


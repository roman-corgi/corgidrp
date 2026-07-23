import os
import re
import json
import astropy.time as time
import warnings
import xml.etree.ElementTree as ET
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
    "discard_setup_frames" : corgidrp.l1_to_l2a.discard_setup_frames,
    "detect_cosmic_rays" : corgidrp.l1_to_l2a.detect_cosmic_rays,
    "calibrate_nonlin": corgidrp.calibrate_nonlin.calibrate_nonlin,
    "correct_nonlinearity" : corgidrp.l1_to_l2a.correct_nonlinearity,
    "update_to_l2a" : corgidrp.l1_to_l2a.update_to_l2a,
    "update_to_intermediate" : corgidrp.l1_to_l2a.update_to_intermediate,
    "add_shot_noise_to_err" : corgidrp.l2a_to_l2b.add_shot_noise_to_err,
    "dark_subtraction" : corgidrp.l2a_to_l2b.dark_subtraction,
    "flat_division" : corgidrp.l2a_to_l2b.flat_division,
    "flat_division_pol" : corgidrp.l2a_to_l2b.flat_division_pol,
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
    "create_onsky_pol_flatfield" : corgidrp.flat.create_onsky_pol_flatfield,
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
    "find_spec_star" : corgidrp.l3_to_l4.find_spec_star,
    "combine_frames_per_visit": corgidrp.combine.combine_frames_per_visit,
    "do_psf_subtraction": corgidrp.l3_to_l4.do_psf_subtraction,
    "spec_psf_subtraction": corgidrp.l3_to_l4.spec_psf_subtraction,
    "determine_wave_zeropoint": corgidrp.l3_to_l4.determine_wave_zeropoint,
    "add_wavelength_map": corgidrp.l3_to_l4.add_wavelength_map,
    "extract_spec": corgidrp.l3_to_l4.extract_spec,
    "update_to_l4": corgidrp.l3_to_l4.update_to_l4,
    "update_to_l4_pol": corgidrp.l3_to_l4.update_to_l4_pol,
    "generate_ct_cal": corgidrp.corethroughput.generate_ct_cal,
    "create_ct_map": corgidrp.corethroughput.create_ct_map,
    "create_nd_filter_cal": corgidrp.nd_filter_calibration.create_nd_filter_cal,
    "create_nd_filter_cal_spec": corgidrp.nd_filter_calibration.create_nd_filter_cal_spec,
    "compute_psf_centroid": corgidrp.spec.compute_psf_centroid,
    "calibrate_dispersion_model": corgidrp.spec.calibrate_dispersion_model,
    "fit_line_spread_function": corgidrp.spec.fit_line_spread_function,
    "split_image_by_polarization_state": corgidrp.l2b_to_l3.split_image_by_polarization_state,
    "calc_stokes_unocculted": corgidrp.pol.calc_stokes_unocculted,
    "generate_mueller_matrix_cal": corgidrp.pol.generate_mueller_matrix_cal,
    "align_polarimetry_frames": corgidrp.l3_to_l4.align_polarimetry_frames,
    "combine_polarization_states": corgidrp.l3_to_l4.combine_polarization_states,
    "subtract_stellar_polarization": corgidrp.l3_to_l4.subtract_stellar_polarization,
    "align_2d_frames": corgidrp.l3_to_l4.align_2d_frames,
    "combine_spec": corgidrp.l3_to_l4.combine_spec,
    "spec_fluxcal": corgidrp.spec.spec_fluxcal
}

recipe_dir = os.path.join(os.path.dirname(__file__), "recipe_templates")

def _load_recipe_template(recipe_filename, user_templates_dir=None):
    """
    Load recipe template, checking user_templates first, then defaults.

    Args:
        recipe_filename (str): Template filename (e.g., "l1_to_l2a_basic.json")
        user_templates_dir (str): Path to user templates directory. If None, uses corgidrp.user_templates_dir

    Returns:
        tuple: (template dict, source path, is_user_template bool)
    """
    if user_templates_dir is None:
        user_templates_dir = corgidrp.user_templates_dir

    # First check user templates directory
    user_template_path = os.path.join(user_templates_dir, recipe_filename)
    if os.path.exists(user_template_path):
        with open(user_template_path, 'r') as f:
            template = json.load(f)
        return template, user_template_path, True

    # Fall back to default template
    default_template_path = os.path.join(recipe_dir, recipe_filename)
    if os.path.exists(default_template_path):
        with open(default_template_path, 'r') as f:
            template = json.load(f)
        return template, default_template_path, False

    # Template not found in either location
    raise FileNotFoundError(
        f"Recipe template '{recipe_filename}' not found in user templates "
        f"({user_templates_dir}) or default templates ({recipe_dir})"
    )

def _validate_template_structure(user_template, default_template_path, recipe_filename):
    """
    Validate that user template has same step names/order as default.
    Raises ValueError if mismatch found when enforce_template_structure=True.

    Args:
        user_template (dict): User-provided template
        default_template_path (str): Path of the default template from repo
        recipe_filename (str): Template name for error messages

    Raises:
        ValueError: If template structures don't match and enforcement is enabled
    """
    if not corgidrp.enforce_template_structure:
        return  # Validation disabled

    if not os.path.exists(default_template_path):
        raise FileNotFoundError(
            f"Cannot validate user template '{recipe_filename}': "
            f"no corresponding default template found at {default_template_path}. "
            f"User templates must have same filename as a default template when "
            f"enforce_template_structure=True."
        )
    with open(default_template_path, 'r') as f:
        default_template = json.load(f)

        # Extract step names from both templates
        user_steps = [step['name'] for step in user_template.get('steps', [])]
        default_steps = [step['name'] for step in default_template.get('steps', [])]
    
        # Check if step names and order match
        if user_steps != default_steps:
            error_msg = (
                f"User template '{recipe_filename}' has different step structure than default.\n"
                f"User template steps: {user_steps}\n"
                f"Default template steps: {default_steps}\n"
                f"When enforce_template_structure=True, user templates must have the same "
                f"step names in the same order as default templates. You can modify step "
                f"keywords and calibs, but not add/remove/reorder steps.\n"
                f"To allow different step structures, set enforce_template_structure=False "
                f"in your config file or via environment variable."
            )
            raise ValueError(error_msg)

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
    recipe_filepath = None
    if isinstance(template, str):
        if os.path.sep not in template:
            # this is just a template name. check user_templates first, then defaults
            template, recipe_filepath, is_user_template = _load_recipe_template(template)
            # Validate user template structure if enabled
            if is_user_template and corgidrp.enforce_template_structure:
                default_template_path = os.path.join(recipe_dir, os.path.basename(recipe_filepath))
                _validate_template_structure(template, default_template_path, os.path.basename(recipe_filepath))
        else:
            recipe_filepath = template
            template = json.load(open(recipe_filepath, 'r'))
            is_user_template = False  # explicit paths are not user templates

    # generate recipe
    recipes = autogen_recipe(filelist, outputdir, template=template)

    if recipe_filepath is not None:
        if isinstance(recipes, list):
            for r in recipes:
                if isinstance(r, list):
                    for sub_r in r:
                        sub_r["RECIPE_SRC"] = recipe_filepath
                else:
                    r["RECIPE_SRC"] = recipe_filepath
        else:
            recipes["RECIPE_SRC"] = recipe_filepath


    if not isinstance(recipes, list):
        recipes = [recipes]
    # accommodate a list of chains
    if not isinstance(recipes[0], list):
        list_of_recipe_chains = [recipes]
    else:
        list_of_recipe_chains = recipes

    for recipes in list_of_recipe_chains:
        # process recipes
        output_filelist = None
        for i, recipe in enumerate(recipes):
            # check for recipe chaining
            if i > 0 and  len(recipe['inputs']) == 0:
                recipe["inputs"] = []
                for filename in output_filelist:
                    recipe["inputs"].append(filename)

            # check for functions that require CPGS XML info
            for step in recipe['steps']:
                if step['name'].lower() == 'find_spec_star':
                    if not 'keywords' in step:
                        read_cpgs = True
                        step['keywords'] = {}
                    elif "r_lamD" not in step['keywords']:
                        read_cpgs = True
                    else:
                        read_cpgs = False

                    if read_cpgs: # if not already specified.
                        # need to populate satellite spot info from XML
                        cpgs_xml = ET.parse(CPGS_XML_filepath)
                        sat_spot_info = _get_satellite_spot_info_from_xml(cpgs_xml)
                        step['keywords']['r_lamD'] = sat_spot_info['spot1_sep']
                        step['keywords']['phi_deg'] = sat_spot_info['spot1_angle']

            output_filelist = run_recipe(recipe)

    # return just the recipe if there was only one
    if len(list_of_recipe_chains) == 1:
        if len(list_of_recipe_chains[0]) == 1:
            return list_of_recipe_chains[0][0]
        else:
            return list_of_recipe_chains[0]
    else:
        return list_of_recipe_chains

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
        dataset0 = data.Dataset([filelist[0]])
        first_frame = dataset0[0]
        # don't need the actual data, especially if it would take up a lot of RAM just to hold it in cache
        dataset = data.Dataset(filelist, no_data=True, no_err=True, no_dq=True)

    # if user didn't pass in template
    if template is None:
        recipe_filename, chained = guess_template(dataset)

        # handle it as a list of lists moving forward
        if isinstance(recipe_filename, list):
            recipe_filename_list = recipe_filename
        else:
            recipe_filename_list = [recipe_filename]
        if not isinstance(recipe_filename_list[0], list):
            recipe_filename_list_list = [recipe_filename_list]
        else:
            recipe_filename_list_list = recipe_filename_list
        for l in recipe_filename_list_list:
            if not isinstance(l, list):
                raise TypeError("Each element of recipe_filename_list should be a list, but got {0}".format(type(l)))

        recipe_template_list_list = []
        template_sources = []  # Track sources for logging
        for recipe_filename_list in recipe_filename_list_list:
            recipe_template_list = []
            for recipe_filename in recipe_filename_list:
                # load the template recipe (check user_templates first, then defaults)
                template, template_path, is_user_template = _load_recipe_template(recipe_filename)
                # If user template loaded and validation enabled, validate structure
                if is_user_template and corgidrp.enforce_template_structure:
                    # Load default template for comparison
                    default_template_path = os.path.join(recipe_dir, recipe_filename)
                    _validate_template_structure(template, default_template_path, recipe_filename)
                recipe_template_list.append(template)
                template_sources.append((recipe_filename, template_path, is_user_template))
            recipe_template_list_list.append(recipe_template_list)
    else:
        # user passed in a single template
        # Check if it's a string (filename) or already loaded dict
        if isinstance(template, str):
            # It's a filename, load it (check user_templates first)
            template_dict, template_path, is_user_template = _load_recipe_template(template)

            # If user template loaded and validation enabled, validate structure
            if is_user_template and corgidrp.enforce_template_structure:
                # Load default template for comparison
                default_template_path = os.path.join(recipe_dir, template)
                _validate_template_structure(template_dict, default_template_path, template)

            recipe_template_list = [template_dict]
            template_sources = [(template, template_path, is_user_template)]
        else:
            # It's already a loaded dict
            recipe_template_list = [template]
            template_sources = []

        recipe_template_list_list = [recipe_template_list]
        chained = False

    recipe_list_list = []
    source_idx = 0  # Index into template_sources list
    for recipe_template_list in recipe_template_list_list:
        recipe_list = []
        for i, template in enumerate(recipe_template_list):
            # create the personalized recipe
            recipe = template.copy()
            recipe["template"] = False

            # Add template source for traceability
            if source_idx < len(template_sources):
                _, template_path, is_user_template = template_sources[source_idx]
                recipe["RECIPE_SRC"] = template_path
                source_idx += 1

            # for chained recipes, don't put the input in yet since we don't know it
            if i > 0 and chained:
                pass
            else:
                # add pc and analog inputs separately at L2a level
                # for L2a to L2b process photon-counting and analog frames separately
                if first_frame is not None and first_frame.ext_hdr['DATALVL'] == "L2a":
                    split_datasets, unique_vals = dataset.split_dataset(exthdr_keywords=['ISPC'])
                    if len(unique_vals) > 1:
                        for partial_dataset, isPc in zip(split_datasets, unique_vals):
                            if "pc" in recipe["name"] and isPc == 1:
                                for frame in partial_dataset:
                                    recipe["inputs"].append(frame.filename)
                            elif "pc" not in recipe["name"] and isPc == 0:
                                for frame in partial_dataset:
                                    recipe["inputs"].append(frame.filename)

                    else:
                        for filename in filelist:
                            recipe["inputs"].append(filename)
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

                if step["name"].lower() == "create_nd_filter_cal_spec":
                    if "keywords" in step and step["keywords"].get("outputdir", "").upper() == "AUTOMATIC":
                        step["keywords"]["outputdir"] = recipe["outputdir"]

            recipe_list.append(recipe)
        recipe_list_list.append(recipe_list)

    # Print the recipes
    print("\n" + "="*60)
    user_templates_dir = corgidrp.user_templates_dir
    if len(recipe_list_list) > 1:
        print("Generated Recipe Chains:")
        for chain_idx, recipe_list in enumerate(recipe_list_list):
            print(f"\nChain {chain_idx + 1}:")
            for recipe_idx, recipe in enumerate(recipe_list):
                print(f"\n  Recipe {recipe_idx + 1}: {recipe['name']}")
                if 'RECIPE_SRC' in recipe:
                    src_label = " (user template)" if user_templates_dir in recipe['RECIPE_SRC'] else " (default template)"
                    print(f"    Source: {recipe['RECIPE_SRC']}{src_label}")
    else:
        if len(recipe_list_list[0]) > 1:
            print("Generated Recipe Chain:")
            for recipe_idx, recipe in enumerate(recipe_list_list[0]):
                print(f"\nRecipe {recipe_idx + 1}: {recipe['name']}")
                if 'RECIPE_SRC' in recipe:
                    src_label = " (user template)" if user_templates_dir in recipe['RECIPE_SRC'] else " (default template)"
                    print(f"  Source: {recipe['RECIPE_SRC']}{src_label}")
        else:
            print(f"Generated Recipe: {recipe_list_list[0][0]['name']}")
            if 'RECIPE_SRC' in recipe_list_list[0][0]:
                recipe = recipe_list_list[0][0]
                src_label = " (user template)" if user_templates_dir in recipe['RECIPE_SRC'] else " (default template)"
                print(f"  Source: {recipe['RECIPE_SRC']}{src_label}")
    print("="*60 + "\n")

    # if list of chains, return that.  If single list, return that.  If single
    # recipe, return that.
    if len(recipe_list_list) > 1: # list of chains
        return recipe_list_list
    else:
        if len(recipe_list_list[0]) > 1: # single list
            return recipe_list_list[0]
        else: #single recipe
            return recipe_list_list[0][0]

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
        str or list: the best template filename, a list of multiple template filenames, or a list of template chains
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
        # elif image.pri_hdr['VISTYPE'][:11] == "CGIVST_ENG_":
        #     # if this is an ENG calibration visit
        #     # for either pupil or image
        #     recipe_filename = "l1_to_l2a_eng.json"
        elif image.pri_hdr['VISTYPE'] == "CGIVST_CAL_BORESIGHT":
            recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", 'l2b_to_boresight.json'] #"l1_to_boresight.json"
            chained = True
        elif image.pri_hdr['VISTYPE'] == "CGIVST_CAL_FLAT":

            if image.ext_hdr.get('DPAMNAME', '') in ('POL0', 'POL45'):
                recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_polflat.json"]
                chained = True
            else:
                recipe_filename = "l1_flat_and_bp.json"
        elif image.pri_hdr['VISTYPE'] == "CGIVST_CAL_DRK":
            _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
            if image.ext_hdr['ISPC'] == 1:
                recipe_filename = ["l1_to_l2b_pc_dark_1.json", "l1_to_l2b_pc_dark_2.json"]# "l1_to_l2b_pc_dark.json"
                chained = True
            elif len(unique_vals) > 1: # darks for noisemap creation
                recipe_filename = ["l1_to_l2a_noisemap_1.json", "l1_to_l2a_noisemap_2.json"]#"l1_to_l2a_noisemap.json"
                chained = True
            else: # then len(unique_vals) is 1 and not PC: traditional darks
                recipe_filename = ["build_trad_dark_image_1.json", "build_trad_dark_image_2.json"] #"build_trad_dark_image.json"
                chained = True
        elif image.pri_hdr['VISTYPE'] == "CGIVST_CAL_PUPIL_IMAGING":
            recipe_filename = [["l1_to_l2a_nonlin_1.json", "l1_to_l2a_nonlin_2.json", "l1_to_l2a_nonlin_3.json"],
                               ["l1_to_kgain_1.json", "l1_to_kgain_2.json"]] # ["l1_to_l2a_nonlin.json","l1_to_kgain.json"]
            chained = True # in this case, each sub-list is chained
        elif image.pri_hdr['VISTYPE'] in ("CGIVST_CAL_ABSFLUX_FAINT", "CGIVST_CAL_ABSFLUX_BRIGHT"):
            is_spec_mode = image.ext_hdr.get('DPAMNAME', '').startswith('PRISM')
            has_nd_filter = any(img.ext_hdr.get('FPAMNAME', '').startswith('ND') for img in dataset)
            if is_spec_mode:
                if has_nd_filter:
                    recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b_spec.json", "l2b_to_nd_filter_spec.json"]
                    chained = True
                else:
                    recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b_spec.json", "l2b_to_spec_flux.json"]
                    chained = True
            else:
                _, fsm_unique = dataset.split_dataset(exthdr_keywords=['FSMX', 'FSMY'])
                if len(fsm_unique) > 1:
                    recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", "l2b_to_nd_filter.json"]
                    chained = True
                else:
                    # Check for polarimetry mode to use appropriate flux calibration recipe
                    if image.ext_hdr.get('DPAMNAME', '') in ['POL0', 'POL45']:
                        recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b_pol.json", "l2b_to_fluxcal_factor_pol.json"]
                    else:
                        recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", "l2b_to_fluxcal_factor.json"]
                    chained = True
        elif image.pri_hdr['VISTYPE'] == 'CGIVST_CAL_CORETHRPT':
            recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b.json", 'l2b_to_corethroughput.json']
            chained = True
        elif image.pri_hdr['VISTYPE'] == 'CGIVST_CAL_SPEC_LINESPREAD':
            recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b_spec.json", 'l2b_to_spec_linespread.json']
            chained = True
        elif image.pri_hdr['VISTYPE'] == 'CGIVST_CAL_TPUMP':
            recipe_filename = ['trap_pump_cal_1.json', 'trap_pump_cal_2.json']
            chained = True
        elif image.pri_hdr['VISTYPE'] == 'CGIVST_CAL_POL_SETUP':
            recipe_filename = ["l1_to_l2a_basic.json", "l2a_to_l2b_pol.json", "l2b_to_polcal.json"]
            chained = True
        else:
            recipe_filename = "l1_to_l2a_basic.json"
              # science data and all else (including photon counting)

    # L2a -> L2b data processing
    elif image.ext_hdr['DATALVL'] == "L2a":
        if image.pri_hdr['VISTYPE'] == "CGIVST_CAL_DRK":
            _, unique_vals = dataset.split_dataset(exthdr_keywords=['EXPTIME', 'EMGAIN_C', 'KGAINPAR'])
            if image.ext_hdr['ISPC'] == 1:
                recipe_filename = ["l2a_to_l2b_pc_dark_1.json", "l2a_to_l2b_pc_dark_2.json"]#"l2a_to_l2b_pc_dark.json"
                chained = True
            elif len(unique_vals) > 1: # darks for noisemap creation
                recipe_filename = ["l2a_to_l2a_noisemap_1.json", "l2a_to_l2a_noisemap_2.json"] # "l2a_to_l2a_noisemap.json"
                chained = True
            else: # then len(unique_vals) is 1 and not PC: traditional darks
                recipe_filename = ["l2a_build_trad_dark_image_1.json", "l2a_build_trad_dark_image_2.json"] #"l2a_build_trad_dark_image.json"
                chained = True
        else:
            # Check if this is spectroscopy data (DPAMNAME in [PRISM3, PRISM2], not sure of VISTYPE yet)
            is_spectroscopy = image.ext_hdr.get('DPAMNAME', '') in ['PRISM3', 'PRISM2']

            is_polarimetry = image.ext_hdr.get('DPAMNAME', '') in ['POL0', 'POL45']

            _, unique_vals = dataset.split_dataset(exthdr_keywords=['ISPC'])

            if len(unique_vals) > 1: #Satspots are not PC
                if is_spectroscopy:
                    recipe_filename = [["l2a_to_l2b_pc_spec_1.json", "l2a_to_l2b_pc_spec_2.json", "l2a_to_l2b_pc_spec_3.json"], ["l2a_to_l2b_spec.json"]] #"l2a_to_l2b_pc_spec.json"
                elif is_polarimetry:
                    recipe_filename = [["l2a_to_l2b_pc_1.json", "l2a_to_l2b_pc_2.json", "l2a_to_l2b_pol_pc_3.json"],["l2a_to_l2b_pol.json"]] #"l2a_to_l2b_pc_pol.json"
                else:
                    recipe_filename = [["l2a_to_l2b_pc_1.json", "l2a_to_l2b_pc_2.json", "l2a_to_l2b_pc_3.json"], ["l2a_to_l2b.json"]]#l2a_to_l2b_pc.json
                chained = True


            else:
                if is_spectroscopy:
                    if image.ext_hdr['ISPC'] == 1:
                        recipe_filename = ["l2a_to_l2b_pc_spec_1.json", "l2a_to_l2b_pc_spec_2.json", "l2a_to_l2b_pc_spec_3.json"] #"l2a_to_l2b_pc_spec.json"
                        chained = True

                    else:
                        recipe_filename = "l2a_to_l2b_spec.json"

                elif is_polarimetry:
                    if image.ext_hdr['ISPC'] == 1:
                        recipe_filename = ["l2a_to_l2b_pc_1.json", "l2a_to_l2b_pc_2.json", "l2a_to_l2b_pol_pc_3.json"] #"l2a_to_l2b_pc_pol.json"
                        chained = True
                    else:
                        recipe_filename = "l2a_to_l2b_pol.json"
                else:
                    if image.ext_hdr['ISPC'] == 1:
                        recipe_filename = ["l2a_to_l2b_pc_1.json", "l2a_to_l2b_pc_2.json", "l2a_to_l2b_pc_3.json"] #l2a_to_l2b_pc.json
                        chained = True
                    else:
                        recipe_filename = "l2a_to_l2b.json"  # science data and all else
    # L2b -> L3 data processing
    elif image.ext_hdr['DATALVL'] == "L2b":
        if image.pri_hdr['VISTYPE'] in ("CGIVST_CAL_ABSFLUX_FAINT", "CGIVST_CAL_ABSFLUX_BRIGHT"):
            is_spec_mode = image.ext_hdr.get('DPAMNAME', '').startswith('PRISM')
            has_nd_filter = any(img.ext_hdr.get('FPAMNAME', '').startswith('ND') for img in dataset)
            if is_spec_mode:
                if has_nd_filter:
                    recipe_filename = "l2b_to_nd_filter_spec.json"
                else:
                    recipe_filename = "l2b_to_spec_flux.json"
            else:
                _, fsm_unique = dataset.split_dataset(exthdr_keywords=['FSMX', 'FSMY'])
                if len(fsm_unique) > 1:
                    recipe_filename = "l2b_to_nd_filter.json"
                else:
                    if image.ext_hdr['DPAMNAME'] == 'POL0' or image.ext_hdr['DPAMNAME'] == 'POL45':
                        recipe_filename = 'l2b_to_fluxcal_factor_pol.json'
                    else:
                        recipe_filename = "l2b_to_fluxcal_factor.json"
        elif image.pri_hdr['VISTYPE'] == 'CGIVST_CAL_CORETHRPT':
            recipe_filename = 'l2b_to_corethroughput.json'
        elif image.pri_hdr['VISTYPE'] == "CGIVST_CAL_POL_SETUP":
            recipe_filename = "l2b_to_polcal.json"
        elif image.ext_hdr['DPAMNAME'] == 'POL0' or image.ext_hdr['DPAMNAME'] == 'POL45':
            recipe_filename = "l2b_to_l3_pol.json"
        elif 'TDD' not in image.pri_hdr['VISTYPE'] and image.pri_hdr['VISTYPE'] != "CGIVST_CAL_SPEC_TGTREF":
            warnings.warn("Only VISTYPE TDD and certain cal frames should be processed beyond L2b. Double-check which frames are being processed from L2b -> L3.")
            recipe_filename = "l2b_to_l3.json"
        else:
            recipe_filename = "l2b_to_l3.json"
    # L3 -> L4 data processing
    elif image.ext_hdr['DATALVL'] == "L3":
        if image.ext_hdr['DPAMNAME'] == 'POL0' or image.ext_hdr['DPAMNAME'] == 'POL45':
            recipe_filename = "l3_to_l4_pol.json"
        elif image.ext_hdr['DPAMNAME'] in ['PRISM3', 'PRISM2']:
            if image.pri_hdr['VISTYPE'] != 'CGIVST_CAL_SPEC_TGTREF':
                # coronagraphic spec obs - PSF subtraction
                recipe_filename = "l3_to_l4_psfsub_spec.json"
            else:
                # noncoronagraphic spec obs - no PSF subtraction
                recipe_filename = "l3_to_l4_noncoron_spec.json"
        else:
            if image.pri_hdr['VISTYPE'] != 'CGIVST_CAL_TGTREF_PHOT':
                # coronagraphic obs - PSF subtraction
                recipe_filename = "l3_to_l4.json"
            else:
                # noncoronagraphic obs - no PSF subtraction
                recipe_filename = "l3_to_l4_nopsfsub.json"
    else:
        raise NotImplementedError("Cannot automatically guess the input dataset with 'DATALVL' = {0}".format(image.ext_hdr['DATALVL']))
    return recipe_filename, chained

def save_data(dataset_or_image, outputdir, suffix="", ram_heavy_save=False):
    """
    Saves the dataset or image that has currently been outputted by the last step function.
    Records calibration frames into the caldb during the process

    Args:
        dataset_or_image (corgidrp.data.Dataset or corgidrp.data.Image): data to save
        outputdir (str): path to directory where files should be saved
        suffix (str): optional suffix to tack onto the filename.
                      E.g.: `test.fits` with `suffix="dark"` becomes `test_dark.fits`
        ram_heavy_save (bool):  If True, the input is assumed to have no data loaded into memory. (Only metadata was
            manipulated in step leading up to save_data.) The data is loaded from the filepath frame by frame, and
            each Image is saved to outputdir.  Defaults to False.
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
    dataset.save(filedir=outputdir, filenames=filenames, ram_heavy_save=ram_heavy_save)

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
    # these settings should only apply to this recipe, so we will restore old settings later
    old_settings = {}
    for setting in recipe.get("drpconfig", {}):
        # capture previous value before overriding
        old_settings[setting] = getattr(corgidrp, setting, None)
        setattr(corgidrp, setting, recipe["drpconfig"][setting])

    # save recipe before running recipe
    if save_recipe_file:
        recipe_filename = "{0}_{1}_recipe.json".format(recipe["name"], time.Time.now().isot)
        recipe_filename = recipe_filename.replace(":", ".")  # replace colons with periods for compatibility with Windows machines
        recipe_filepath = os.path.join(recipe["outputdir"], recipe_filename)
        with open(recipe_filepath, "w") as json_file:
            json.dump(recipe, json_file, indent=4)

    # determine if this is a RAM-heavy recipe which needs crop-stack processing
    #sort_pupilimg_frames included here b/c it sorts through a large number of frame (>700) b/c
    # EM gain cal files (sorted here and excluded, processed by SSC) are included in the visits
    if "ram_heavy" in recipe:
        ram_heavy_bool = bool(recipe["ram_heavy"])
    else:
        ram_heavy_bool = False
    if "process_in_chunks" in recipe:
        ram_increment_bool = bool(recipe["process_in_chunks"])
    else:
        ram_increment_bool = False
    if ram_heavy_bool and ram_increment_bool:
        warnings.warn('\'ram_heavy\' supercedes \'process_in_chunks\', so frames will be read in all at once with no data loaded.')

    # read in data, if not doing bp map
    if not recipe["inputs"]:
        curr_dataset = []
        ram_heavy_bool = False
        filelist_chunks = [0] #anything of length 1
    else:
        filelist = recipe["inputs"]
        if ram_increment_bool and not ram_heavy_bool: #ram_heavy_bool supercedes ram_increment_bool
            # how many frames to process at a time (before getting the RAM-heaviest function in the recipe) if RAM-heavy
            filelist_chunks = [filelist[n:n+corgidrp.chunk_size] for n in range(0, len(filelist), corgidrp.chunk_size)]
        else:
            filelist_chunks = [filelist]

    try:
        tot_steps = len(recipe["steps"])
        save_step = False
        output_filepaths = []
        for filelist in filelist_chunks:
            if recipe["inputs"]:
                if ram_heavy_bool:
                    curr_dataset = data.Dataset(filelist, no_data=True, no_err=True, no_dq=True)
                    recipe_temp = recipe.copy()
                    # don't want to keep all ~26000 filepaths in all ~26000 ext headers b/c that's a lot of memory
                    recipe_temp["inputs"] = "See RECIPE header value in {0}".format(curr_dataset[-1].filepath)
                else:
                    curr_dataset = data.Dataset(filelist)
                    recipe_temp = recipe
                # write the recipe into the image extension header
                curr_dataset[-1].ext_hdr["RECIPE"] = json.dumps(recipe)
                if len(curr_dataset) > 1:
                    for frame in curr_dataset[:-1]:
                        frame.ext_hdr["RECIPE"] = json.dumps(recipe_temp)
            # execute each pipeline step
            print('Executing recipe: {0}'.format(recipe['name']))
            if isinstance(filelist, list):
                print('number of frames: ', len(filelist))
            if ram_increment_bool and len(filelist_chunks) > 1:
                print('Processing frames in chunks of {0} frames'.format(corgidrp.chunk_size))
            if ram_heavy_bool:
                print('Processing frames in RAM-heavy mode (data not loaded into memory until necessary, one frame at a time)')
            for i, step in enumerate(recipe["steps"]):
                print("Walker step {0}/{1}: {2}".format(i+1, tot_steps, step["name"]))
                if step["name"].lower() == "save":
                    # special save instruction

                    # see if suffix is specified as a keyword
                    if "keywords" in step and "suffix" in step["keywords"]:
                        suffix =  step["keywords"]["suffix"]
                    else:
                        suffix = ''
                    if "keywords" in step and "ram_heavy_save" in step["keywords"]:
                        ram_heavy_save = step["keywords"]["ram_heavy_save"]
                    else:
                        ram_heavy_save = False
                    save_data(curr_dataset, recipe["outputdir"], suffix=suffix, ram_heavy_save=ram_heavy_save)
                    if isinstance(curr_dataset, data.Dataset):
                        output_filepaths += [frame.filepath for frame in curr_dataset]
                    else:
                        output_filepaths += [curr_dataset.filepath]
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
                            if ram_heavy_bool:
                                ref_image = data.Image(ref_image.filepath) #load in data for calibration matching
                            _fill_in_calib_files(step, this_caldb, ref_image)

                            # also update the recipe we used in the headers
                            if ram_heavy_bool:
                                recipe_temp = recipe.copy()
                                # don't want to keep all ~26000 filepaths in all ~26000 ext headers b/c that's a lot of memory
                                recipe_temp["inputs"] = "See RECIPE header value in {0}".format(curr_dataset[-1].filepath)
                            else:
                                recipe_temp = recipe
                            list_of_frames[-1].ext_hdr["RECIPE"] = json.dumps(recipe)
                            if len(list_of_frames) > 1:
                                for frame in list_of_frames[:-1]:
                                    frame.ext_hdr["RECIPE"] = json.dumps(recipe_temp)

                        # load the calibration files in from disk
                        for calib in step["calibs"]:
                            if step["calibs"][calib] is not None:
                                # special case for pol flat because it has multiple files
                                if calib == "FlatFieldPOL0" or calib == "FlatFieldPOL45":
                                    calib_dtype = data.datatypes['FlatField']
                                elif calib == "FluxcalFactorPOL0" or calib == "FluxcalFactorPOL45":
                                    calib_dtype = data.datatypes['FluxcalFactor']
                                else:
                                    calib_dtype = data.datatypes[calib]
                                cal_file = calib_dtype(step["calibs"][calib])
                            else:
                                cal_file = None
                            other_args += (cal_file,)

                    kwargs = step.get("keywords", {})

                    # run the step!
                    curr_dataset = step_func(curr_dataset, *other_args, **kwargs)

                    # make sure RECIPE header is propagated to output
                    if isinstance(curr_dataset, data.Dataset):
                        for frame in curr_dataset:
                            if "RECIPE" not in frame.ext_hdr:
                                frame.ext_hdr["RECIPE"] = json.dumps(recipe)
                    elif hasattr(curr_dataset, 'ext_hdr') and "RECIPE" not in curr_dataset.ext_hdr:
                        curr_dataset.ext_hdr["RECIPE"] = json.dumps(recipe)

        if not save_step:
            output_filepaths = None

        return output_filepaths
    finally:
        # restore old pipeline settings that this recipe overwrote (even if a step raises)
        for setting, old_val in old_settings.items():
            setattr(corgidrp, setting, old_val)

def _get_satellite_spot_info_from_xml(xml_tree):
    """
    Extracts satellite spot information from the CPGS XML file

    Args:
        xml_tree (ElementTree): loaded in CPGS XML file

    Returns:
        dict: dictionary with satellite spot information
            "num_spots": int, number of satellite spots
            "spot1_contrast": float, contrast of spot 1
            "spot1_sep": float, separation of spot 1 in lam/D
            "spo1_angle": float, angle of spot 1 in degrees
            "spot2_contrast": float, contrast of spot 2
            "spot2_sep": float, separation of spot 2 in lam/D
            "spo2_angle": float, angle of spot 2 in degrees
    """
    obs_specification = xml_tree.getroot()
    cpgs_input_info = obs_specification.find("cpgs_input")
    try:
        sat_spot_info = cpgs_input_info.find("satellite_spots_command_info")
    except:
        raise Exception("Field with satellite spot info 'satellite_spots_command_info' not found in CPGS XML file. Please ensure that the CPGS file is formatted correctly.")
    sat_spot_output = {}
    sat_spot_output['num_spots'] = 0
    string = re.compile(r'sep[1-2]=[\d.]+,\s*angle[1-2]=[\d.]+,\s*fr[1-2]=[\w.-]+,\s*filt[1-2]=[\w]+')
    sat_spots = string.findall(sat_spot_info.text)
    key = ['sep','angle','contrast','filt']
    for sat_spot in sat_spots:
        fields = sat_spot.split(",")
        sat_spot_output['num_spots'] += 1
        for i, field in enumerate(fields):
            value = field.split("=")[1]
            if i <=2:
                sat_spot_output[f'spot{sat_spot_output['num_spots']}_{key[i]}'] = float(value)
            else:
                sat_spot_output[f'spot{sat_spot_output['num_spots']}_{key[i]}'] = str(value)

    return sat_spot_output
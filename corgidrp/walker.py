import os
import json
import astropy.time as time
import corgidrp
import corgidrp.data as data
import corgidrp.caldb as caldb
import corgidrp.l1_to_l2a

all_steps = {
    "prescan_biassub" : corgidrp.l1_to_l2a.prescan_biassub,
    "detect_cosmic_rays" : corgidrp.l1_to_l2a.detect_cosmic_rays,
    "correct_nonlinearity" : corgidrp.l1_to_l2a.correct_nonlinearity,
    "update_to_l2a" : corgidrp.l1_to_l2a.update_to_l2a
}

recipe_dir = os.path.join(os.path.dirname(__file__), "recipe_templates")

def autogen_recipe(filelist, outputdir, template=None):
    """
    Automatically creates a recipe by identifyng and populating a template

    Args:
        filelist (list of str): list of filepaths to files
        outputdir (str): output directory folderpath
        template (json): enables passing in of custom template, if desired

    Returns:
        json: the JSON recipe to process the filelist
    """
    # load the first frame to check what kind of data and identify recipe
    first_frame = data.autoload(filelist[0])
    
    # if user didn't pass in template
    if template is None:
        recipe_filename = guess_template(first_frame)

        # load the template recipe
        recipe_filepath = os.path.join(recipe_dir, recipe_filename)
        template = json.load(open(recipe_filepath, 'r'))

    # create the personalized recipe
    recipe = template.copy()
    recipe["template"] = False

    for filename in filelist:
        recipe["inputs"].append(filename)
    
    recipe["outputdir"] = outputdir

    ## Populate calibration files that need to be automatically populated
    this_caldb = caldb.CalDB()
    for step in recipe["steps"]:
        if "calibs" in step:
            for calib in step["calibs"]:
                # order matters, so only one calibration file per dictionary
                
                if step["calibs"][calib].upper() == "AUTOMATIC":
                    calib_dtype = data.datatypes[calib]
                    best_cal_file = this_caldb.get_calib(first_frame, calib_dtype)
                    # set calibration file to this one
                    step["calibs"][calib] = best_cal_file.filepath

    return recipe


def guess_template(image):
    """
    Guesses what template should be used to process a specific image

    Args:
        image (corgidrp.data.Image): an Image file to process

    Returns:
        str: the best template filename
    """
    if image.ext_hdr['DATA_LEVEL'] == "L1":
        if image.pri_hdr['OBSTYPE'] == "ENG":
            recipe_filename = "l1_to_l2a_eng.json"
        else:
            recipe_filename = "l1_to_l2a_basic.json"
    else:
        raise NotImplementedError()

    return recipe_filename

def run_recipe(recipe, save_recipe_file=True):
    """
    Run the specified recipe

    Args:
        recipe (dict or str): either the filepath to the recipe or the already loaded in recipe
        save_recipe_file (bool): saves the recipe as a JSON file in the outputdir (true by default)
    """
    if isinstance(recipe, str):
        # need to load in
        recipe = json.load(open(recipe, "r"))
    
    # configure pipeline as needed
    for setting in recipe['drpconfig']:
        # equivalent to corgidrp.setting = recipe['drpconfig'][setting]
        setattr(corgidrp, setting, recipe['drpconfig'][setting])

    # read in data
    filelist = recipe["inputs"]
    curr_dataset = data.Dataset(filelist)

    # write the recipe into the image extension header
    for frame in curr_dataset:
        frame.ext_hdr["RECIPE"] = json.dumps(recipe)

    # save recipe before running recipe
    if save_recipe_file:
        recipe_filename = "{0}_{1}_recipe.json".format(recipe["name"], time.Time.now().isot)
        recipe_filepath = os.path.join(recipe["outputdir"], recipe_filename)
        with open(recipe_filepath, "w") as json_file:
            json.dump(recipe, json_file, indent=4)

    # execute each pipeline step
    for step in recipe["steps"]:
        if step["name"].lower() == "save":
            # special save instruction
            curr_dataset.save(recipe["outputdir"])

        else:
            step_func = all_steps[step["name"]]
            
            # TODO: handle calibrations; any other possible required args?
            other_args = ()
            if "calibs" in step:
                for calib in step["calibs"]:
                    calib_dtype = data.datatypes[calib]
                    cal_file = calib_dtype(step["calibs"][calib])
                    other_args += (cal_file,)
                

            if "keywords" in step:
                kwargs = step["keywords"]
            else:
                kwargs = {}

            # run the step!
            curr_dataset = step_func(curr_dataset, *other_args, **kwargs)





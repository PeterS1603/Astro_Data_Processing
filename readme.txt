
Persistence_correction is a class based approach to correcting persistence in exposure series. It utilizes a persistence model created uniquely foreach detector (determined through a separate class and utilizing specialized data sets) to remove artifacts caused by image latency stored as a .json file. Initializing the function with the .json file images can be corrected for in sequence. The following shows pseudo code for how persistence_correction can be implemented into a preexisting data reduction pipeline given the data to create the model was raw.


From Persist_Class.py import persistence_correction

persist = persistence_correction(json_file_path)


For filename in img_series:

	determine background value for the image

	persistence_corrected_img = persist.correct_img(filename)

		#Note that the first image will note be corrected for persistence

	persist_mask = persist.master_mask #mask of pixels in next image that will be corrected 

	background subtract image #First image doesn't need to consider the persist mask but subsequent images in the series should background subtract ~persist_mask for TMMT json model

From this point image correction can continue as normal with persistence_corrected_img being the image data utilized in further reduction.


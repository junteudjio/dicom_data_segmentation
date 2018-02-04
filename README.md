# dicom_data_segmentation : A framework for DICOM images segmentation.

----

## Installation

- First step:
```bash
$ git clone repo-url
$ cd repo-dir/dicom_data_pipeline
$ python setup.py install|develop
```
Doing this will install  dicom_data_pipeline and its dependencies. and also scikit-image.

- Second and last step: install Pytorch on your machine by following the instructions on http://pytorch.org/ .

## How to run

Make sure every script is ran from the directory dicom_data_segmentation/dicom_data_segmentation.

This package and its cousin dicom_data_pipeline integrates painlessly. So:
- run dicom_data_segmentation/dicom_data_segmentation/run_data_parser.py to parse DICOM images.
- run dicom_data_segmentation/dicom_data_segmentation/run_segmentation.py to segment those images.
- Each time check for new files and folders created in _logs, _data, _plots, _results directories (editable defaults).


## Part 1 : Parse the o-contours

### Changes added to dicom_data_pipeline:

- Now it can handle n-contours at once (not only 2), to make sure we don't have to refactor anymore if we need to handle 3 or more contours at once.

- The main change was to update the semantic of the parameter contour_type: can now be (single) i-contour, (multi with '.' in the middle) i-contour.o-contour .
Doing the refactoring this way enables us to keep most of our code unchanged.

- Another one is the way of constructing matching pairs/triplets/etc... (img, contours):
Now we find the images for which we have all the contours first in order to reduce the number of time we have to do glob.glob('pattern')
More on this by looking at DataParser._get_dicoms_contours_paths method

- Also very important: change everywhere I wrote something like img_contour_pairs to img_contours. to avoid any confusion to someone reading the code.
If not because of this change in particular, the existing tests cases before refactoring would have remain the same.

- Updated the test as well to reflect the changes in the logic: so new tests cases when loading single, multi contour(s).



## Part 2 : Heuristic LV Segmentation approaches

Please refer to this notebooks where every thing is illustrated with plots and images (preferably in this order):

- FixedThresholdSegmentation.ipynb
- FlexibleThresholdSegmentation.ipynb
- ActiveContourSegmentation.ipynb
- ComparativeAnalysis.ipynb


## Contributor

- junior Teudjio Mbativou | jun.teudjio@gmail.com | https://www.linkedin.com/in/junior-teudjio-3a125b8a


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
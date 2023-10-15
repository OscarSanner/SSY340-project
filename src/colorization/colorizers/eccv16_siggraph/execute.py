from .eccv16 import *
from .base_color import *
from .siggraph17 import *
from .util import *
import matplotlib.pyplot as plt

# Run this from .../src/ to colorize images
# python3 colorization/colorizers/eccv16_&_siggraph/demo_release.py -i ./test_img.png

def eccv16_colorize(source_folder, destination_folder, file_name, use_gpu=False):
	# load colorizers
	colorizer_eccv16 = eccv16(pretrained=True).eval()
	if(use_gpu):
		colorizer_eccv16.cuda()

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(f"{source_folder}/{file_name}")
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())

	plt.imsave(f"{destination_folder}/{file_name}", out_img_eccv16)


def siggraph17_colorize(source_folder, destination_folder, file_name, use_gpu=False):
	# load colorizers
	colorizer_siggraph17 = siggraph17(pretrained=True).eval()
	if(use_gpu):
		colorizer_siggraph17.cuda()

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(f"{source_folder}/{file_name}")
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	plt.imsave(f"{destination_folder}/{file_name}", out_img_siggraph17)

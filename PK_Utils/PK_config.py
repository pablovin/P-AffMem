"""
NETWORK CONFIG FILE

"""
# min-batch size
size_batch=25

# size of hidden vector z
num_z_channels=50 	

# width and height of input image
size_image = 96

# path to save checkpoints, samples, and summary
save_dir='/home/pablo/Documents/Datasets/AffectNet/ExperimentLog/'

# value range of single pixels in an input image
image_value_range = (-1, 1) 

# path to training data
# in case you want to change this: don't let it contain an 's'
data_path = "/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Training/"

# path to validation data
# in case you want to change this: don't let it contain an 's'
validation_path  = '/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Validation/'

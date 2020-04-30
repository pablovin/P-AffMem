"""
NETWORK CONFIG FILE

"""
# min-batch size
size_batch=25

# size of hidden vector z
num_z_channels=50 	

# width and height of input image
size_image = 96
# value range of single pixels in an input image
image_value_range = (-1, 1)


# """Local"""
# save_dir='/home/pablo/Documents/Datasets/AffectNet/ExperimentLog/' #Local
# vggMat = "/home/pablo/Documents/Datasets/VGG-Face/vgg-face.mat" #Local
# data_path = "/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Training/" #Local
# validation_path  = '/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Validation/' #Local
# device = '/device:GPU:0'


"""GCloud"""
# save_dir='/home/pablo/dataset/AffectNet/ExperimentLog/' #GCloud
# data_path = "/home/pablo/dataset/AffectNet/AffectNetProcessed_Training/" #GCloud
# validation_path  ="/home/pablo/dataset/AffectNet/AffectNetProcessed_Validation/"
# vggMat = "/home/pablo/dataset/AffectNet/vgg-face.mat" #Gcloud
# device = '/device:CPU:0'

"""CoLab"""
save_dir='/content/dataset/ExperimentLog/'
data_path = "/content/dataset/AffectNetProcessed_Training/" #
validation_path  ="/content/dataset/AffectNetProcessed_Validation/"
vggMat = "/content/dataset/vgg-face.mat"
device = '/device:GPU:0'
import csv
import cv2
import os
finalImageSize = 96

"""Local"""
# imagesDirectory = "/home/pablo/Documents/Datasets/AffectNet/Manually_Annotated_Images" # Local
# csvFileValidation ="/home/pablo/Documents/Datasets/AffectNet/Manually_validation.csv"
# saveDirectoryValidation="/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Validation"
#
# csvFileTraining ="/home/pablo/Documents/Datasets/AffectNet/Manually_training.csv"
# saveDirectoryTraining="/home/pablo/Documents/Datasets/AffectNet/AffectNetProcessed_Training"

""" G Cloud """
imagesDirectory = "/home/pablovin/datasets/Manually_Annotated_Images" #Gcloud

csvFileValidation ="/home/pablo/dataset/AffectNet/Manually_validation.csv"
saveDirectoryValidation="/home/pablo/dataset/AffectNet/AffectNetProcessed_Validation/"

csvFileTraining ="/home/pablo/Documents/Datasets/AffectNet/Manually_training.csv"
saveDirectoryTraining="/home/pablo/dataset/AffectNet/AffectNetProcessed_Training/"

for csvFile, saveDirectory in zip((csvFileValidation, csvFileTraining), (saveDirectoryValidation,saveDirectoryTraining)):
    with open(csvFile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 0:
                try:
                    imgPath = imagesDirectory+"/"+row[0]
                    faceXY = (int(row[1]),int(row[2]))
                    faceSize = (int(row[3]),int(row[4]))
                    faceCategory = row[-3]
                    valence = row[-2]
                    arousal = row[-1]

                    # print ("imgPath:" + imgPath)

                    noFace = float(arousal) == -2 or float(faceCategory) > 7

                    imageName = row[0].split("/")[1].split(".")[0] + "__" + str(
                        faceCategory) + "__" + arousal + "__" + valence + ".png"


                    if os.path.isfile(imgPath) and not noFace and not os.path.isfile(saveDirectory + "/"+imageName):
                        img = cv2.imread(imgPath)

                        face = img[faceXY[0]:faceXY[0]+faceSize[0], faceXY[1]:faceXY[1]+faceSize[1]]
                        face = cv2.resize(face,(finalImageSize,finalImageSize))


                        cv2.imwrite(saveDirectory + "/"+imageName, face)

                        print("Saving:" + str(imageName))
                    else:
                        print ("Skip!")
                except:
                 print("Error!")
            line_count += 1




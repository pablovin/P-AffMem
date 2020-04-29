# -*- coding: utf-8 -*-
"""
Main P-AFFMem framework. We omitted some functions (image pre-processing, for example) to simplify the code.
The main purpose of the code is to facilitate the general understanding of our model, in particular the execution pipeline.

"""

def runModel():
      
    from AffectiveMemory import AffectiveMemory
    from metrics import concordance_correlation_coefficient_python


    datasetFolderTrain = "/trainExamples/"  # OMG_training
    datasetFolderTest = "/testExamples"  # OMG_Testing


    """ Loading the testset
    """
    dataLoader = VisionLoader_AffectNet_FrameBased_FileList_Dimensional.VisionLoader_AffChallenge2017(
        experimentManager.logManager, preProcessingProperties)


    dataLoader.loadTestData(datasetFolderTest, "", augmentData=False)

    initializeAffMem = True
    for video in OMGDataset:
	    videoFrames, originalLabel = dataLoader(datasetFolderTrain)
            encodedVideoFrames = []
	    for frame in videoFrames:
		    #for pratical reason, encode all the video frames using the PK encoder
		    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
				    # resotre the PK graph and checkpoints
				    new_saver = tf.train.import_meta_graph(
					'saved.meta')
				    new_saver.restore(sess, tf.train.latest_checkpoint(
					'.checkpoints'))
				    graph = tf.get_default_graph()
				    
				    #Get the PK encoder
				    encodedImages = sess.graph.get_tensor_by_name("encoder/Tanh:0")
				   
				    # Pre-process the original image
				    img = preProcess(frame, (96, 96))
				    query_images = numpy.tile(img, (48, 1, 1, 1))
				    feed_dict = {images_tensor: query_images}

		                    #encode the original image
				    encodedVideoFrames.append(sess.run(op_to_restore, feed_dict))
				   


            # initialize the affective memory 
    	    affMem = AffectiveMemory()
	    for frameNumber in range(len(videoFrames)):
	            
                    trainingEncodings = []
                    trainingLabels = []
                    # Obtain the PK representations for the first 25 frames of each video
                    if frameNumber < 25:
			    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
				    # restore graph
				    new_saver = tf.train.import_meta_graph(
					'saved.meta')
				    new_saver.restore(sess, tf.train.latest_checkpoint(
					'.checkpoints'))
				    graph = tf.get_default_graph()

				    op_to_restore = sess.graph.get_tensor_by_name("generator/Tanh:0")

				    arousal_tensor = graph.get_tensor_by_name("arousal_labels:0")
				    valence_tensor = graph.get_tensor_by_name("valence_labels:0")
				    images_tensor = graph.get_tensor_by_name("input_images:0")
				    encodedImages = sess.graph.get_tensor_by_name("encoder/Tanh:0")


				    # Obtain the desired arousal and valence
				    valenceOriginal = numpy.arange(-1.0, 1.0, 0.01)
				    valence = numpy.repeat(valenceOriginal, 7).reshape((48, 1))

				    arousal = [numpy.arange(-1.0, 1.0, 0.01)]
				    arousal = numpy.repeat(arousal, 7, axis=0)
				    arousal = numpy.asarray([item for sublist in arousal for item in sublist]).reshape((48, 1))



				    # Pre-process the original image
				    img = preProcess(videoFrames[frameNumber], (96, 96))
				    query_images = numpy.tile(img, (48, 1, 1, 1))
				    feed_dict = {arousal_tensor: arousal, valence_tensor: valence, images_tensor: query_images}

				    # Obtain the generated images
				    x = sess.run(op_to_restore, feed_dict)

				    # Obtain the generated images
				    for im in x:
				    	totalImages.append(im*255)

	  			        # Encode each of the generated images
				    	encodedInput = [preProcess(x*255, (96,96))]
					query_images = numpy.tile(img, (48, 1, 1, 1))
				    	feed_dict = {images_tensor: query_images}
				    	trainingEncodings.append(sess.run(op_to_restore, feed_dict))


				    # Create the training samples
				    trainingLabels = numpy.array([arousal,valence])

				    trainingLabels = numpy.reshape(expectedLabels,(len(arousal+1),2))

                    			   
		    if initializeAffectiveMemory:
		        affMem.initNetwork(encodedImages, labels=expectedLabels, True)
		        initializeAffectiveMemory = False
		   
                    #Append the current frame to the training encodings
                    trainingEncodings.append(encodedVideoFrames[frameNumber])

		    # Guarantee that the original image has no valid arousal or valence associated to it, thus, will not be used to update the neurons' labels
                    trainingLabels.append([50,50])

 	            #Train the model
                    affMem.train(encodedImages, labels=expectedLabels, epochs=10, a_treshold=0.4, l_rates=(0.087, 0.032))

                    #Evaluate over all the frames of the video
                    arousalCCC, valenceCCC = evaluate_affectiveMemory(encodedVideoFrames, originalLabel)
                    print ("Current Frames", str(frameNumber)," - ArousalCCC:", str(arousalCCC), " - ValenceCCC:", str(valenceCCC))

     

/**
 * @Author: Yuting Liu and Jude Shavlik.  
 *
 * Copyright 2017.  Free for educational and basic-research use.
 *
 * The main class for Lab3 of cs638/838.
 *
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 *
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in Lab3.java, insert that class here to simplify grading.
 *
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 */

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;

import javax.imageio.ImageIO;

public class Lab3 {

	private static int     imageSize = 32; // Images are imageSize x imageSize.  The provided data is 128x128, but this can be resized by setting this value (or passing in an argument).
	                                       // You might want to resize to 8x8, 16x16, 32x32, or 64x64; this can reduce your network size and speed up debugging runs.
	                                       // ALL IMAGES IN A TRAINING RUN SHOULD BE THE *SAME* SIZE.
	private static enum    Category { airplanes, butterfly, flower, grand_piano, starfish, watch };  // We'll hardwire these in, but more robust code would not do so.

	private static final Boolean    useRGB = true; // If true, FOUR units are used per pixel: red, green, blue, and grey.  If false, only ONE (the grey-scale value).
	private static       int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use red+blue+green+grey.  Otherwise just use the grey value.

	private static String    modelToUse = "deep"; // Should be one of { "perceptrons", "oneLayer", "deep" };  You might want to use this if you are trying approaches other than a Deep ANN.
	private static int       inputVectorSize;         // The provided code uses a 1D vector of input features.  You might want to create a 2D version for your Deep ANN code.  
	                                                  // Or use the get2DfeatureValue() 'accessor function' that maps 2D coordinates into the 1D vector.
	                                                  // The last element in this vector holds the 'teacher-provided' label of the example.

	private static double eta       =    0.5, fractionOfTrainingToUse = 1.0, dropoutRate = 0.,momentum = 0.0, weight_decay = 0.; // To turn off drop out, set dropoutRate to 0.0 (or a neg number).
	private static int    maxEpochs = 10000; // Feel free to set to a different value.
	private static int numKernels = 2;
	private static int kernelSize = 5;

	/*************************************************************************************************/
	static int size = ((imageSize - kernelSize + 1) / 2 - kernelSize + 1) / 2;
	static Layer layerInput = Layer.buildInputLayer(new Layer.Size(imageSize, imageSize));
	static Layer layerC1 = Layer.buildConvLayer(numKernels, numKernels, new Layer.Size(kernelSize, kernelSize), new Layer.Size(imageSize - kernelSize + 1, imageSize - kernelSize + 1));
	static Layer layerP1 = Layer.buildPoolingLayer(numKernels, new Layer.Size((imageSize - kernelSize + 1) / 2, (imageSize - kernelSize + 1) / 2));
	static PoolingLayer p1 = new PoolingLayer();
	static Layer layerC2 = Layer.buildConvLayer(numKernels * numKernels, numKernels, new Layer.Size(kernelSize, kernelSize), new Layer.Size((imageSize - kernelSize + 1) / 2 - kernelSize + 1, (imageSize - kernelSize + 1) / 2 - kernelSize + 1));
	static Layer layerP2 = Layer.buildPoolingLayer(numKernels * numKernels, new Layer.Size(size, size));
	static PoolingLayer p2 = new PoolingLayer();
	static Network network;

	/*************************************************************************************************/


	public static void main(String[] args) {
		String trainDirectory = "images/trainset/";
		String  tuneDirectory = "images/tuneset/";
		String  testDirectory = "images/testset/";

        if(args.length > 5) {
            System.err.println("Usage error: java Lab3 <train_set_folder_path> <tune_set_folder_path> <test_set_foler_path> <imageSize>");
            System.exit(1);
        }
        if (args.length >= 1) { trainDirectory = args[0]; }
        if (args.length >= 2) {  tuneDirectory = args[1]; }
        if (args.length >= 3) {  testDirectory = args[2]; }
        if (args.length >= 4) {  imageSize     = Integer.parseInt(args[3]); }

		// Here are statements with the absolute path to open images folder
        File trainsetDir = new File(trainDirectory);
        File tunesetDir  = new File( tuneDirectory);
        File testsetDir  = new File( testDirectory);

        // create three datasets
		Dataset trainset = new Dataset();
        Dataset  tuneset = new Dataset();
        Dataset  testset = new Dataset();

        // Load in images into datasets.
        long start = System.currentTimeMillis();
        loadDataset(trainset, trainsetDir);
        System.out.println("The trainset contains " + comma(trainset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

        start = System.currentTimeMillis();
        loadDataset(tuneset, tunesetDir);
        System.out.println("The  testset contains " + comma( tuneset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

        start = System.currentTimeMillis();
        loadDataset(testset, testsetDir);
        System.out.println("The  tuneset contains " + comma( testset.getSize()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");


		// Network structure.
		ArrayList<Integer> numUnits = new ArrayList<>();
		numUnits.add(numKernels * numKernels * size * size * 4);
		numUnits.add(20);
		numUnits.add(6);
		network = new Network(numUnits, eta, 1.0 - dropoutRate, weight_decay, momentum);
//		Collections.shuffle(trainset.instances);
		List<Instance> tmpList = trainset.getImages();
		Collections.shuffle(tmpList);
		trainset = new Dataset();
		for (Instance i : tmpList) trainset.add(i);

		// Now train a Deep ANN.  You might wish to first use your Lab 2 code here and see how one layer of HUs does.  Maybe even try your perceptron code.
        // We are providing code that converts images to feature vectors.  Feel free to discard or modify.
        start = System.currentTimeMillis();
        trainANN(trainset, tuneset, testset);
        System.out.println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");

    }

	public static void loadDataset(Dataset dataset, File dir) {
        for(File file : dir.listFiles()) {
            // check all files
             if(!file.isFile() || !file.getName().endsWith(".jpg")) {
                continue;
            }
            //String path = file.getAbsolutePath();
            BufferedImage img = null, scaledBI = null;
            try {
                // load in all images
                img = ImageIO.read(file);
                // every image's name is in such format:
                // label_image_XXXX(4 digits) though this code could handle more than 4 digits.
                String name = file.getName();
                int locationOfUnderscoreImage = name.indexOf("_image");

                // Resize the image if requested.  Any resizing allowed, but should really be one of 8x8, 16x16, 32x32, or 64x64 (original data is 128x128).
                if (imageSize != 128) {
                    scaledBI = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
                    Graphics2D g = scaledBI.createGraphics();
                    g.drawImage(img, 0, 0, imageSize, imageSize, null);
                    g.dispose();
                }

                Instance instance = new Instance(scaledBI == null ? img : scaledBI, "", name.substring(0, locationOfUnderscoreImage));

                dataset.add(instance);
            } catch (IOException e) {
                System.err.println("Error: cannot load in the image file");
                System.exit(1);
            }
        }
    }
	///////////////////////////////////////////////////////////////////////////////////////////////

	private static Category convertCategoryStringToEnum(String name) {
		if ("airplanes".equals(name))   return Category.airplanes; // Should have been the singular 'airplane' but we'll live with this minor error.
		if ("butterfly".equals(name))   return Category.butterfly;
		if ("flower".equals(name))      return Category.flower;
		if ("grand_piano".equals(name)) return Category.grand_piano;
		if ("starfish".equals(name))    return Category.starfish;
		if ("watch".equals(name))       return Category.watch;
		throw new Error("Unknown category: " + name);
	}

	private static double getRandomWeight(int fanin, int fanout) { // This is one 'rule of thumb' for initializing weights.  Fine for perceptrons and one-layer ANN at least.
		double range = Math.max(Double.MIN_VALUE, 4.0 / Math.sqrt(6.0 * (fanin + fanout)));
		return (2.0 * random() - 1.0) * range;
	}

	// Map from 2D coordinates (in pixels) to the 1D fixed-length feature vector.
	private static double get2DfeatureValue(Vector<Double> ex, int x, int y, int offset) { // If only using GREY, then offset = 0;  Else offset = 0 for RED, 1 for GREEN, 2 for BLUE, and 3 for GREY.
		return ex.get(unitsPerPixel * (y * imageSize + x) + offset); // Jude: I have not used this, so might need debugging.
	}

	///////////////////////////////////////////////////////////////////////////////////////////////


	// Return the count of TESTSET errors for the chosen model.
    private static int trainANN(Dataset trainset, Dataset tuneset, Dataset testset) {
    	Instance sampleImage = trainset.getImages().get(0); // Assume there is at least one train image!
    	inputVectorSize = sampleImage.getWidth() * sampleImage.getHeight() * unitsPerPixel + 1; // The '-1' for the bias is not explicitly added to all examples (instead code should implicitly handle it).  The final 1 is for the CATEGORY.

    	// For RGB, we use FOUR input units per pixel: red, green, blue, plus grey.  Otherwise we only use GREY scale.
    	// Pixel values are integers in [0,255], which we convert to a double in [0.0, 1.0].
    	// The last item in a feature vector is the CATEGORY, encoded as a double in 0 to the size on the Category enum.
    	// We do not explicitly store the '-1' that is used for the bias.  Instead code (to be written) will need to implicitly handle that extra feature.
    	System.out.println("\nThe input vector size is " + comma(inputVectorSize - 1) + ".\n");

    	Vector<Vector<Double>> trainFeatureVectors = new Vector<Vector<Double>>(trainset.getSize());
    	Vector<Vector<Double>>  tuneFeatureVectors = new Vector<Vector<Double>>( tuneset.getSize());
    	Vector<Vector<Double>>  testFeatureVectors = new Vector<Vector<Double>>( testset.getSize());

        long start = System.currentTimeMillis();
		fillFeatureVectors(trainFeatureVectors, trainset);
        System.out.println("Converted " + trainFeatureVectors.size() + " TRAIN examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

        start = System.currentTimeMillis();
        fillFeatureVectors( tuneFeatureVectors,  tuneset);
        System.out.println("Converted " +  tuneFeatureVectors.size() + " TUNE  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

        start = System.currentTimeMillis();
		fillFeatureVectors( testFeatureVectors,  testset);
        System.out.println("Converted " +  testFeatureVectors.size() + " TEST  examples to feature vectors. Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

        System.out.println("\nTime to start learning!");

        // Call your Deep ANN here.  We recommend you create a separate class file for that during testing and debugging, but before submitting your code cut-and-paste that code here.

        if      ("perceptrons".equals(modelToUse)) return trainPerceptrons(trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Either comment out this line or just right a 'dummy' function.
        else if ("oneLayer".equals(   modelToUse)) return trainOneHU(      trainFeatureVectors, tuneFeatureVectors, testFeatureVectors); // This is optional.  Ditto.
        else if ("deep".equals(       modelToUse)) return trainDeep(       trainFeatureVectors, tuneFeatureVectors, testFeatureVectors,
																			trainset, tuneset, testset);
        return -1;
	}

	private static void fillFeatureVectors(Vector<Vector<Double>> featureVectors, Dataset dataset) {
		for (Instance image : dataset.getImages()) {
			featureVectors.addElement(convertToFeatureVector(image));
		}
	}

	private static Vector<Double> convertToFeatureVector(Instance image) {
		Vector<Double> result = new Vector<Double>(inputVectorSize - 1);

		for (int index = 0; index < inputVectorSize - 1; index++) { // Need to subtract 1 since the last item is the CATEGORY.
			if (useRGB) {
				int xValue = (index / unitsPerPixel) / image.getWidth();
				int yValue = (index / unitsPerPixel) % image.getWidth();
			//	System.out.println("  xValue = " + xValue + " and yValue = " + yValue + " for index = " + index);
				if      (index % unitsPerPixel == 0) result.add(image.getRedChannel()  [xValue][yValue] / 255.0); // If unitsPerPixel > 4, this if-then-elseif needs to be edited!
				else if (index % unitsPerPixel == 1) result.add(image.getGreenChannel()[xValue][yValue] / 255.0);
				else if (index % unitsPerPixel == 2) result.add(image.getBlueChannel() [xValue][yValue] / 255.0);
				else                     result.add(image.getGrayImage()   [xValue][yValue] / 255.0); // Seems reasonable to also provide the GREY value.
			} else {
				int xValue = index % image.getWidth();
				int yValue = index / image.getWidth();
				result.add(                         image.getGrayImage()   [xValue][yValue] / 255.0);
			}
		}
//		result.add((double) convertCategoryStringToEnum(image.getLabel()).ordinal()); // The last item is the CATEGORY, representing as an integer starting at 0 (and that int is then coerced to double).

		return result;
	}

	////////////////////  Some utility methods (cut-and-pasted from JWS' Utils.java file). ///////////////////////////////////////////////////

	private static final long millisecInMinute = 60000;
	private static final long millisecInHour   = 60 * millisecInMinute;
	private static final long millisecInDay    = 24 * millisecInHour;
	public static String convertMillisecondsToTimeSpan(long millisec) {
		return convertMillisecondsToTimeSpan(millisec, 0);
	}
	public static String convertMillisecondsToTimeSpan(long millisec, int digits) {
		if (millisec ==    0) { return "0 seconds"; } // Handle these cases this way rather than saying "0 milliseconds."
		if (millisec <  1000) { return comma(millisec) + " milliseconds"; } // Or just comment out these two lines?
		if (millisec > millisecInDay)    { return comma(millisec / millisecInDay)    + " days and "    + convertMillisecondsToTimeSpan(millisec % millisecInDay,    digits); }
		if (millisec > millisecInHour)   { return comma(millisec / millisecInHour)   + " hours and "   + convertMillisecondsToTimeSpan(millisec % millisecInHour,   digits); }
		if (millisec > millisecInMinute) { return comma(millisec / millisecInMinute) + " minutes and " + convertMillisecondsToTimeSpan(millisec % millisecInMinute, digits); }

		return truncate(millisec / 1000.0, digits) + " seconds";
	}

    public static String comma(int value) { // Always use separators (e.g., "100,000").
    	return String.format("%,d", value);
    }
    public static String comma(long value) { // Always use separators (e.g., "100,000").
    	return String.format("%,d", value);
    }
    public static String comma(double value) { // Always use separators (e.g., "100,000").
    	return String.format("%,f", value);
    }
    public static String padLeft(String value, int width) {
    	String spec = "%" + width + "s";
    	return String.format(spec, value);
    }

    /**
     * Format the given floating point number by truncating it to the specified
     * number of decimal places.
     *
     * @param d
     *            A number.
     * @param decimals
     *            How many decimal places the number should have when displayed.
     * @return A string containing the given number formatted to the specified
     *         number of decimal places.
     */
    public static String truncate(double d, int decimals) {
    	double abs = Math.abs(d);
    	if (abs > 1e13)             {
    		return String.format("%."  + (decimals + 4) + "g", d);
    	} else if (abs > 0 && abs < Math.pow(10, -decimals))  {
    		return String.format("%."  +  decimals      + "g", d);
    	}
        return     String.format("%,." +  decimals      + "f", d);
    }

    /** Randomly permute vector in place.
     *
     * @param <T>  Type of vector to permute.
     * @param vector Vector to permute in place.
     */
    public static <T> void permute(Vector<T> vector) {
    	if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an unbiased permute; I prefer (1) assigning random number to each element, (2) sorting, (3) removing random numbers.
    		// But also see "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which justifies this.
    		/*	To shuffle an array a of n elements (indices 0..n-1):
 									for i from n - 1 downto 1 do
      								j <- random integer with 0 <= j <= i
      								exchange a[j] and a[i]
    		 */

    		for (int i = vector.size() - 1; i >= 1; i--) {  // Note from JWS (2/2/12): to match the above I reversed the FOR loop that Trevor wrote, though I don't think it matters.
    			int j = random0toNminus1(i + 1);
    			if (j != i) {
    				T swap =    vector.get(i);
    				vector.set(i, vector.get(j));
    				vector.set(j, swap);
    			}
    		}
    	}
    }

    public static Random randomInstance = new Random(638 * 838);  // Change the 638 * 838 to get a different sequence of random numbers.

    /**
     * @return The next random double.
     */
    public static double random() {
        return randomInstance.nextDouble();
    }

    /**
     * @param lower
     *            The lower end of the interval.
     * @param upper
     *            The upper end of the interval. It is not possible for the
     *            returned random number to equal this number.
     * @return Returns a random integer in the given interval [lower, upper).
     */
    public static int randomInInterval(int lower, int upper) {
    	return lower + (int) Math.floor(random() * (upper - lower));
    }


    /**
     * @param upper
     *            The upper bound on the interval.
     * @return A random number in the interval [0, upper).
     */
    public static int random0toNminus1(int upper) {
    	return randomInInterval(0, upper);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////  Write your own code below here.  Feel free to use or discard what is provided.

	private static int trainPerceptrons(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
		Vector<Vector<Double>> perceptrons = new Vector<Vector<Double>>(Category.values().length);  // One perceptron per category.

		for (int i = 0; i < Category.values().length; i++) {
			Vector<Double> perceptron = new Vector<Double>(inputVectorSize);  // Note: inputVectorSize includes the OUTPUT CATEGORY as the LAST element.  That element in the perceptron will be the BIAS.
			perceptrons.add(perceptron);
			for (int indexWgt = 0; indexWgt < inputVectorSize; indexWgt++) perceptron.add(getRandomWeight(inputVectorSize, 1)); // Initialize weights.
		}

		if (fractionOfTrainingToUse < 1.0) {  // Randomize list, then get the first N of them.
			int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
			Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.
			for (int i = 0; i <numberToKeep; i++) {
				trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
			}
			trainFeatureVectors = trainFeatureVectors_temp;
		}

        int trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;
        long  overallStart = System.currentTimeMillis(), start = overallStart;

		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

            // CODE NEEDED HERE!

	        System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
	        reportPerceptronConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
	        start = System.currentTimeMillis();
		}
    	System.out.println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch)
    						+ " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
    	return testSetErrorsAtBestTune;
	}

	private static void reportPerceptronConfig() {
		System.out.println(  "***** PERCEPTRON: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) + ", eta = " + truncate(eta, 2) + ", dropout rate = " + truncate(dropoutRate, 2)	);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////   ONE HIDDEN LAYER

	private static boolean debugOneLayer               = false;  // If set true, more things checked and/or printed (which does slow down the code).
	private static int    numberOfHiddenUnits          = 250;

	private static int trainOneHU(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
	    long overallStart   = System.currentTimeMillis(), start = overallStart;
        int  trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;

		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

            // CODE NEEDED HERE!

	        System.out.println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
	        reportOneLayerConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
	        start = System.currentTimeMillis();
		}

		System.out.println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch)
		                    + " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
    	return testSetErrorsAtBestTune;
	}

	private static void reportOneLayerConfig() {
		System.out.println(  "***** ONE-LAYER: UseRGB = " + useRGB + ", imageSize = " + imageSize + "x" + imageSize + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2)
		        + ", eta = " + truncate(eta, 2)   + ", dropout rate = "      + truncate(dropoutRate, 2) + ", number HUs = " + numberOfHiddenUnits
			//	+ ", activationFunctionForHUs = " + activationFunctionForHUs + ", activationFunctionForOutputs = " + activationFunctionForOutputs
			//	+ ", # forward props = " + comma(forwardPropCounter)
				);
	//	for (Category cat : Category.values()) {  // Report the output unit biases.
	//		int catIndex = cat.ordinal();
    //
	//		System.out.print("  bias(" + cat + ") = " + truncate(weightsToOutputUnits[numberOfHiddenUnits][catIndex], 6));
	//	}   System.out.println();
	}

	// private static long forwardPropCounter = 0;  // Count the number of forward propagations performed.


	////////////////////////////////////////////////////////////////////////////////////////////////  DEEP ANN Code


	private static int trainDeep(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors,	Vector<Vector<Double>> testFeatureVectors,
								 Dataset trainset, Dataset tuneset, Dataset testset) {
		Vector<Vector<Double>> trainLabels = getLabelVector(trainset);
		Vector<Vector<Double>> tuneLabels = getLabelVector(tuneset);
		Vector<Vector<Double>> testLabels = getLabelVector(testset);

		double best = 0.0;
		int times = 0;
		for (int k = 0; k < maxEpochs; k++) {

			for (int i = 0; i < trainFeatureVectors.size(); i++) {
				///************************************ Forward *****************************************/
				Vector<Double> feature = trainFeatureVectors.get(i);
				layerInput.outMaps = Layer.fillArray(feature, 1, imageSize);

				Convolution.forward(layerInput, layerC1);
				p1.forward(layerC1, layerP1);
				Convolution.forward(layerP1, layerC2);
				p2.forward(layerC2, layerP2);

				Vector<Double> tmp = Layer.outputArray(layerP2.outMaps, numKernels * numKernels, size);
				network.forwardPropagation(tmp);


				///************************************ Back *****************************************/
				Vector<Double> label = trainLabels.get(i);
				network.backPropagation(label);

				Vector<Double> errors = network.getErrorArray();
				layerP2.error = Layer.fillArray(errors, numKernels * numKernels, size);

				p2.back(layerC2, layerP2);
				Convolution.backprop(layerP1, layerC2, eta);
				p1.back(layerC1, layerP1);
				Convolution.backprop(layerInput, layerC1, eta);
			}


			// Early stopping.
			double trainAccuracy = testDeep(trainFeatureVectors, trainLabels, null, false);
			double tuneAccuracy = testDeep(tuneFeatureVectors, tuneLabels, null, false);
			double testAccuracy = testDeep(testFeatureVectors, testLabels, new String[]{ "airplanes", "butterfly", "flower", "grand_piano", "starfish", "watch" }, k % 20 == 0);

			if (true)  System.out.println(k + ", " + trainAccuracy + ", " + tuneAccuracy + ", " + testAccuracy);

			if (k < 300) continue;

			if (tuneAccuracy > best) {
				best = tuneAccuracy;
				times = 0;
			} else {
				times += 1;
				if (times > 200) break;
			}
		}

		System.out.println(testDeep(testFeatureVectors, testLabels, new String[]{ "airplanes", "butterfly", "flower", "grand_piano", "starfish", "watch" }, true));

		return -1;
	}

	private  static double testDeep(Vector<Vector<Double>> featureVectors, Vector<Vector<Double>> data_labels, String[] labels, boolean print) {

		double right = 0.0;

		int[][] confusion = new int[data_labels.get(0).size()][data_labels.get(0).size()];

		for (int i = 0; i < featureVectors.size(); i++) {
			Vector<Double> data = featureVectors.get(i);
			Vector<Double> data_label = data_labels.get(i);

			double tmp_p = network.dropout_p;
			network.dropout_p = 1.1;


			///************************************ Forward *****************************************/
			Vector<Double> feature = featureVectors.get(i);
			layerInput.outMaps = Layer.fillArray(feature, 1, imageSize);

			Convolution.forward(layerInput, layerC1);
			p1.forward(layerC1, layerP1);
			Convolution.forward(layerP1, layerC2);
			p2.forward(layerC2, layerP2);

			Vector<Double> tmp = Layer.outputArray(layerP2.outMaps, numKernels * numKernels, size);
			network.forwardPropagation(tmp);



			network.dropout_p = tmp_p;

			int correct_index = -1;
			for (int j = 0; j < data_label.size(); j++) {
				if (data_label.get(j) > 0.5) correct_index = j;
			}

			if (correct_index == network.outputIndex()) right += 1.0;

			confusion[correct_index][network.outputIndex()] += 1;
		}

		if(print) {
			for (int i = 0; i < labels.length; i++) {
				for (int j = 0; j < labels.length; j++) {
					System.out.print(confusion[i][j] + "\t");
				}
				System.out.println();
			}
		}

		return right / featureVectors.size();

	}

	private static Vector<Vector<Double>> getLabelVector(Dataset dataset) {
		Vector<Vector<Double>> dataLabels = new Vector<>();

		for (Instance img: dataset.getImages()) {
			Vector<Double> tmp = new Vector<>();
			for ( int i = 0; i < 6; i++) tmp.add(0.0);

			if ("airplanes".equals(img.getLabel()))   tmp.set(0, 1.); // Should have been the singular 'airplane' but we'll live with this minor error.
			else if ("butterfly".equals(img.getLabel()))   tmp.set(1, 1.);
			else if ("flower".equals(img.getLabel()))      tmp.set(2, 1.);
			else if ("grand_piano".equals(img.getLabel())) tmp.set(3, 1.);
			else if ("starfish".equals(img.getLabel()))    tmp.set(4, 1.);
			else if ("watch".equals(img.getLabel()))       tmp.set(5, 1.);

			dataLabels.add(tmp);
		}

		return dataLabels;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////


	public static class Network {
		protected static final boolean ReLU = false;

		public ArrayList<ArrayList<Perceptron>> perceptrons = new ArrayList<>();

		public double rate = 0.1;
		public double dropout_p = 1.1;
		public double weight_decay = 0.04;
		public double momentum = 0.1;

		private Random random = new Random();

		public Network(ArrayList<Integer> numUnits, double rate, double dropout_p, double weight_decay, double momentum) {

			// Init perceptrons.
			for (int i = 0; i < numUnits.size(); i++) {
				ArrayList<Perceptron> layer = new ArrayList<>();

				for (int j = 0; j < numUnits.get(i); j++) {
					layer.add(new Perceptron());
				}

				if (i != numUnits.size() - 1) layer.add(new Perceptron()); // bias unit.

				perceptrons.add(layer);
			}

			// Init weights.

			for (int i = 0; i < perceptrons.size() - 1; i++) {
				for (int j = 0; j < perceptrons.get(i).size(); j++) {
					for (int k = 0; k < perceptrons.get(i + 1).size(); k++) {
						if (i != perceptrons.size() - 2 && k == perceptrons.get(i + 1).size() - 1) continue;	// bias.

						Weight weight = new Weight(random.nextDouble());
						weight.input = perceptrons.get(i).get(j);
						weight.output = perceptrons.get(i + 1).get(k);
						perceptrons.get(i).get(j).outputs.add(weight);
						perceptrons.get(i + 1).get(k).inputs.add(weight);
					}
				}
			}

			// Learning rate.
			this.rate = rate;
			this.dropout_p = dropout_p;
			this.weight_decay = weight_decay;
			this.momentum = momentum;
		}


		public void train(Vector<Double> data, Vector<Double> label) {

			forwardPropagation(data);
			backPropagation(label);

		}

		public double test(Vector<Vector<Double>> datas, Vector<Vector<Double>> data_labels,
						   String[] labels, boolean print) {
			double right = 0.0;

			int[][] confusion = new int[data_labels.get(0).size()][data_labels.get(0).size()];

			for (int i = 0; i < datas.size(); i++) {
				Vector<Double> data = datas.get(i);
				Vector<Double> data_label = data_labels.get(i);

				double tmp = dropout_p;
				dropout_p = 1.1;
				forwardPropagation(data);
				dropout_p = tmp;

				int correct_index = -1;
				for (int j = 0; j < data_label.size(); j++) {
					if (data_label.get(j) > 0.5) correct_index = j;
				}

				if (correct_index == outputIndex()) right += 1.0;

				confusion[correct_index][outputIndex()] += 1;
			}

			if(print) {
				for (int i = 0; i < labels.length; i++) {
					for (int j = 0; j < labels.length; j++) {
						System.out.print(confusion[i][j] + "\t");
					}
					System.out.println();
				}
			}

			return right / datas.size();
		}

		public int outputIndex() {
			double max = 0.0;
			int index = 0;

			ArrayList<Perceptron> layer = perceptrons.get(perceptrons.size() - 1);

			for (int i = 0; i < layer.size(); i++) {
				if (layer.get(i).fx > max) {
					max = layer.get(i).fx;
					index = i;
				}
//            System.out.print(layer.get(i).fx + " ");
			}
//        System.out.println();
			return index;
		}

		private double sigmoid(double x) {
			return (1/( 1 + Math.pow(Math.E,(-1*x))));
		}

		public void forwardPropagation(Vector<Double> data) {
			// Input layer.
			for (int i = 0; i < data.size(); i++) {
				perceptrons.get(0).get(i).fx = data.get(i);
			}

			// Hidden layers and output layer.
			for (int i = 1; i < perceptrons.size(); i++) {
				for (Perceptron p : perceptrons.get(i)) {
					if (p.inputs.size() == 0) continue; // bias unit.

					// Dropout
					if (i != perceptrons.size() - 1 && random.nextDouble() > dropout_p) {
						// Drop
						p.dropout = true;
						continue;
					} else {
						// Keep
						p.dropout = false;

						p.fx = 0;

						for (Weight weight : p.inputs) {
							// If the input node is not dropped.
							if (!weight.input.dropout) p.fx += weight.input.fx * weight.value;
						}

						// ReLU
						/*************************************/
						if (i == 1 && ReLU) {
							p.fx = p.fx > 0 ? p.fx : 0;
							continue;
						}
						/*************************************/

						p.fx = sigmoid(p.fx);
					}
				}
			}
		}

		public void backPropagation(Vector<Double> label) {
			// Delta
			for (int i = 0; i < label.size(); i++) {	// Output layer.
				Perceptron p = perceptrons.get(perceptrons.size() - 1).get(i);
				p.delta = - (label.get(i) - p.fx) * p.fx * (1 - p.fx);
			}

			// Delta hidden layer.
			for (int i = perceptrons.size() - 2; i >= 0; i--) {
				for (Perceptron p : perceptrons.get(i)) {
					if (p.dropout) continue;

					p.delta = 0;

					for (Weight w : p.outputs) {
						if (!w.output.dropout) p.delta += w.output.delta * w.value;
					}

					// ReLu
					/************************************/
					if (i == 1 && ReLU) {
						p.delta *= p.fx > 0 ? 1 : 0;
						continue;
					}
					/************************************/

					p.delta *= p.fx * ( 1- p.fx );
				}
			}

			// Upadte weights.
			for (int i = 0; i < perceptrons.size() - 1; i++) {
				for (Perceptron p : perceptrons.get(i)) {
					if (p.dropout) continue;

					for (Weight w : p.outputs) {
						if (!w.output.dropout) {
							w.delta = - rate * p.fx * w.output.delta - rate * weight_decay * w.value + momentum * w.delta;
							w.value += w.delta;
						}

					}
				}
			}
		}


		public Vector<Double> getErrorArray() {
			Vector<Double> result = new Vector<>();
			for (Perceptron p : this.perceptrons.get(0)) {
				result.add(p.delta);
			}
			return result;
		}
	}


	public static class PoolingLayer {

		public int[][] map;

		public void forward(Layer layer1, Layer layer2) {
			map = new int[layer2.outputSize.x][layer2.outputSize.y];

			for (int num  = 0; num < layer1.outMapNum; num ++) {
				for (int c = 0; c < 4; c++) {
					for (int x = 0; x < layer2.outputSize.x; x ++) {
						for (int y = 0; y < layer2.outputSize.y; y ++) {
							int xx = x * 2;
							int yy = y * 2;

							double m = layer1.outMaps[num][c][xx][yy];
							map[x][y] = 0;

							if (layer1.outMaps[num][c][xx + 1][yy] > m) {
								m = layer1.outMaps[num][c][xx + 1][yy];
								map[x][y] = 1;
							}

							if (layer1.outMaps[num][c][xx][yy + 1] > m) {
								m = layer1.outMaps[num][c][xx][yy + 1];
								map[x][y] = 2;
							}

							if (layer1.outMaps[num][c][xx + 1][yy + 1] > m) {
								m = layer1.outMaps[num][c][xx + 1][yy + 1];
								map[x][y] = 3;
							}

							layer2.outMaps[num][c][x][y] = m;
						}
					}
				}
			}
		}

		public void back(Layer layer1, Layer layer2) {
			for (int num  = 0; num < layer1.outMapNum; num ++) {
				for (int c = 0; c < 4; c++) {
					for (int x = 0; x < layer2.outputSize.x; x ++) {
						for (int y = 0; y < layer2.outputSize.y; y ++) {
							int xx = x * 2;
							int yy = y * 2;

							layer1.error[num][c][xx][yy] = 0.0;
							layer1.error[num][c][xx + 1][yy] = 0.0;
							layer1.error[num][c][xx][yy + 1] = 0.0;
							layer1.error[num][c][xx + 1][yy + 1] = 0.0;

							if (map[x][y] == 0) layer1.error[num][c][xx][yy] = layer1.error[num][c][x][y];
							else if (map[x][y] == 1) layer1.error[num][c][xx + 1][yy] = layer1.error[num][c][x][y];
							else if (map[x][y] == 2) layer1.error[num][c][xx][yy + 1] = layer1.error[num][c][x][y];
							else if (map[x][y] == 3) layer1.error[num][c][xx + 1][yy + 1] = layer1.error[num][c][x][y];
						}
					}
				}
			}


		}




    /*
    public final static int numCores = 4;
    public static int imgLength;

    public ArrayList<Double> input = new ArrayList<>();
    public ArrayList<Double> output = new ArrayList<>();

    public ArrayList<Double> delta = new ArrayList<>();
    public ArrayList<Double> delta2 = new ArrayList<>();

    public PoolingLayer(ArrayList<Double> input, ArrayList<Double> output) {
        this.input = input;
        this.output = output;

        imgLength = (int) Math.sqrt(input.size() / numCores);
    }

    public void forwardPropagation() {
        for (int core = 0; core < numCores; core++) {
            int start = (input.size() / numCores) * core;

            for (int x = 0; x < imgLength; x += 2) {
                for (int y = 0; y < imgLength; y += 2) {
                    int index1 = x       * imgLength + y;
                    int index2 = x       * imgLength + y + 1;
                    int index3 = (x + 1) * imgLength + y;
                    int index4 = (x + 1) * imgLength + y + 1;

                    double d = Math.max(Math.max(input.get(start + index1), input.get(start + index2)),
                            Math.max(input.get(start + index3), input.get(start + index4)));

                    output.add(d);
                }
            }
        }
    }

    public void backPropagation() {
        for (int i = 0; i < input.size(); i ++) delta.add(0.);

        for (int core = 0; core < numCores; core++) {
            int start = (input.size() / numCores) * core;

            int start2 = (delta2.size() / numCores) * core;

            for (int xx = 0; xx < imgLength / 2; xx += 1) {
                for (int yy = 0; yy < imgLength / 2; yy += 1) {
                    int index_2 = xx * imgLength / 2 + yy;

                    int x = xx * 2;
                    int y = yy * 2;

                    int index1 = x       * imgLength + y;
                    int index2 = x       * imgLength + y + 1;
                    int index3 = (x + 1) * imgLength + y;
                    int index4 = (x + 1) * imgLength + y + 1;

                    for (int index: new int[]{index1, index2, index3, index4}) {
                        if (input.get(start + index).equals(output.get(start2 + index_2))) {
                            delta.set(start + index, delta2.get(start2 + index_2));
                            break;
                        }
                    }

                }
            }
        }

    }
    */

	}




	public static class Convolution {

/*
	public void createNextLayer(Layer preLayer ){
		Layer currLayer =  new Layer();
		currLayer.kernelSize = new Layer.Size(5,5);
		currLayer.initKernel();

		currLayer.outputSize = new Layer.Size(preLayer.outputSize.x - currLayer.kernelSize.x + 1,
				preLayer.outputSize.y - currLayer.kernelSize.y + 1);

		currLayer.outMapNum = preLayer.outMapNum * currLayer.kernelNum;

	}
	*/

		public static void forward(Layer prevLayer, Layer currLayer){ // How to update bias?

			int numKernel = currLayer.kernelNum;
			int outputX = currLayer.outputSize.x;
			int outputY = currLayer.outputSize.y;
			for (int i = 0;i < currLayer.outMapNum; i++){
				for (int j = 0; j < currLayer.channelNum; j++){

					currLayer.outMaps[i][j] = convolve(prevLayer.outMaps[i/numKernel][j],
							currLayer.kernel[i % numKernel][j])  ;
					//True means take ReLu as the activation function.
					matrixAdd(currLayer.outMaps[i][j],
							createConstantMatrix(currLayer.bias[i % numKernel][j], outputX, outputY), true);
				}
			}

		}
		//public double[][][][] error;   // outMapNum * channelNum *  outputSize.x * outputSize.y
		//public double[][][][] kernel;  // kernelNum * channelNum * kernelSize * kernelSize
		//public double[][] bias; // kernelNum * channelNum;
		public static void backprop(Layer prevLayer, Layer currLayer, double stepSize){

			int numKernel = currLayer.kernelNum;
			int kernelx = currLayer.kernelSize.x;
			int kernely = currLayer.kernelSize.y;

			double[][][][] dKernel = new double[numKernel][currLayer.channelNum][kernelx][kernely]; //store gradient
			double[][] dBias = createConstantMatrix(0.0, numKernel, currLayer.channelNum);

			for (int i = 0; i < numKernel; i++)
				for (int j =0; j < currLayer.channelNum; j++){
					dKernel[i][j] = createConstantMatrix(0.0, kernelx, kernely);
				}

			// update error[][][][] of prevLayer
			for (int i = 0 ; i < currLayer.outMapNum; i++)
				for (int j = 0; j < currLayer.channelNum; j++){
					if (i % numKernel == 0)
						prevLayer.setErrorZero(i/numKernel, j);



					matrixAdd(prevLayer.error[i/numKernel][j] ,
							convolve( zeroPadding(currLayer.error[i][j], kernelx, kernely),
									reverseKernel(currLayer.kernel[i % numKernel][j])),false
					);
					// two ways to calculate, may need debug
					matrixAdd(dKernel[i % numKernel][j] ,
							convolve (prevLayer.outMaps[i/numKernel][j] , currLayer.error[i][j]),
							false);

					dBias[i % numKernel][j] += avgMatrix(currLayer.error[i][j]);

				}
			// update kernel[][][][] and bias of currLayer
			for (int i = 0; i < numKernel; i++)
				for (int j = 0; j < currLayer.channelNum; j++){
					constantTimeMatrix(-stepSize, dKernel[i][j]);
					matrixAdd(currLayer.kernel[i][j],dKernel[i][j] , false);

					// "-" if error = currentLabel - trueLabel;
					currLayer.bias[i][j] -=  dBias[i][j]/ prevLayer.outMapNum;
				}
		}


		private static double[][] convolve (double[][]image, double[][]kernel ) // convolution
		{
			int iRow = image.length;
			int iColumn = image[0].length;
			int kRow = kernel.length;
			int kColumn = kernel[0].length;

			double[][] result = new double[iRow - kRow + 1 ][ iColumn - kColumn + 1];
			for (int i = 0; i < iRow - kRow + 1; i++)
				for (int j = 0; j < iColumn - kColumn + 1; j++)
				{
					double tmp = 0;
					for (int x = i; x < i + kRow; x++)
						for (int y = j; y < j + kColumn; y++)
						{
							tmp += image[x][y] * kernel[x - i][y - j];

						}
					result[i][j] = tmp;

				}
			return result;
		}

		// rotate 180
		private static double[][] reverseKernel(double[][] kernel){
			int row = kernel.length;
			int col = kernel[0].length;
			double[][] reverse = new double[row][col];
			for (int i = 0; i < row; i++)
				for (int j = 0; j< col; j++){
					reverse[i][j] = kernel[row - i - 1][col - j - 1];
				}
			return reverse;
		}

		// zero padding.
		private static double[][] zeroPadding (double[][] matrix, int kernelx, int kernely){
			int row = matrix.length;
			int col = matrix[0].length;
			int newRow = row + 2 * kernelx - 2;
			int newCol = col + 2 * kernely -2;
			double[][] padding = new double[newRow ][newCol];
			for (int i = 0; i < newRow; i++)
				for (int j = 0; j < newCol;j++){
					if(i > kernelx -2 && j > kernely - 2 && i < newRow  - kernelx + 1 && j< newCol -kernely+1)
						padding[i][j] = matrix[i - kernelx + 1][j - kernely + 1];
					else
						padding[i][j] = 0;
				}
			return padding;
		}

		// matrix addition
		private static void matrixAdd( double[][] m1, double [][] m2, boolean reLu){
			int row = m1.length;
			int col = m1[0].length;

			for (int i = 0 ; i < row; i++)
				for (int j = 0; j < col; j++)
				{
					m1[i][j] += m2[i][j];
					if (reLu){
						m1[i][j] = reLu(m1[i][j]);
					}

				}

		}

		private static double[][] createConstantMatrix(double bias, int row, int col)
		{
			double[][] biasM = new double[row][col];
			for (int i = 0; i < row ;i ++)
				for (int j = 0; j< col; j++){
					biasM[i][j] = bias;
				}
			return biasM;
		}

		private static void constantTimeMatrix(double number, double[][] matrix)
		{
			int row = matrix.length;
			int col = matrix[0].length;

			for (int i = 0 ; i < row; i++)
				for (int j = 0; j < col; j++)
					matrix[i][j] *= number;
		}

		private static double avgMatrix(double[][] matrix){
			double sum = 0.0;
			for (int i = 0; i < matrix.length; i++)
				for (int j = 0; j < matrix[0].length; j++)
					sum += matrix[i][j];
			return sum/(matrix.length * matrix[0].length) ;
		}



		private static double sigmoid(double x) {
			return (1/( 1 + Math.pow(Math.E,(-1*x))));
		}
		private static double reLu(double x){
			return x > 0 ? x: -x;
		}



		public static class LayerBuilder {
			private List<Layer> mLayers;

			public LayerBuilder() {
				mLayers = new ArrayList<Layer>();
			}

			public LayerBuilder(Layer layer) {
				this();
				mLayers.add(layer);
			}

			public LayerBuilder addLayer(Layer layer) {
				mLayers.add(layer);
				return this;
			}

		}


	}



	public static class Layer {

		public String type; //type of layer
		public double[][][][] kernel;  // kernelNum * channelnum * kernelSize * kernelSize
		public double[][] bias; // kernelNum * channelNum;

		public Size kernelSize;
		public int kernelNum  = 5;


		public Size outputSize;
		public int outMapNum;

		public int channelNum = 4;

		public double[][][][] outMaps; // outMapNum * channelNum *  outputSize.x * outputSize.y
		public double[][][][] error;   // outMapNum * channelNum *  outputSize.x * outputSize.y

		public Layer(){

		}

		//for the input layer, each image get a outMaps of 1 * 4 * imageHeight * imageWidth, here mapSize is the image Size;
		public static Layer buildInputLayer(Size mapSize) {
			Layer layer = new Layer();
			layer.type = "input";
			layer.outMapNum = 1;//
			layer.outputSize = mapSize;//
			layer.initError();
			return layer;
		}

		//  currLayer.outMapNum = preLayer.outMapNum * currLayer.kernelNum;
		//  currLayer.outputSize = new Layer.Size(preLayer.outputSize.x - currLayer.kernelSize.x + 1, preLayer.outputSize.y - currLayer.kernelSize.y + 1);
		public static Layer buildConvLayer(int outMapNum, int kernelNum, Size kernelSize, Size mapSize) {
			Layer layer = new Layer();
			layer.type = "conv";
			layer.outMapNum = outMapNum;
			layer.kernelSize = kernelSize;
			layer.outputSize = mapSize;
			layer.kernelNum = kernelNum;
			layer.initOutMaps();
			layer.initBias();
			layer.initError();
			layer.initKernel();
			return layer;
		}

		public static Layer buildPoolingLayer(int outMapNum, Size mapSize) {
			Layer layer = new Layer();
			layer.outMapNum = outMapNum;
			layer.outputSize = mapSize;
			layer.initOutMaps();
			layer.initError();
			return layer;
		}

		public void initOutMaps() {
			this.outMaps = new double[outMapNum][channelNum][outputSize.x][outputSize.y];

		}

		public  void initError(){
			this.error = new double[outMapNum][channelNum][outputSize.x][outputSize.y];
		}

		public void setErrorZero(int i, int j){
			for (int x = 0; x < outputSize.x; x++ )
				for (int y = 0; y < outputSize.y; y++)
					this.error[i][j][x][y] = 0;
		}

		public void initKernel() {
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
			this.kernel = new double[kernelNum][channelNum][kernelSize.x][kernelSize.y];
			for (int i = 0; i < kernelNum; i++)
				for (int j = 0; j < channelNum; j++)
					this.kernel[i][j] = randomMatrix(kernelSize.x, kernelSize.y);
		}
		public void initBias(){
			this.bias = randomMatrix(kernelNum, channelNum);
		}


		public double[][] getKernel(int i, int j) {
			return kernel[i][j];
		}



		public static double[][] randomMatrix(int x, int y) {
			double[][] matrix = new double[x][y];
			Random r = new Random();
			for (int i = 0; i < x; i++) {
				for (int j = 0; j < y; j++) {

					matrix[i][j] = (r.nextDouble() - 0.5) / 10;
				}
			}
			return matrix;
		}

		public static class Size{
			public  int x;
			public  int y;
			public Size(int x, int y)
			{
				this.x = x;
				this.y = y;
			}
		}

		public static double[][][][] fillArray(Vector<Double> data, int mapNum, int length) {
			double[][][][] output = new double[mapNum][4][length][length];

			for (int num = 0; num < mapNum; num ++) {
				int mapStart = num * (4 * length * length);
				for ( int x = 0; x < length; x ++) {
					for ( int y = 0; y < length; y ++) {
						for (int c = 0; c < 4; c++) {
							output[num][c][x][y] = data.get(mapStart + 4 * (x * length + y) + c);
						}
					}
				}
			}

			return output;
		}

		public static Vector<Double> outputArray(double[][][][] map, int mapNum, int length) {
			int size = mapNum * 4 * length * length;

			Vector<Double> result = new Vector<>(size);

			for (int index = 0; index < size; index++) { // Need to subtract 1 since the last item is the CATEGORY.
				int num = index / (4 * length * length);
				int offset = index % (4 * length * length);
				int x = (offset / 4) / length;
				int y = (offset / 4) % length;
				int c = offset % 4;

				result.add(map[num][c][x][y]);
			}

			return result;
		}

	}



	public static class Perceptron {
		public ArrayList<Weight> inputs = new ArrayList<>();
		public ArrayList<Weight> outputs = new ArrayList<>();

		public double fx = 1;
		public double delta = 0;
		public boolean dropout = false;

		public Perceptron() {

		}

	}

	public static class Weight {
		public Perceptron input;
		public Perceptron output;

		public double value;
		public double delta = 0.0;

		public Weight(double value) {
			this.value = value;
		}


	}

}

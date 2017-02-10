/**
 * Created by zhicheng on 1/22/17.
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Lab2 {

    protected static final boolean debugFlag = true;

    public static void main(String[] args) throws FileNotFoundException {

        // Read file.
        DataReader dataReader1 = new DataReader();
        DataReader dataReader2 = new DataReader();
        DataReader dataReader3 = new DataReader();
        if (debugFlag) {
            dataReader1.read("red-wine-quality-train.data");
//        dataReader1.read("Thoracic_Surgery_Data_train.data");

            dataReader2.read("red-wine-quality-tune.data");
//        dataReader2.read("Thoracic_Surgery_Data_tune.data");

            dataReader3.read("red-wine-quality-test.data");
//        dataReader3.read("Thoracic_Surgery_Data_test.data");
        } else {
            dataReader1.read(args[0]);
            dataReader2.read(args[1]);
            dataReader3.read(args[2]);
        }

        // Network structure.
        ArrayList<Integer> numUnits = new ArrayList<>();
        numUnits.add(dataReader1.numFeatures);
        numUnits.add(10);
        numUnits.add(1);
        Network network = new Network(numUnits, 0.5);

        // Training process.
        double earlyStopAccuracy = 0.0;
        int earlyStopCount = 0;

        for (int k = 0; k < 100; k++) {
            for (int i = 0; i < dataReader1.data.size(); i++) {
                ArrayList<Double> tmp = new ArrayList<>();
                tmp.add(dataReader1.data_label.get(i));

                network.train(dataReader1.data.get(i), tmp);
            }

            double tuneAccuracy = network.test(dataReader2.data, dataReader2.data_label, null, false);

            // Early stopping.
            if (tuneAccuracy> earlyStopAccuracy) {
                earlyStopAccuracy = tuneAccuracy;
                earlyStopCount = 0;
            } else {
                earlyStopCount += 1;
                if (earlyStopCount > 10) break;
            }
        }

        // Print out 1: The label is print out in the test method.
        double testAccuracy = network.test(dataReader3.data, dataReader3.data_label, dataReader3.labels, true);
        // Print out 2: Overall accuracy.
        System.out.println("\nTest Accuracy: " + testAccuracy);
    }

}

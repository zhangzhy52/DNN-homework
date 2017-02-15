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
        DataReader dataReader = new DataReader();
        DataReader dataReader2 = new DataReader();

        if (debugFlag) {
            dataReader.read("protein-secondary-structure.train");
            dataReader2.read("protein-secondary-structure.train");
        } else {
            dataReader.read(args[0]);
        }

        // Network structure.
        ArrayList<Integer> numUnits = new ArrayList<>();
        numUnits.add(dataReader.data.get(0).size());
        numUnits.add(10);
        numUnits.add(3);
        Network network = new Network(numUnits, 0.05, 0.5, 0.1, 0.);

        // Training process.
//        double earlyStopAccuracy = 0.0;
//        int earlyStopCount = 0;

        for (int k = 0; k < 10; k++) {
            for (int i = 0; i < dataReader.data.size(); i++)
                network.train(dataReader.data.get(i), dataReader.data_label.get(i));

//            double tuneAccuracy = network.test(dataReader.data, dataReader.data_label);
//
//            // Early stopping.
//            if (tuneAccuracy> earlyStopAccuracy) {
//                earlyStopAccuracy = tuneAccuracy;
//                earlyStopCount = 0;
//            } else {
//                earlyStopCount += 1;
//                if (earlyStopCount > 100) break;
//            }
        }

        double testAccuracy = network.test(dataReader.data, dataReader.data_label);
        System.out.println("\nTest Accuracy: " + testAccuracy);
    }

}

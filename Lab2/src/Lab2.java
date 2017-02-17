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

    protected static final int num_units = 10;
    protected static final double rate = 0.5;
    protected static final double dropout_p = 1.0;
    protected static final double weight_decay = 0.;
    protected static final double momentum = 0.;

    public static void main(String[] args) throws FileNotFoundException {

        long startTime = System.currentTimeMillis();
        System.out.println("Units: " + num_units + ", rate: " + rate + ", dropout: " + dropout_p
                + ", weight_decay: " + weight_decay + ", momentum: " + momentum);

        // Read file.
        DataReader dataReader = new DataReader();

        if (debugFlag) {
            dataReader.read("protein-secondary-structure.all");
        } else {
            dataReader.read(args[0]);
        }

        // Network structure.
        ArrayList<Integer> numUnits = new ArrayList<>();
        numUnits.add(dataReader.trainData.get(0).size());
        numUnits.add(num_units);
        numUnits.add(3);
        Network network = new Network(numUnits, rate, dropout_p, weight_decay, momentum);

        // Training process.
//        double earlyStopAccuracy = 0.0;
//        int earlyStopCount = 0;
        double best = 0.0;

        for (int k = 0; k < 10000; k++) {
            if (k == 300) network.rate = 0.1;


            for (int i = 0; i < dataReader.trainData.size(); i++)
                network.train(dataReader.trainData.get(i), dataReader.train_data_label.get(i));

            // Early stopping.
            double trainAccuracy = network.test(dataReader.trainData, dataReader.train_data_label);
            double tuneAccuracy = network.test(dataReader.tuneData, dataReader.tune_data_label);
            double testAccuracy = network.test(dataReader.testData, dataReader.test_data_label);
            best = Math.max(testAccuracy, best);
            System.out.println(k + ", " + trainAccuracy + ", " + tuneAccuracy + ", " + testAccuracy + ", " + best);
//            if (tuneAccuracy> earlyStopAccuracy) {
//                earlyStopAccuracy = tuneAccuracy;
//                earlyStopCount = 0;
//            } else {
//                earlyStopCount += 1;
//                if (earlyStopCount > 300) break;
//            }
        }

//        double testAccuracy = network.test(dataReader.testData, dataReader.test_data_label);
//        System.out.println("\nTest Accuracy: " + testAccuracy);

        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println(totalTime / 1000);

    }

}

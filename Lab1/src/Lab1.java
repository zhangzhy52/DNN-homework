/**
 * Created by zhicheng on 1/22/17.
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Lab1 {

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

    public static class Network {

        public ArrayList<ArrayList<Perceptron>> perceptrons = new ArrayList<>();

        public double rate;

        public Network(ArrayList<Integer> numUnits, double rate) {

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
            Random random = new Random();
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
        }


        public void train(ArrayList<Double> data, ArrayList<Double> label) {

            forwardPropagation(data);
            backPropagation(label);

        }

        public double test(ArrayList<ArrayList<Double>> datas, ArrayList<Double> data_labels,
                           ArrayList<String> labels, boolean print) {

            double right = 0.0;

            for (int i = 0; i < datas.size(); i++) {
                ArrayList<Double> data = datas.get(i);
                double data_label = data_labels.get(i);

                forwardPropagation(data);

                int predict = perceptrons.get(perceptrons.size() - 1).get(0).fx < 0.5 ? 0 : 1;
                if (Math.abs(predict - data_label) < 0.5) {
                    right += 1;
                }

                // Print the label.
                if (print) {
                    System.out.println(labels.get(predict));
                }
            }


            return right / datas.size();
        }

        private double sigmoid(double x) {
            return (1/( 1 + Math.pow(Math.E,(-1*x))));
        }

        private void forwardPropagation(ArrayList<Double> data) {
            // Input layer.
            for (int i = 0; i < data.size(); i++) {
                perceptrons.get(0).get(i).fx = data.get(i);
            }

            // Hidden layers and output layer.
            for (int i = 1; i < perceptrons.size(); i++) {
                for (Perceptron p : perceptrons.get(i)) {
                    if (p.inputs.size() == 0) continue; // bias unit.

                    p.fx = 0;

                    for (Weight weight : p.inputs) {
                        p.fx += weight.input.fx * weight.value;
                    }

                    p.fx = sigmoid(p.fx);
                }
            }
        }

        private void backPropagation(ArrayList<Double> label) {
            // Delta
            for (int i = 0; i < label.size(); i++) {	// Output layer.
                Perceptron p = perceptrons.get(perceptrons.size() - 1).get(i);
                p.delta = - (label.get(i) - p.fx) * p.fx * (1 - p.fx);
            }

            // Delta hidden layer.
            for (int i = perceptrons.size() - 2; i >= 0; i--) {
                for (Perceptron p : perceptrons.get(i)) {
                    p.delta = 0;

                    for (Weight w : p.outputs) {
                        p.delta += w.output.delta * w.value;
                    }

                    p.delta *= p.fx * ( 1- p.fx );
                }
            }

            // Upadte weights.
            for (int i = 0; i < perceptrons.size() - 1; i++) {
                for (Perceptron p : perceptrons.get(i)) {
                    for (Weight w : p.outputs) {
                        w.value -= rate * p.fx * w.output.delta;
                    }
                }
            }
        }
    }

    public static class DataReader {
        private Scanner scanner;

        public int numFeatures = -1;
        public int numExamples = -1;
        public ArrayList<ArrayList<String>> features;
        public ArrayList<String> labels;
        public ArrayList<ArrayList<String>> examples;

        public ArrayList<ArrayList<Double>> data;
        public ArrayList<Double> data_label;

        DataReader() {
        }

        public void read(String fileName) throws FileNotFoundException {
            init();
            scanner = new Scanner(new File(fileName));

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim().replaceAll(" +", " ");

                if (!isUseful(line)) continue;

                if (numFeatures <= 0) {
                    numFeatures = Integer.parseInt(line);
                } else if (features.size() < numFeatures) {
                    ArrayList<String> tmp = new ArrayList<>();
                    String[] strings = line.split(" ");

                    tmp.add(strings[strings.length - 2]);
                    tmp.add(strings[strings.length - 1]);

                    features.add(tmp);
                } else if (labels.size() < 2) {
                    labels.add(line);
                } else if (numExamples <=0 ) {
                    numExamples = Integer.parseInt(line);
                } else if (examples.size() < numExamples){
                    ArrayList<String> tmp = new ArrayList<>();
                    String[] strings = line.split(" ");

                    tmp.add(strings[1]);
                    for (int i = 2; i < 2 + numFeatures; i++) {
                        tmp.add(strings[i]);
                    }

                    examples.add(tmp);
                }
            }

            for (ArrayList<String> example : examples) {
                data_label.add(example.get(0).equals(labels.get(0)) ? 0.0 : 1.0);

                ArrayList<Double> tmp = new ArrayList<>();
                for (int i = 0; i < numFeatures; i++) {
                    tmp.add(example.get(i + 1).equals(features.get(i).get(0)) ? 0.0 : 1.0);
                }
                data.add(tmp);
            }

        }

        private void init() {
            numFeatures = -1;
            numExamples = -1;
            features = new ArrayList<>();
            labels = new ArrayList<>();
            examples = new ArrayList<>();
            data = new ArrayList<>();
            data_label = new ArrayList<>();
        }

        private boolean isUseful(String line) {
            if (line.isEmpty()) return false;
            if (line.startsWith("//")) return false;

            return true;
        }

    }

    public static class Perceptron {
        public ArrayList<Weight> inputs = new ArrayList<>();
        public ArrayList<Weight> outputs = new ArrayList<>();

        public double fx = 1;
        public double delta = 0;

        public Perceptron() {

        }
    }

    public static class Weight {
        public Perceptron input;
        public Perceptron output;

        public double value;

        public Weight(double value) {
            this.value = value;
        }
    }

}

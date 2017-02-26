/**
 * Created by zhicheng on 1/22/17.
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;

public class Lab2 {

    protected static final boolean debugFlag = false;

    protected static final int num_units = 10;
    protected static final double rate = 0.5;
    protected static final double dropout_p = 0.5;
    protected static final double weight_decay = 0.;
    protected static final double momentum = 0.1;

    public static void main(String[] args) throws FileNotFoundException {

        long startTime = System.currentTimeMillis();
        if (debugFlag)
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
        double earlyStopAccuracy = 0.0;
        int earlyStopCount = 0;
        double best = 0.0;

        for (int k = 0; k < 1000; k++) {
            if (k == 200) network.rate = 0.1;

            for (int i = 0; i < dataReader.trainData.size(); i++)
                network.train(dataReader.trainData.get(i), dataReader.train_data_label.get(i));

            // Early stopping.
            double trainAccuracy = network.test(dataReader.trainData, dataReader.train_data_label, null, false);
            double tuneAccuracy = network.test(dataReader.tuneData, dataReader.tune_data_label, null, false);
            double testAccuracy = network.test(dataReader.testData, dataReader.test_data_label, null, false);
            best = Math.max(testAccuracy, best);

            if (debugFlag)  System.out.println(k + ", " + trainAccuracy + ", " + tuneAccuracy + ", " + testAccuracy + ", " + best);

            if (tuneAccuracy> earlyStopAccuracy) {
                earlyStopAccuracy = tuneAccuracy;
                earlyStopCount = 0;
            } else {
                earlyStopCount += 1;
                if (earlyStopCount > 200) break;
            }
        }

        double testAccuracy = network.test(dataReader.testData, dataReader.test_data_label, dataReader.labels, true);
        System.out.println("\nTest Accuracy: " + testAccuracy);

        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        if(debugFlag) System.out.println(totalTime / 1000);

    }

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


        public void train(ArrayList<Double> data, ArrayList<Double> label) {

            forwardPropagation(data);
            backPropagation(label);

        }

        public double test(ArrayList<ArrayList<Double>> datas, ArrayList<ArrayList<Double>> data_labels,
                           ArrayList<String> labels, boolean print) {
            double right = 0.0;

            for (int i = 0; i < datas.size(); i++) {
                ArrayList<Double> data = datas.get(i);
                ArrayList<Double> data_label = data_labels.get(i);

                double tmp = dropout_p;
                dropout_p = 1.1;
                forwardPropagation(data);
                dropout_p = tmp;

                if (data_label.get(outputIndex()) > 0.9) right += 1;

                if (print) System.out.println(labels.get(outputIndex()));
            }

            return right / datas.size();
        }

        private int outputIndex() {
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

        private void forwardPropagation(ArrayList<Double> data) {
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

        private void backPropagation(ArrayList<Double> label) {
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
    }

    public static class DataReader {
        private Scanner scanner;

        public int numFeatures = 20;
        public int numLabels = 0;
        //public int numExamples = -1;
        public int numProteins = 0;
        public int winSize = 17;

        public ArrayList<String> features;
        public ArrayList<String> labels;
        public ArrayList<ArrayList<String[]>> trainExamples;
        public ArrayList<ArrayList<String[]>> tuneExamples;
        public ArrayList<ArrayList<String[]>> testExamples;

        public ArrayList<ArrayList<Double>> trainData;
        public ArrayList<ArrayList<Double>> train_data_label;
        public ArrayList<ArrayList<Double>> tuneData;
        public ArrayList<ArrayList<Double>> tune_data_label;
        public ArrayList<ArrayList<Double>> testData;
        public ArrayList<ArrayList<Double>> test_data_label;

        DataReader() {
        }

        public void read(String fileName) throws FileNotFoundException {
            init();
            scanner = new Scanner(new File(fileName));
            ArrayList<String[]> tmp = new ArrayList<>();

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim().replaceAll(" +", " ");
                if (!isUseful(line)) continue;
                String[] strings = line.split("\\s+");



                if (line.startsWith("<>"))
                {
                    numProteins++;
                    if (!tmp.isEmpty())
                    {
                        if (numProteins %5==0)
                            tuneExamples.add(tmp);
                        else if (numProteins %5 ==1) {
                            testExamples.add(tmp);
                        }else {
                            trainExamples.add(tmp);
                        }
                    }
                    tmp = new ArrayList<>();
                    continue;
                }
                if (! features.contains(strings[0]))
                    features.add(strings[0]);
                if (! labels.contains(strings[1]))
                    labels.add( strings[1]);
                tmp.add(strings);
            }
            numFeatures = features.size() + 1; // add one more for padding,let it be [0,0,0,...1]
            numLabels = labels.size();
            int halfWindow = (winSize + 1)/2;
            getDataFeatureLabel(trainExamples, trainData, train_data_label, halfWindow);
            getDataFeatureLabel(tuneExamples, tuneData, tune_data_label, halfWindow);
            getDataFeatureLabel(testExamples, testData, test_data_label, halfWindow);
        }

        private ArrayList<Double> onehotCoding(int index, int size)
        {
            ArrayList<Double> onehot = new ArrayList<>();
            //Double[] onehot =  new Double[size];
            for  (int i = 0;  i < size; i++)
                onehot.add(0.0);
            //onehot[i] = 0.0;
            onehot.set(index, 1.0);
            //onehot[index] = 1.0;
            return onehot;
        }


        private void getDataFeatureLabel(ArrayList<ArrayList<String[]>> examples,
                                         ArrayList<ArrayList<Double>> data, ArrayList<ArrayList<Double>> data_label,
                                         int halfWindow)
        {
            for (ArrayList<String[]> example : examples) {
                for (int e = 0; e < example.size(); e++){

                    //int f_index = features.indexOf(example.get(e)[0]);
                    int l_index = labels.indexOf(example.get(e)[1]);
                    data_label.add(onehotCoding(l_index, numLabels));
                    ArrayList<Double> tmp_feature = new ArrayList<>(); // For ading data_feature
                    // padding number is  win/2 - e

                    for (int w = e - halfWindow + 1; w < e + halfWindow; w++){

                        if ( w < 0 || w >= example.size())
                            tmp_feature.addAll(onehotCoding(numFeatures - 1, numFeatures));
                        else
                        {
                            tmp_feature.addAll(onehotCoding(features.indexOf(example.get(w)[0]), numFeatures));
                        }
                    }
                    data.add(tmp_feature);

                }

            }
        }


        private void init() {
            numFeatures = -1;
            //numExamples = -1;
            features = new ArrayList<>();
            labels = new ArrayList<>();
            trainExamples = new ArrayList<>();
            tuneExamples = new ArrayList<>();
            testExamples = new ArrayList<>();
            trainData = new ArrayList<>();
            tuneData = new ArrayList<>();
            testData = new ArrayList<>();
            train_data_label = new ArrayList<>();
            tune_data_label = new  ArrayList<>();
            test_data_label = new ArrayList<>();
        }



        private boolean isUseful(String line) {
            if (line.isEmpty()) return false;
            if (line.startsWith("//")) return false;
            if (line.startsWith("#")) return false;
            if (line.equals("end")) return false;
            if (line.equals("<end>") ) return false;

            return true;
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

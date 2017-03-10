/**
 * Created by zhicheng on 1/22/17.
 */

import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

public class Network {
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

    private void forwardPropagation(Vector<Double> data) {
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

    private void backPropagation(Vector<Double> label) {
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


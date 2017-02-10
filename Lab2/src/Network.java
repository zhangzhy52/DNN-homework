/**
 * Created by zhicheng on 1/22/17.
 */

import java.util.ArrayList;
import java.util.Random;

public class Network {

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


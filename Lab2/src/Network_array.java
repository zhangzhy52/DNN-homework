import java.util.*;

/**
 * Created by zhicheng on 2/14/17.
 */
public class Network_array {

    public double[][] p_fx;
    public double[][] p_delta;
    public boolean[][] p_drop;

    public double[][][] w_value;
    public double[][][] w_delta;

    public double rate = 0.1;
    public double dropout_p = 1.1;
    public double weight_decay = 0.04;
    public double momentum = 0.1;

    private Random random = new Random();

    int[] numUnits;

    public Network_array(int[] numUnits, double rate, double dropout_p, double weight_decay, double momentum) {

        int max = 0;
        for (int num : numUnits) max = Math.max(max, num);

        // Init perceptrons.
        p_fx = new double[numUnits.length][max + 1];
        p_delta = new double[numUnits.length][max + 1];
        p_drop = new boolean[numUnits.length][max + 1];

        for (double[] p : p_fx) Arrays.fill(p, 1.0);
        for (double[] p : p_delta) Arrays.fill(p, 0.0);
        for (boolean[] p : p_drop) Arrays.fill(p, false);

        // Init weights.
        w_value = new double[numUnits.length - 1][max + 1][max + 1];
        w_delta = new double[numUnits.length - 1][max + 1][max + 1];

        for (double[][] w1 : w_value) for (double[] w2 : w1) Arrays.fill(w2, random.nextDouble());
        for (double[][] w1 : w_delta) for (double[] w2 : w1) Arrays.fill(w2, 0);

        // Learning rate.
        this.rate = rate;
        this.dropout_p = dropout_p;
        this.weight_decay = weight_decay;
        this.momentum = momentum;
        this.numUnits = numUnits;
    }


    public void train(double[] data, double[] label) {

        forwardPropagation(data);
        backPropagation(label);

    }

    public double test(double[][] datas, double[][] data_labels) {
        double correct = 0.0;

        for (int i = 0; i < datas.length; i++) {
            forwardPropagation(datas[i]);

            if (data_labels[i][outputIndex()] > 0.9) correct += 1;
        }

        return correct / datas.length;
    }

    private int outputIndex() {
        double max = 0.0;
        int index = 0;

        double[] layer = p_fx[p_fx.length - 1];

        for (int i = 0; i < numUnits[numUnits.length - 1]; i++) {
            if (layer[i] > max) {
                max = layer[i];
                index = i;
            }
        }

        return index;
    }

    private double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    private void forwardPropagation(double[] data) {
        // Input layer.
        System.arraycopy(data, 0, p_fx[0], 0, data.length);

        // Hidden layers and output layer.
        for (int l = 1; l < numUnits.length; l++) {
            for (int i = 0; i < numUnits[l]; i++) {

                // Dropout
                if (l != numUnits.length - 1 && random.nextDouble() > dropout_p) {
                    // Drop
                    p_drop[l][i] = true;
                } else {
                    // Keep
                    p_drop[l][i] = false;

                    double sum = 0.0;

                    for (int j = 0; j < numUnits[l - 1] + 1; j++) {
                        // If the input node is not dropped.
                        if (!p_drop[l - 1][j]) sum += p_fx[l - 1][j] * w_value[l - 1][j][i];
                    }

                    p_fx[l][i] = sigmoid(sum);
                }
            }
        }
    }

    private void backPropagation(double[] label) {
        // Delta
        for (int i = 0; i < label.length; i++) {
            double fx = p_fx[p_fx.length - 1][i];
            p_delta[p_delta.length - 1][i] = - (label[i] - fx) * fx * (1 - fx);
        }

        // Delta hidden layer.
        for (int l = numUnits.length - 2; l >= 0; l--) {
            for (int i = 0; i < numUnits[l] + 1; i ++) {
                if (p_drop[l][i]) continue;

                p_delta[l][i] = 0.0;

                for (int j = 0; j < numUnits[l + 1]; j++) {
                    if (!p_drop[l + 1][j]) p_delta[l][i] += p_delta[l + 1][j] * w_value[l][i][j];
                }

                p_delta[l][i] *= p_fx[l][i] * ( 1 - p_fx[l][i]);
            }
        }

        // Upadte weights.
        for (int l = 0; l < numUnits.length - 1; l++) {
            for (int i = 0; i < numUnits[l] + 1; i ++) {
                if (p_drop[l][i]) continue;

                for (int j = 0; j < numUnits[l + 1]; j ++) {
                    if (!p_drop[l + 1][j]) {
                        w_delta[l][i][j] = -rate * p_fx[l][i] * p_delta[l + 1][j]   // Regular
                                - rate * weight_decay * w_value[l][i][j]            // Weight decay
                                + momentum * w_delta[l][i][j];                      // Momentum

                        w_value[l][i][j] += w_delta[l][i][j];
                    }

                }
            }
        }
    }

}

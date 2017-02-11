/**
 * Created by zhicheng on 1/22/17.
 */
public class Weight {
    public Perceptron input;
    public Perceptron output;

    public double value;
    public double delta = 0.0;

    public Weight(double value) {
        this.value = value;
    }


}

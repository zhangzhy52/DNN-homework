/**
 * Created by zhicheng on 1/22/17.
 */

import java.util.ArrayList;

public class Perceptron {
    public ArrayList<Weight> inputs = new ArrayList<>();
    public ArrayList<Weight> outputs = new ArrayList<>();

    public double fx = 1;
    public double delta = 0;
    public boolean dropout = false;

    public Perceptron() {

    }

}

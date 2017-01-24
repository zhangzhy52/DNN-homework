/**
 * Created by zhicheng on 1/22/17.
 */
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class Lab1 {

    public static void main(String[] args) throws FileNotFoundException {

        DataReader dataReader1 = new DataReader();
//        dataReader1.read("red-wine-quality-train.data");
        dataReader1.read("Thoracic_Surgery_Data_train.data");

        DataReader dataReader2 = new DataReader();
//        dataReader2.read("red-wine-quality-tune.data");
        dataReader2.read("Thoracic_Surgery_Data_tune.data");

        ArrayList<Integer> numUnits = new ArrayList<>();
        numUnits.add(dataReader1.numFeatures);
        numUnits.add(10);
        numUnits.add(1);
        Network network = new Network(numUnits, 0.5);

        for (int k = 0; k < 10; k++) {
            for (int i = 0; i < dataReader1.data.size(); i++) {
                ArrayList<Double> tmp = new ArrayList<>();
                tmp.add(dataReader1.data_label.get(i));

                network.train(dataReader1.data.get(i), tmp);
            }


            ArrayList<ArrayList<Double>> data_label = new ArrayList<>();
            for (Double d : dataReader2.data_label) {
                ArrayList<Double> tmp = new ArrayList<>();
                tmp.add(d);
                data_label.add(tmp);
            }
            System.out.println(network.test(dataReader2.data, data_label));


            ArrayList<ArrayList<Double>> data_label2 = new ArrayList<>();
            for (Double d : dataReader1.data_label) {
                ArrayList<Double> tmp = new ArrayList<>();
                tmp.add(d);
                data_label2.add(tmp);
            }
            System.out.println(network.test(dataReader1.data, data_label2));
            System.out.println();

        }
    }

}

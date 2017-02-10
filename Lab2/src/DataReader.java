import javax.xml.crypto.Data;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.StringJoiner;

/**
 * Created by zhicheng on 1/23/17.
 */
public class DataReader {
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

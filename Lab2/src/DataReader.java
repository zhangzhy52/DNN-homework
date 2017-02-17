import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
//import java.util.StringJoiner;

/**
 * Created by zhicheng on 1/23/17.
 */
public class DataReader {
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

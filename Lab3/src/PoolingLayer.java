import java.util.ArrayList;
import java.util.Vector;

/**
 * Created by zhicheng on 3/8/17.
 * Pooling Layer.
 */
public class PoolingLayer {

    public int[][] map;

    public void forward(Layer layer1, Layer layer2) {
        map = new int[layer2.outputSize.x][layer2.outputSize.y];

        for (int num  = 0; num < layer1.outMapNum; num ++) {
            for (int c = 0; c < 4; c++) {
                for (int x = 0; x < layer2.outputSize.x; x ++) {
                    for (int y = 0; y < layer2.outputSize.y; y ++) {
                        int xx = x * 2;
                        int yy = y * 2;

                        double m = layer1.outMaps[num][c][xx][yy];
                        map[x][y] = 0;

                        if (layer1.outMaps[num][c][xx + 1][yy] > m) {
                            m = layer1.outMaps[num][c][xx + 1][yy];
                            map[x][y] = 1;
                        }

                        if (layer1.outMaps[num][c][xx][yy + 1] > m) {
                            m = layer1.outMaps[num][c][xx][yy + 1];
                            map[x][y] = 2;
                        }

                        if (layer1.outMaps[num][c][xx + 1][yy + 1] > m) {
                            m = layer1.outMaps[num][c][xx + 1][yy + 1];
                            map[x][y] = 3;
                        }

                        layer2.outMaps[num][c][x][y] = m;
                    }
                }
            }
        }
    }

    public void back(Layer layer1, Layer layer2) {
        for (int num  = 0; num < layer1.outMapNum; num ++) {
            for (int c = 0; c < 4; c++) {
                for (int x = 0; x < layer2.outputSize.x; x ++) {
                    for (int y = 0; y < layer2.outputSize.y; y ++) {
                        int xx = x * 2;
                        int yy = y * 2;

                        layer1.error[num][c][xx][yy] = 0.0;
                        layer1.error[num][c][xx + 1][yy] = 0.0;
                        layer1.error[num][c][xx][yy + 1] = 0.0;
                        layer1.error[num][c][xx + 1][yy + 1] = 0.0;

                        if (map[x][y] == 0) layer1.error[num][c][xx][yy] = layer1.error[num][c][x][y];
                        else if (map[x][y] == 1) layer1.error[num][c][xx + 1][yy] = layer1.error[num][c][x][y];
                        else if (map[x][y] == 2) layer1.error[num][c][xx][yy + 1] = layer1.error[num][c][x][y];
                        else if (map[x][y] == 3) layer1.error[num][c][xx + 1][yy + 1] = layer1.error[num][c][x][y];
                    }
                }
            }
        }


    }




    /*
    public final static int numCores = 4;
    public static int imgLength;

    public ArrayList<Double> input = new ArrayList<>();
    public ArrayList<Double> output = new ArrayList<>();

    public ArrayList<Double> delta = new ArrayList<>();
    public ArrayList<Double> delta2 = new ArrayList<>();

    public PoolingLayer(ArrayList<Double> input, ArrayList<Double> output) {
        this.input = input;
        this.output = output;

        imgLength = (int) Math.sqrt(input.size() / numCores);
    }

    public void forwardPropagation() {
        for (int core = 0; core < numCores; core++) {
            int start = (input.size() / numCores) * core;

            for (int x = 0; x < imgLength; x += 2) {
                for (int y = 0; y < imgLength; y += 2) {
                    int index1 = x       * imgLength + y;
                    int index2 = x       * imgLength + y + 1;
                    int index3 = (x + 1) * imgLength + y;
                    int index4 = (x + 1) * imgLength + y + 1;

                    double d = Math.max(Math.max(input.get(start + index1), input.get(start + index2)),
                            Math.max(input.get(start + index3), input.get(start + index4)));

                    output.add(d);
                }
            }
        }
    }

    public void backPropagation() {
        for (int i = 0; i < input.size(); i ++) delta.add(0.);

        for (int core = 0; core < numCores; core++) {
            int start = (input.size() / numCores) * core;

            int start2 = (delta2.size() / numCores) * core;

            for (int xx = 0; xx < imgLength / 2; xx += 1) {
                for (int yy = 0; yy < imgLength / 2; yy += 1) {
                    int index_2 = xx * imgLength / 2 + yy;

                    int x = xx * 2;
                    int y = yy * 2;

                    int index1 = x       * imgLength + y;
                    int index2 = x       * imgLength + y + 1;
                    int index3 = (x + 1) * imgLength + y;
                    int index4 = (x + 1) * imgLength + y + 1;

                    for (int index: new int[]{index1, index2, index3, index4}) {
                        if (input.get(start + index).equals(output.get(start2 + index_2))) {
                            delta.set(start + index, delta2.get(start2 + index_2));
                            break;
                        }
                    }

                }
            }
        }

    }
    */

}

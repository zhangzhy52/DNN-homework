import java.util.Random;
import java.util.Vector;



public class Layer {

    public String type; //type of layer
    public double[][][][] kernel;  //channel* image_number* kernelSize * kernelSize
    public double[] bias;
    public Size kernelSize;
    public int kernelNum  = 20;
    public Size outputSize;
    public int outMapNum;

    public int channelNum = 4;

    public double[][][][] outMaps;
    public double[][][][] error;

    private Layer(){

    }

    public static Layer buildConvLayer(int outMapNum, Size kernelSize) {
        Layer layer = new Layer();
        layer.type = "conv";
        layer.outMapNum = outMapNum;
        layer.kernelSize = kernelSize;
        return layer;
    }

    public void initOutMaps() {
        this.outMaps = new double[outMapNum][channelNum][outputSize.x][outputSize.y];
        for (int i = 0; i < outMapNum; i++)
            for (int j = 0; j < channelNum; j++)
                outMaps[i][j] = randomMatrix(kernelSize.x, kernelSize.y);
    }

    public void initKernel() {
//		int fan_out = getOutMapNum() * kernelSize.x * kernelSize.y;
//		int fan_in = frontMapNum * kernelSize.x * kernelSize.y;
//		double factor = 2 * Math.sqrt(6 / (fan_in + fan_out));
        this.kernel = new double[kernelNum][channelNum][kernelSize.x][kernelSize.y];
        for (int i = 0; i < kernelNum; i++)
            for (int j = 0; j < channelNum; j++)
                kernel[i][j] = randomMatrix(kernelSize.x, kernelSize.y);
    }


    public double[][] getKernel(int i, int j) {
        return kernel[i][j];
    }



    public static double[][] randomMatrix(int x, int y) {
        double[][] matrix = new double[x][y];
        Random r = new Random();
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {

                matrix[i][j] = (r.nextDouble() - 0.05) / 10;
            }
        }
        return matrix;
    }

    public static class Size{
        public final int x;
        public final int y;
        public Size(int x, int y)
        {
            this.x = x;
            this.y = y;
        }
    }




}

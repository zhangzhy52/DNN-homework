import java.util.ArrayList;
import java.util.List;



public class Convolution {



    private void forward(Layer prevLayer, Layer currLayer){
        int channel = currLayer.channelNum;
        int numKernel = currLayer.kernelNum;
        int kernelSizeX = currLayer.kernelSize.x;
        int kernelSizeY = currLayer.kernelSize.y;
        currLayer.outputSize = new Size(prevLayer.outputSize.x - kernelSizeX, kernelSizeY);

        for ()
            for (int j = 0; j < channel; j++)
                for ()
                    for()
                    { new double[kernelNum][channelNum][kernelSize.x][kernelSize.y];
                        new double[outMapNum][channelNum][outputSize.x][outputSize.y];
                        currLayer.outMaps[][][][] = * currLayer.kernel[][j][][];
                    }

    }





    private double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }



    public static class LayerBuilder {
        private List<Layer> mLayers;

        public LayerBuilder() {
            mLayers = new ArrayList<Layer>();
        }

        public LayerBuilder(Layer layer) {
            this();
            mLayers.add(layer);
        }

        public LayerBuilder addLayer(Layer layer) {
            mLayers.add(layer);
            return this;
        }
    }


}

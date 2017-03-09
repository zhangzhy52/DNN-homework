import java.util.ArrayList;
import java.util.List;



public class Convolution {


	public void createNextLayer(Layer preLayer ){
		Layer currLayer =  new Layer();
		currLayer.kernelSize = new Layer.Size(5,5);
		currLayer.initKernel();
		
		currLayer.outputSize = new Layer.Size(preLayer.outputSize.x - currLayer.kernelSize.x + 1,
				preLayer.outputSize.y - currLayer.kernelSize.y + 1);
		
		currLayer.outMapNum = preLayer.outMapNum * currLayer.kernelNum;
		
	}
	

    private void forward(Layer prevLayer, Layer currLayer){
        
        int numKernel = currLayer.kernelNum;
        int prevNum = prevLayer.outMapNum;
        for (int i = 0;i < currLayer.outMapNum; i++){
            for (int j = 0; j < currLayer.channelNum; j++){
               
                currLayer.outMaps[i][j] = convolve(prevLayer.outMaps[i/numKernel][j],
                				currLayer.kernel[i/prevNum][j]);  
            }
        }

    }
    
    private void backprop(Layer nextLayer, Layer currLayer){
    
    	
    }
    
   
    private double[][] convolve (double[][]image, double[][]kernel ) // convolution
    {
    	int iRow = image.length;
    	int iColumn = image[0].length;
    	int kRow = kernel.length;
    	int kColumn = kernel[0].length;
    	
    	double[][] result = new double[iRow - kRow + 1 ][ iColumn - kColumn + 1];
    	for (int i = 0; i < iRow - kRow + 1; i++)
    		for (int j = 0; j < iColumn - kColumn + 1; j++)
    		{
    			double tmp = 0;
    			for (int x = i; x < i + kRow; x++)
    				for (int y = j; y < j + kColumn; y++)
    				{
    					tmp += image[x][y] * kernel[x - i][y - j];
    				}
    			result[i][j] = tmp;
    		}
    	return result;
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

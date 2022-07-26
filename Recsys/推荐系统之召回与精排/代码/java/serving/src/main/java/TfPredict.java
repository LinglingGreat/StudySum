import org.tensorflow.*;
import org.tensorflow.Graph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
public class TfPredict {
    public static void main(String args[]){
        byte[] graphDef = loadTensorflowModel("F:\\serving\\src\\main\\model\\rf.pb");
        float inputs[][] = new float[2][4];

        inputs[0]= new float[]{1.3f, 1.4f, 0.9f, 1.2f};
        inputs[1]= new float[]{1.2f, 0.4f, 1.9f, 1.4f};
        Tensor<Float> input = covertArrayToTensor(inputs);
        Graph g = new Graph();
        g.importGraphDef(graphDef);
        Session s = new Session(g);
        Tensor result = s.runner().feed("input", input).fetch("output").run().get(0);

        long[] rshape = result.shape();
        int rs = (int) rshape[0];
        long realResult[] = new long[rs];
        result.copyTo(realResult);

        for(long a: realResult ) {
            System.out.println(a);
        }
    }
    static private byte[] loadTensorflowModel(String path){
        try {
            return Files.readAllBytes(Paths.get(path));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    static private Tensor<Float> covertArrayToTensor(float inputs[][]){
        return Tensors.create(inputs);
    }
}

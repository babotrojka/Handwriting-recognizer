package hr.fer.mekorac.neural.algorithms;

import hr.fer.mekorac.data.Data;
import hr.fer.mekorac.neural.NeuralNetwork;
import hr.fer.mekorac.neural.Neuron;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

public class BatchNeuralNetwork extends NeuralNetwork {
    public BatchNeuralNetwork(UnaryOperator<Double> activationFunction, int inputSize, int outputSize, int... hiddenSizes) {
        super(activationFunction, inputSize, outputSize, hiddenSizes);
    }

    public BatchNeuralNetwork(int inputSize, int outputSize, int... hiddenSizes) {
        super(inputSize, outputSize, hiddenSizes);
    }

    public BatchNeuralNetwork(UnaryOperator<Double> activationFunction, List<Double> inputValues, int outputSize, int... hiddenSizes) {
        super(activationFunction, inputValues, outputSize, hiddenSizes);
    }

    public BatchNeuralNetwork(List<Double> inputValues, int outputSize, int... hiddenSizes) {
        super(inputValues, outputSize, hiddenSizes);
    }

    public BatchNeuralNetwork(NeuralNetwork other) {
        super(other);
    }

    @Override
    public void backpropagation(Data data) {
        RealMatrix[] errs = new RealMatrix[outputLayerIndex];
        for(int i = 0; i < errs.length; i++) {
            errs[i] = MatrixUtils.createRealMatrix(findLayer(i).size() + 1, findLayer(i + 1).size());
        }

        //System.out.println(errs[errs.length - 1].getRowDimension() + " : " + errs[errs.length - 1].getColumnDimension());
        for (int s = 0; s < data.getDataSize(); s++) {
            RealVector trueOutputs = realVectorFromListDouble(data.getOutputDataAt(s));

            RealVector outputVector = this.calculate(data.getInputDataAt(s));

            double[] onesArray = new double[outputVector.getDimension()];
            Arrays.fill(onesArray, 1);
            RealVector ones = MatrixUtils.createRealVector(onesArray);
            RealVector errorVector = outputVector.ebeMultiply(ones.subtract(outputVector)).ebeMultiply(trueOutputs.subtract(outputVector));

            RealVector oneOne = MatrixUtils.createRealVector(new double[]{1});
            RealVector outs = oneOne.append(vectorOfOutputsFromLayer(outputLayerIndex - 1));//vektor izlaza
            errs[errs.length - 1] = errs[errs.length - 1].add(outs.outerProduct(errorVector));

            //System.out.println("output error: " + errorVector);
            for (int layer = hidden.size(); layer > 0; layer--) {
                List<Neuron> currentLayer = findLayer(layer);
                double[] errors = new double[currentLayer.size()];
                for (int pos = 0; pos < currentLayer.size(); pos++) {
                    Neuron currentNeuron = currentLayer.get(pos);
                    //System.out.println("weights from me " + layer + " " + pos + " " + vectorOfWeightsFromMe(layer, pos));
                    //System.out.println("after mul " + this.vectorOfWeightsFromMe(layer, pos).ebeMultiply(errorVector));
                    double downstreamSum = sumOfVector(this.vectorOfWeightsFromMe(layer, pos).ebeMultiply(errorVector));
                    //System.out.println("downSum " + layer + " " + pos + " " + downstreamSum);
                    //System.out.println("current output: " + currentNeuron.getOutput());
                    double currentError = currentNeuron.getOutput() * (1 - currentNeuron.getOutput()) * downstreamSum; //Racunanje errora
                    errors[pos] = currentError;
                    //System.out.println("cur err " + layer + " " + pos + ": " + currentError);

                }
                errorVector = MatrixUtils.createRealVector(errors);

                /*onesArray = new double[errorVector.getDimension()];
                Arrays.fill(onesArray, 1);
                ones = MatrixUtils.createRealVector(onesArray);
*/
                outs = oneOne.append(vectorOfOutputsFromLayer(layer - 1));
                errs[layer - 1] = errs[layer - 1].add(outs.outerProduct(errorVector));
            }
        }

        for (int i = 0; i < this.output.size(); i++) {
            //System.out.println("w0 change for output: " + this.output.get(i).getWeight_0() + this.ETA * errorVector.getEntry(i)   );
            this.output.get(i).setWeight_0(this.output.get(i).getWeight_0() + this.ETA * errs[errs.length - 1].getEntry(0, i));
        }

        for (int layer = hidden.size(); layer >= 0; layer--) {
            List<Neuron> currentLayer = findLayer(layer);
            for (int pos = 0; pos < currentLayer.size(); pos++) {
                Neuron currentNeuron = currentLayer.get(pos);
                if (layer > 0) currentNeuron.setWeight_0(currentNeuron.getWeight_0() + this.ETA * errs[layer - 1].getEntry(0, pos));
                for (int i = 0; i < currentNeuron.getNumberOfWeightsTo(); i++) { //mijenjanje weightsa
                    //System.out.printf("(%d, %d) to %d old weight %f%n", layer, pos, i, currentNeuron.getWeightTo(i));
                    //System.out.printf("(%d, %d) to %d output %f%n", layer, pos, i, currentNeuron.getOutput());
                    currentNeuron.setWeightTo(i, currentNeuron.getWeightTo(i) + this.ETA * errs[layer].getEntry(pos + 1, i));
                    //System.out.printf("(%d, %d) to %d new weight %f%n", layer, pos, i, currentNeuron.getWeightTo(i));
                }
            }
        }
    }
}

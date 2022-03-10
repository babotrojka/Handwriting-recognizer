package hr.fer.mekorac.neural.algorithms;

import hr.fer.mekorac.data.Data;
import hr.fer.mekorac.neural.NeuralNetwork;
import hr.fer.mekorac.neural.Neuron;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;

import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

public class StochasticNeuralNetwork extends NeuralNetwork {


    public StochasticNeuralNetwork(UnaryOperator<Double> activationFunction, int inputSize, int outputSize, int... hiddenSizes) {
        super(activationFunction, inputSize, outputSize, hiddenSizes);
    }

    public StochasticNeuralNetwork(int inputSize, int outputSize, int... hiddenSizes) {
        super(inputSize, outputSize, hiddenSizes);
    }

    public StochasticNeuralNetwork(UnaryOperator<Double> activationFunction, List<Double> inputValues, int outputSize, int... hiddenSizes) {
        super(activationFunction, inputValues, outputSize, hiddenSizes);
    }

    public StochasticNeuralNetwork(List<Double> inputValues, int outputSize, int... hiddenSizes) {
        super(inputValues, outputSize, hiddenSizes);
    }

    public StochasticNeuralNetwork(NeuralNetwork other) {
        super(other);
    }

    @Override
    public void backpropagation(Data data) {
        for (int s = 0; s < data.getDataSize(); s++) {
            RealVector trueOutputs = realVectorFromListDouble(data.getOutputDataAt(s));

            RealVector outputVector = this.calculate(data.getInputDataAt(s));

            double[] onesArray = new double[outputVector.getDimension()];
            Arrays.fill(onesArray, 1);
            RealVector ones = MatrixUtils.createRealVector(onesArray);
            RealVector errorVector = outputVector.ebeMultiply(ones.subtract(outputVector)).ebeMultiply(trueOutputs.subtract(outputVector));

            //System.out.println("output error: " + errorVector);
            for (int i = 0; i < this.output.size(); i++) {
                //System.out.println("w0 change for output: " + this.output.get(i).getWeight_0() + this.ETA * errorVector.getEntry(i)   );
                this.output.get(i).setWeight_0(this.output.get(i).getWeight_0() + this.ETA * errorVector.getEntry(i));
            }

            for (int layer = hidden.size(); layer >= 0; layer--) {
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

                    if (layer > 0) currentNeuron.setWeight_0(currentNeuron.getWeight_0() + this.ETA * currentError);
                    for (int i = 0; i < currentNeuron.getNumberOfWeightsTo(); i++) { //mijenjanje weightsa
                        //System.out.printf("(%d, %d) to %d old weight %f%n", layer, pos, i, currentNeuron.getWeightTo(i));
                        //System.out.printf("(%d, %d) to %d output %f%n", layer, pos, i, currentNeuron.getOutput());
                        currentNeuron.setWeightTo(i, currentNeuron.getWeightTo(i) + this.ETA * currentNeuron.getOutput() * errorVector.getEntry(i));
                        //System.out.printf("(%d, %d) to %d new weight %f%n", layer, pos, i, currentNeuron.getWeightTo(i));
                    }
                }
                errorVector = MatrixUtils.createRealVector(errors);
            }
        }
    }


}

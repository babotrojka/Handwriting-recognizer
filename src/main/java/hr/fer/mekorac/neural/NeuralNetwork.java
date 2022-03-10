package hr.fer.mekorac.neural;

import hr.fer.mekorac.data.Data;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorPreservingVisitor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;

public abstract class NeuralNetwork {
    protected List<Neuron> input;
    public List<List<Neuron>> hidden;
    protected List<Neuron> output;

    protected int outputLayerIndex;

    private UnaryOperator<Double> activationFunction;
    protected double ETA = 0.2;

    /**
     * Init with random values for weights
     *
     * @param inputSize
     * @param outputSize
     * @param hiddenSizes
     */
    public NeuralNetwork(UnaryOperator<Double> activationFunction, int inputSize, int outputSize, int... hiddenSizes) {
        this.activationFunction = activationFunction;

        Random r = new Random();

        input = new ArrayList<>();
        for (int i = 0; i < inputSize; i++) {//init input
            input.add(new Neuron(0));
        }

        output = new ArrayList<>();
        for (int i = 0; i < outputSize; i++) //init output
            output.add(new Neuron(r.nextGaussian() * 0.1));

        int nextSize = hiddenSizes.length == 0 ? outputSize : hiddenSizes[0];

        for (int i = 0; i < inputSize; i++) //weights from input
            for (int j = 0; j < nextSize; j++)
                input.get(i).addWeightTo(r.nextGaussian() * 0.1);

        hidden = new ArrayList<>();
        for (int i = 0; i < hiddenSizes.length; i++) {
            hidden.add(new ArrayList<>());
            for (int j = 0; j < hiddenSizes[i]; j++) {
                hidden.get(i).add(new Neuron(r.nextGaussian() * 0.1)); //hidden neuron
                nextSize = i < hiddenSizes.length - 1 ? hiddenSizes[i + 1] : outputSize; //weights from hidden
                for (int k = 0; k < nextSize; k++)
                    hidden.get(i).get(j).addWeightTo(r.nextGaussian() * 0.1);
            }
        }
        outputLayerIndex = hiddenSizes.length + 1;
    }

    public NeuralNetwork(int inputSize, int outputSize, int... hiddenSizes) {
        this(d -> 1 / (1 + Math.exp(-d)), inputSize, outputSize, hiddenSizes);
    }

    public NeuralNetwork(UnaryOperator<Double> activationFunction, List<Double> inputValues, int outputSize, int... hiddenSizes) {
        this(activationFunction, inputValues.size(), outputSize, hiddenSizes);
        setInputValues(inputValues);
    }


    public NeuralNetwork(List<Double> inputValues, int outputSize, int... hiddenSizes) {
        this(d -> 1 / (1 + Math.exp(-d)), inputValues, outputSize, hiddenSizes);
    }

    public NeuralNetwork(NeuralNetwork other) {
        this(other.input.size(), other.output.size(), other.hidden.stream().mapToInt(List::size).toArray());
    }


    public void setETA(double ETA) {
        this.ETA = ETA;
    }

    public void setInputValues(List<Double> inputValues) {
        if(input.size() != inputValues.size())
            throw new IllegalArgumentException("Invalid size of input values");

        for(int i = 0; i < input.size(); i++)
            input.get(i).setOutput(inputValues.get(i));
    }

    /**
     * Gets vector of weights to a particular neuron.
     * Layer 0 is input layer and represents invalid input
     * Position starts from 0
     * @param layer
     * @param position
     * @return
     */
    private RealVector vectorOfWeightsToMe(int layer, int position) {
        if(layer == 0)
            throw new IllegalArgumentException("Layer cant be 0");
        if(layer > hidden.size() + 1)
            throw new IllegalArgumentException("Layer is out of bounds");

        List<Neuron> layerBefore = findLayer(layer - 1);

        if(position < 0 || position >= layerBefore.get(0).getNumberOfWeightsTo())
            throw new IllegalArgumentException("Position is out of bounds");

        double[] weights = new double[layerBefore.size()];
        for(int i = 0; i < layerBefore.size(); i++) {
            weights[i] = layerBefore.get(i).getWeightTo(position);
        }

        return MatrixUtils.createRealVector(weights);
    }

    /**
     * Vector of weights from given neuron
     * @param layer
     * @param position
     * @return
     */
    protected RealVector vectorOfWeightsFromMe(int layer, int position) {
        if(layer == outputLayerIndex)
            throw new IllegalArgumentException("Output layer doesnt have out weights");
        if(layer < 0 || layer > outputLayerIndex)
            throw new IllegalArgumentException("Layer is out of bounds");

        List<Neuron> currentLayer = findLayer(layer);

        if(position < 0 || position >= currentLayer.size())
            throw new IllegalArgumentException("Position is out of bounds");

        return realVectorFromListDouble(currentLayer.get(position).getWeights_to());
    }

    /**
     * Sets weights of given neuron
     * @param layer
     * @param position
     * @param weights
     */
    private void setWeightsFromMe(int layer, int position, RealVector weights) {
        if(layer == outputLayerIndex)
            throw new IllegalArgumentException("Output layer doesnt have out weights");
        if(layer < 0 || layer > outputLayerIndex)
            throw new IllegalArgumentException("Layer is out of bounds");

        List<Neuron> currentLayer = findLayer(layer);

        if(position < 0 || position >= currentLayer.size())
            throw new IllegalArgumentException("Position is out of bounds");

        currentLayer.get(position).setWeights_to(Arrays.stream(weights.toArray()).boxed().collect(Collectors.toList()));
    }

    /**
     * Gets vector of outputs from given layer
     * @param layer
     * @return
     */
    protected RealVector vectorOfOutputsFromLayer(int layer) {
        return realVectorFromListDouble(findLayer(layer).stream().map(Neuron::getOutput).collect(Collectors.toList()));
    }

    /**
     * Gets vector of weights 0 from given layer
     * @param layer
     * @return
     */
    private RealVector vectorOfWeights0(int layer) {
        return realVectorFromListDouble(findLayer(layer).stream().map(Neuron::getWeight_0).collect(Collectors.toList()));
    }

    /**
     * Sets weight0 of every neuron to given value
     * @param layer
     * @param weights0
     */
    private void setVectorOfWeights0(int layer, RealVector weights0) {
        List<Neuron> currentLayer = findLayer(layer);
        if(currentLayer.size() != weights0.getDimension())
            throw new IllegalArgumentException("Looking for length " + currentLayer.size() + ", provided is " + weights0.getDimension());

        for(int i = 0; i < currentLayer.size(); i++)
            currentLayer.get(i).setWeight_0(weights0.getEntry(i));
    }

    /**
     * Util function to create vector from list of doubles
     * @param nums
     * @return
     */
    protected RealVector realVectorFromListDouble(List<Double> nums) {
        double[] weights = new double[nums.size()];
        for(int i = 0; i < nums.size(); i++)
            weights[i] = nums.get(i);

        return MatrixUtils.createRealVector(weights);
    }

    /**
     * Gets RealMatrix of out weights from given layer to the next one
     * @param layer
     * @return
     */
    private RealMatrix matrixOfOutWeightsFromLayer(int layer) {
        if(layer == hidden.size() + 2)
            throw new IllegalArgumentException("Output layer doesnt have out weights");

        List<Neuron> neuronLayer = findLayer(layer);
        List<Neuron> nextLayer = findLayer(layer + 1);

        RealMatrix matrix = MatrixUtils.createRealMatrix(nextLayer.size(), neuronLayer.size());
        for(int i = 0; i < nextLayer.size(); i++) {
            matrix.setRowVector(i, vectorOfWeightsToMe(layer + 1, i));
        }
        return matrix;
    }


    /**
     * Constructs a vector of all weights in this neural network
     * For each layer, for each neuron, we append vectorOfWeightsFromMe
     * Then we add weights0 from next layer
     * @return
     */
    public RealVector vectorOfAllWeights() {
        RealVector v = MatrixUtils.createRealVector(new double[0]);
        for(int i = 0; i < outputLayerIndex; i++) {
            List<Neuron> currentLayer = findLayer(i);
            for(int j = 0; j < currentLayer.size(); j++)
                v = v.append(vectorOfWeightsFromMe(i, j));
            v = v.append(vectorOfWeights0(i + 1));
        }

        return v;
    }

    /**
     * Sets weights of this nn from given vector
     * @param weights
     */
    public void setWeightsFromVector(RealVector weights) {
        int weightsRead = 0;
        for(int i = 0; i < outputLayerIndex; i++) {
            int currentLayerSize = findLayer(i).size();
            int nextLayerSize = findLayer(i + 1).size();
            for(int j = 0; j < currentLayerSize; j++) {
                RealVector weightsFromMe = weights.getSubVector(weightsRead, nextLayerSize);
                weightsRead += nextLayerSize;
                setWeightsFromMe(i, j, weightsFromMe);
            }

            RealVector weightsOfNextLayer = weights.getSubVector(weightsRead, nextLayerSize);
            weightsRead += nextLayerSize;
            setVectorOfWeights0(i + 1, weightsOfNextLayer);
        }
    }

    /**
     * Sets output of given layer to vector values
     * @param vector
     * @param layer
     */
    private void setOutput(RealVector vector, int layer) {
        List<Neuron> neurons = findLayer(layer);
        if(neurons.size() != vector.getDimension())
            throw new IllegalArgumentException();

        for(int i = 0; i < neurons.size(); i++) {
            neurons.get(i).setOutput(vector.getEntry(i));
        }
    }

    private void performActivationFunctionOnVector(RealVector input) {
        for(int i = 0; i < input.getDimension(); i++)
            input.setEntry(i, activationFunction.apply(input.getEntry(i)));
    }


    private RealVector calculateAndSetOutputOfLayer(int layer) {
        if(layer == 0)
            return realVectorFromListDouble(input.stream().map(Neuron::getOutput).collect(Collectors.toList()));

        RealMatrix weights = matrixOfOutWeightsFromLayer(layer - 1);
        RealVector outputs = vectorOfOutputsFromLayer(layer - 1);

        RealVector weights0 = vectorOfWeights0(layer);

        RealVector net = weights.operate(outputs).add(weights0);
        //if(layer != outputLayerIndex)
            performActivationFunctionOnVector(net);

        setOutput(net, layer);

        return net;
    }

    /**
     * Calculates output from input data
     * @param inputData
     * @return
     */
    public RealVector calculate(List<Double> inputData) {
        setInputValues(inputData);

        for(int i = 0; i < hidden.size(); i++) //Setting outputs of hidden layers
            calculateAndSetOutputOfLayer(i + 1);

        calculateAndSetOutputOfLayer(outputLayerIndex);

        return realVectorFromListDouble(output.stream().map(Neuron::getOutput).collect(Collectors.toList()));
    }

    public double calculateMeanSquaredErrorFromData(Data data) {
        double meanError = 0;
        for(int i = 0; i < data.getDataSize(); i++) {
            RealVector result = this.calculate(data.getInputDataAt(i)); //ova linija
            for(int j = 0; j < result.getDimension(); j++)
                meanError += Math.pow(result.getEntry(j) - data.getOutputDataAt(i).get(j), 2); //ova linija
        }

        return meanError / data.getDataSize();
    }

    public abstract void backpropagation(Data data);

    public double sumOfVector(RealVector v) {
        return v.walkInDefaultOrder(new RealVectorPreservingVisitor() {
            double sum;

            @Override
            public void start(int i, int i1, int i2) {
                sum = 0;
            }

            @Override
            public void visit(int i, double v) {
                sum += v;
            }

            @Override
            public double end() {
                return sum;
            }
        });
    }

    protected List<Neuron> findLayer(int layer) {
        if(layer < 0 || layer > hidden.size() + 2)
            throw new IllegalArgumentException("Layer is out of bounds");

        List<Neuron> neuronlayer = null;
        if(layer == 0)
            neuronlayer = input;
        else if(layer == hidden.size() + 1)
            neuronlayer = output;
        else
            neuronlayer = hidden.get(layer - 1);

        return neuronlayer;
    }


}

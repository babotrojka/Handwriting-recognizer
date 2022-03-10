package hr.fer.mekorac.neural;

import java.util.ArrayList;
import java.util.List;

public class Neuron {
    private double weight_0;
    private double output;

    private List<Double> weights_to;

    public Neuron(double weight_0) {
        this.weight_0 = weight_0;
        weights_to = new ArrayList<>();
    }

    public void addWeightTo(double weight) {
        weights_to.add(weight);
    }

    public Double getWeightTo(int index) {
        return weights_to.get(index);
    }

    public void setWeightTo(int index, double value) {
        weights_to.set(index, value);
    }

    public int getNumberOfWeightsTo() {
        return weights_to.size();
    }

    public double getOutput() {
        return output;
    }

    public double getWeight_0() {
        return weight_0;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public void setWeight_0(double weight_0) {
        this.weight_0 = weight_0;
    }

    public List<Double> getWeights_to() {
        return weights_to;
    }

    public void setWeights_to(List<Double> weights_to) {
        this.weights_to = weights_to;
    }
}

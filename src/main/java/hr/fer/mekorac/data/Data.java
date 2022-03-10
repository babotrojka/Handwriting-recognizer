package hr.fer.mekorac.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class Data {
    protected List<List<Double>> inputData = new ArrayList<>();
    protected List<List<Double>> outputData = new ArrayList<>();

    public List<Double> getInputDataAt(int index) {
        return inputData.get(index);
    }
    public List<Double> getOutputDataAt(int index) {
        return outputData.get(index);
    }
    public int getDataSize() {
        return inputData.size();
    }

    public void shuffle() {
        class Combined {
            List<Double> input;
            List<Double> output;

            public Combined(List<Double> input, List<Double> output) {
                this.input = input;
                this.output = output;
            }
        }

        List<Combined> combineds = new ArrayList<>();
        for(int i = 0; i < getDataSize(); i++) {
            combineds.add(new Combined(inputData.get(i), outputData.get(i)));
        }
        Collections.shuffle(combineds);

        inputData = new ArrayList<>();
        outputData = new ArrayList<>();
        for(int i = 0; i < combineds.size(); i++) {
            inputData.add(combineds.get(i).input);
            outputData.add(combineds.get(i).output);
        }
    }
}

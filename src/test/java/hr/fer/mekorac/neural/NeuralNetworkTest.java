package hr.fer.mekorac.neural;

import hr.fer.mekorac.data.Data;
import hr.fer.mekorac.data.QuadraticData;
import hr.fer.mekorac.data.SampleData;
import hr.fer.mekorac.neural.algorithms.BatchNeuralNetwork;
import hr.fer.mekorac.neural.algorithms.StochasticNeuralNetwork;
import org.apache.commons.math3.linear.MatrixUtils;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

public class NeuralNetworkTest {

    private Path qFile = Path.of("q.txt");
    private Path qTest = Path.of("qTest.txt");

    private Path qLargeTrain = Path.of("q_large_train.txt");
    private Path qLargeTest = Path.of("q_large_test.txt");

    private Path samples = Path.of("dataset.txt");
    private Path samples_test = Path.of("dataset_test.txt");


    @Test
    public void testQuadratic1() {
        Data data = new QuadraticData(qLargeTrain);
        Data testData = new QuadraticData(qLargeTest);
        NeuralNetwork network = new StochasticNeuralNetwork(1, 1, 6);
        network.setETA(0.1);

        double msq;
        int cnt = 0;
        do {
            network.backpropagation(data);
            msq = network.calculateMeanSquaredErrorFromData(data);
            //System.out.println("msq from train data: " + msq);
            //System.out.println("msq from test data: " + network.calculateMeanSquaredErrorFromData(testData));
            data.shuffle();
            cnt++;
            if(cnt % 1000 == 0) {
                System.out.printf("At %dth iteration. MSE is %f\n", cnt, msq);
            }
        } while (msq > 0.0015);

        for(int i = 0; i < testData.getDataSize(); i++) {
            List<Double> input = testData.getInputDataAt(i);
            System.out.println("For input " + input + " gotten " + network.calculate(input));
        }

        System.out.println("Finished after " + cnt  + " epochs");
        System.out.printf("MSE on test set: %f\n", network.calculateMeanSquaredErrorFromData(testData));
    }

    @Test
    public void calculate() {
        Data data = new QuadraticData(qFile);
        NeuralNetwork network = new StochasticNeuralNetwork(1, 1, 6);
        System.out.println(network.vectorOfAllWeights());
        System.out.println("For input " + data.getInputDataAt(0) + " output is " + network.calculate(data.getInputDataAt(0)));
    }

    @Test
    public void backprop() {
        Data data = new QuadraticData(qFile);
        NeuralNetwork network = new StochasticNeuralNetwork(1, 1, 6);
        network.setWeightsFromVector(MatrixUtils.createRealVector(new double[] {-0.0557028688, 0.0146467139, 0.157927397, 0.0751835901, -0.0900666985, 0.019095667, 0.1334643053, -0.1418765501, 0.1697882797, -0.0399688401, 0.1032794851, 0.1172992668, -0.1354952822, -0.0069427131, 0.0550165668, -0.1108954477, -0.133899909, -0.0855885343, 0.0779448108}));
        System.out.println("For input " + data.getInputDataAt(0) + " output is " + network.calculate(data.getInputDataAt(0)));
        //network.backpropagation(data.getInputDataAt(0), data.getOutputDataAt(0));
    }

    @Test
    public void testLetters()  {
        Data data = new SampleData(samples);
        Data testData = new SampleData(samples_test);

        //for(int i = 0; i < data.getDataSize(); i++) System.out.println(data.getInputDataAt(i).size());
        NeuralNetwork network = new BatchNeuralNetwork(50 * 2, 5, 10);
        network.setETA(0.01);

        double msq;
        int cnt = 0;
        do {
            network.backpropagation(data);
            msq = network.calculateMeanSquaredErrorFromData(data);
            //System.out.println("msq from train data: " + msq);
            //System.out.println("msq from test data: " + network.calculateMeanSquaredErrorFromData(testData));
            data.shuffle();
            cnt++;
            if(cnt % 100 == 0) {
                System.out.printf("At %dth iteration. MSE is %f\n", cnt, msq);
            }
        } while (msq > 0.0015);

        for(int i = 0; i < testData.getDataSize(); i++) {
            List<Double> input = testData.getInputDataAt(i);
            System.out.println(network.calculate(input));
        }

        System.out.println("Finished after " + cnt  + " epochs");
        System.out.printf("MSE on test set: %f\n", network.calculateMeanSquaredErrorFromData(testData));

    }
}

package hr.fer.mekorac.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class QuadraticData extends Data{
    public QuadraticData(Path file) {
        try {
            List<String> lines = Files.readAllLines(file);
            for(String line : lines) {
               List<Double> inputs = new ArrayList<>(Arrays.asList(Double.parseDouble(line.split(",")[0])));
               inputData.add(inputs);

                List<Double> outputs = new ArrayList<>(Arrays.asList(Double.parseDouble(line.split(",")[1])));
                outputData.add(outputs);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

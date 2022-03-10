package hr.fer.mekorac.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SampleData extends Data{

    public SampleData(Path file) {
        try {
            List<String> lines = Files.readAllLines(file);
            for(String line : lines) {
                String inputLine = line.split("%")[0];
                String outputLine = line.split("%")[1];

                List<Double> inputs = new ArrayList<>();
                String[] points = inputLine.split("\t");
                //System.out.println(points.length);
                for(String point : points) {
                    inputs.add(Double.parseDouble(point.split(",")[0]));
                    inputs.add(Double.parseDouble(point.split(",")[1]));
                }
                inputData.add(inputs);
                List<Double> reversedInputs = new ArrayList<>();
                for(int i = inputs.size() - 2; i >= 0; i -= 2) {
                    reversedInputs.add(inputs.get(i));
                    reversedInputs.add(inputs.get(i + 1));
                }
                inputData.add(reversedInputs);

                List<Double> o = new ArrayList<>();
                String[] outs = outputLine.split(",");
                for(String out : outs) {
                    o.add(Double.parseDouble(out.trim()));
                }
                outputData.add(o);
                outputData.add(o);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

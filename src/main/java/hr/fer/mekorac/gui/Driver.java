package hr.fer.mekorac.gui;

import hr.fer.mekorac.data.Data;
import hr.fer.mekorac.data.SampleData;
import hr.fer.mekorac.neural.NeuralNetwork;
import hr.fer.mekorac.neural.algorithms.BatchNeuralNetwork;
import hr.fer.mekorac.neural.algorithms.StochasticNeuralNetwork;
import hr.fer.mekorac.preprocessing.Board;
import hr.fer.mekorac.preprocessing.Preprocessing;
import hr.fer.mekorac.preprocessing.point.Point;
import org.apache.commons.math3.linear.RealVector;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.stream.Collectors;

public class Driver extends JFrame {
    private String type;
    private String file;
    private int M;
    private int epochs;
    private double msq;

    NeuralNetwork network;

    private Map<String, List<List<Point<Integer>>>> data = new HashMap<>();
    private String[] letters = new String[] {"alpha", "beta", "gamma", "delta", "eta"};
    private Map<String, String> letterRep = new HashMap<>();

    String currentSelection;

    public Driver(String type, int M, int epochs, double msq, String file) throws HeadlessException {
        this.M = M;
        this.type = type;
        this.epochs = epochs;
        this.msq = msq;
        this.file = file;

        setBounds(100, 100, 500, 500);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        prepareLetters();
        initGUI();
    }

    private void prepareLetters() {
        for (int i = 0; i < letters.length; i++) {
            data.put(letters[i], new ArrayList<>());
            StringBuilder bobTheBuilder = new StringBuilder();
            for(int j = 0; j < letters.length; j++) {
                bobTheBuilder.append(j == i ? "1" : "0");
                if(j < letters.length - 1) bobTheBuilder.append(",");
            }
            letterRep.put(letters[i], bobTheBuilder.toString());
        }

        currentSelection = letters[0];
    }

    private void initGUI() {
        JTabbedPane tabbedPane = new JTabbedPane();

        JPanel pane = new JPanel();
        pane.setLayout(new BorderLayout());

        setLayout(new BorderLayout());

        JPanel north = new JPanel();

        JLabel total = new JLabel("Total so far for this letter is " + data.get(currentSelection).size());

        JComboBox<String> options = new JComboBox<>(letters);
        options.addActionListener(l -> {
            currentSelection = options.getSelectedItem().toString();
            total.setText("Total so far for this letter is " + data.get(currentSelection).size());
        });
        north.add(options);
        north.add(total);

        pane.add(north, BorderLayout.NORTH);

        Board board = new Board(new ArrayList<>());
        pane.add(board, BorderLayout.CENTER);

        JPanel south = new JPanel();
        JButton clearButton = new JButton("Clear");
        clearButton.addActionListener(l -> board.clearBoard());
        south.add(clearButton);

        JButton saveButton = new JButton("Save letter");
        saveButton.addActionListener(l -> {
            data.get(currentSelection).add(board.getPoints());
            board.clearBoard();
            total.setText("Total so far for this letter is " + data.get(currentSelection).size());
        });
        south.add(saveButton);
        JButton finishButton = new JButton("Finish!");
        finishButton.addActionListener(l -> {
            try {
                BufferedWriter writer = new BufferedWriter(new FileWriter(file));
                for(String letter : data.keySet()) {
                    //System.out.println("Looking at letter " + letter + " with size " + data.get(letter).size());
                    for(List<Point<Integer>> sample : data.get(letter)) {
                        List<Point<Double>> results = new Preprocessing(sample, M).process();
                        //System.out.println("Size of preprocessing results " + results.size());
                        for(Point<Double> p : results) {
                            writer.write(p.getX() + "," + p.getY());
                            if(results.indexOf(p) < results.size() - 1) writer.write("\t");
                            else writer.write("%");
                        }
                        writer.write(letterRep.get(letter) + "\n");
                    }
                }
                System.out.println("WRITING FINISHED!");
                writer.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
            resetData();
            board.clearBoard();
            total.setText("Total so far for this letter is " + data.get(currentSelection).size());
        });
        south.add(finishButton);
        pane.add(south, BorderLayout.SOUTH);

        pane.addMouseMotionListener(new BoardMouseMotionListener(board));
        tabbedPane.addTab("Training", pane);

        JPanel classPane = new JPanel();
        classPane.setLayout(new BorderLayout());
        JLabel results = new JLabel("Letter: -\tAlpha: -\tBeta: -\tGamma: -\tDelta: -\tEta: -\t".replaceAll("\t", "   "));
        classPane.add(results, BorderLayout.NORTH);

        Board classBoard = new Board(new ArrayList<>());
        classPane.add(classBoard, BorderLayout.CENTER);

        JPanel classSouth = new JPanel();
        JButton classify = new JButton("Classify");
        classify.setEnabled(false);

        switch (type) {
            case "stochastic" -> network = new StochasticNeuralNetwork(2 * M, data.size(), 10);
            case "batch" -> {
                network = new BatchNeuralNetwork(2 * M, data.size(), 10);
                network.setETA(0.01);
            }
            default -> {
                System.err.println("Wrong network type!");
                System.exit(0);
            }
        }
        JButton train = new JButton("Train");
        train.addActionListener(l -> {
            Data data = new SampleData(Path.of(file));

            double msq;
            int cnt = 0;
            do {
                network.backpropagation(data);
                msq = network.calculateMeanSquaredErrorFromData(data);
                if (msq < 0.0015) break;
                data.shuffle();
                cnt++;
                if(cnt % 100 == 0) {
                    System.out.printf("At %dth iteration. MSE is %f\n", cnt, msq);
                }
            } while (cnt < epochs);
            System.out.println("Finished after " + cnt  + " epochs with msq: " + msq);

            train.setEnabled(false);
            classify.setEnabled(true);
        });
        classSouth.add(train);

        JButton classClearButton = new JButton("Clear");
        classClearButton.addActionListener(l -> {
          results.setText("Letter: -\tAlpha: -\tBeta: -\tGamma: -\tDelta: -\tEta: -\t".replaceAll("\t", "   "));
          classBoard.clearBoard();
        });
        classSouth.add(classClearButton);

        classify.addActionListener(l -> {
            List<Double> input = new ArrayList<>();
            new Preprocessing(classBoard.getPoints(), M).process().forEach(p -> {
                input.add(p.getX());
                input.add(p.getY());
            });
            RealVector networkResults = network.calculate(input);
            StringBuilder bobTheBuilder = new StringBuilder();
            bobTheBuilder.append("Result: ").append(letters[networkResults.getMaxIndex()]).append("\t\t");
            for(int i = 0; i < letters.length; i++)
                bobTheBuilder.append(letters[i]).append(": ").append(String.format("%.3f", networkResults.getEntry(i))).append("\t");
            results.setText(bobTheBuilder.toString().replaceAll("\t", "   "));
        });
        classSouth.add(classify);

        classPane.add(classSouth, BorderLayout.SOUTH);
        classPane.addMouseMotionListener(new BoardMouseMotionListener(classBoard));

        tabbedPane.addTab("Classify", classPane);

        tabbedPane.setSelectedIndex(1);
        add(tabbedPane);
    }

    private void resetData() {
        data.clear();
        for (String l : letters)
            data.put(l, new ArrayList<>());
    }


    public static void main(String[] args) {
        if(args.length != 5) {
            System.err.println("Wrong number of arguments! I need: type, M, epochs, msq, file");
            System.exit(0);
        }

        String type = args[0];
        int M = Integer.parseInt(args[1]);
        int epochs = Integer.parseInt(args[2]);
        double msq = Double.parseDouble(args[3]);
        String file = args[4];

        SwingUtilities.invokeLater(() -> {
            new Driver(type, M, epochs, msq, file).setVisible(true);
                });
    }
}

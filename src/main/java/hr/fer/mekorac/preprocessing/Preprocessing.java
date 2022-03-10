package hr.fer.mekorac.preprocessing;

import hr.fer.mekorac.preprocessing.point.Point;

import java.util.ArrayList;
import java.util.List;

public class Preprocessing {
    private int len;
    private List<Double> xs = new ArrayList<>();
    private List<Double> ys = new ArrayList<>();

    private int M;

    public Preprocessing(List<Point<Integer>> ps, int M) {
        len = ps.size();
        for (Point<Integer> p : ps) {
            xs.add((double) p.getX());
            ys.add((double) p.getY());
        }

        this.M = M;
    }

    public List<Point<Double>> process() {
        double x_mean = xs.stream().mapToDouble(x -> x).sum() / xs.size();
        double y_mean = ys.stream().mapToDouble(y -> y).sum() / ys.size();
        for(int i = 0; i < len; i++) {
            xs.set(i, xs.get(i) - x_mean);
            ys.set(i, ys.get(i) - y_mean);
        }

        double maxx = 0;
        for(double x : xs) if(x > maxx) maxx = x;

        double maxy = 0;
        for(double y : ys) if(y > maxy) maxy = y;

        double max = Math.max(maxx, maxy);
        for(int i = 0; i < len; i++) {
            xs.set(i, xs.get(i) / max);
            ys.set(i, ys.get(i) / max);
        }

        double[] D = new double[len];
        D[0] = 0;
        for(int i = 1; i < len; i++) {
            D[i] = dist(xs.get(i), xs.get(i - 1), ys.get(i), ys.get(i - 1)) + D[i - 1];
        }
        double total_D = D[D.length - 1];

        List<Point<Double>> finalPoints = new ArrayList<>();
        for(int k = 0, j = 0; k < M; k++) {
            double dist = k * total_D / (M - 1);
            for(; j < len; j++) {
                if(D[j] >= dist) {
                    finalPoints.add(new Point<>(xs.get(j), ys.get(j)));
                    break;
                }
            }
        }
        while(finalPoints.size() < M) finalPoints.add(new Point<>(xs.get(len - 1), ys.get(len - 1))); //in case there is a place unfilled

        return finalPoints;
    }

    private double dist(double x1, double y1, double x2, double y2) {
        return Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    }
}

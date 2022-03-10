package hr.fer.mekorac.preprocessing;

import hr.fer.mekorac.preprocessing.point.Point;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class Board extends JPanel {
    private List<Point<Integer>> points;
    private final int yOffset = 20;
    private final int xOffset = 0;


    public Board(List<Point<Integer>> points) {
        this.points = points;
        setBackground(Color.WHITE);
    }

    public void setPoints(List<Point<Integer>> points) {
        this.points = points;
    }

    public List<Point<Integer>> getPoints() {
        return points;
    }

    public void clearBoard() {
        this.setPoints(new ArrayList<>());
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (int i = 0; i < points.size() - 1; i++) {
            g.drawLine(points.get(i).getX() - xOffset, points.get(i).getY()-yOffset, points.get(i + 1).getX()-xOffset, points.get(i + 1).getY() - yOffset);
        }
    }
}

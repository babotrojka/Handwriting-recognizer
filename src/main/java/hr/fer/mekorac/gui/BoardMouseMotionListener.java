package hr.fer.mekorac.gui;

import hr.fer.mekorac.preprocessing.Board;
import hr.fer.mekorac.preprocessing.point.Point;

import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.util.List;

public class BoardMouseMotionListener implements MouseMotionListener {
    Board board;

    public BoardMouseMotionListener(Board board) {
        this.board = board;
    }

    @Override
    public void mouseDragged(MouseEvent e) {
        //System.out.println("Mouse on " + e.getX() + ", " + e.getY());
        board.getPoints().add(new Point<>(e.getX(), e.getY()));
        board.repaint();
    }

    @Override
    public void mouseMoved(MouseEvent e) {
    }

}

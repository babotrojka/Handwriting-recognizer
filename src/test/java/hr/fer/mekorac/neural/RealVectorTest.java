package hr.fer.mekorac.neural;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.junit.jupiter.api.Test;

public class RealVectorTest {

    @Test
    public void outerProductTest() {
        RealVector r1 = MatrixUtils.createRealVector(new double[] {1, 2, 3});
        RealVector r2 = MatrixUtils.createRealVector(new double[] {2, 3});
        System.out.println(r1.outerProduct(r2));
    }
}

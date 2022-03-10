package hr.fer.mekorac.data;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

public class SampleDataTest {
    String file = "dataset.txt";

    @Test
    public void test1() {
        Assertions.assertEquals(100, new SampleData(Path.of(file)).getDataSize());
    }
}

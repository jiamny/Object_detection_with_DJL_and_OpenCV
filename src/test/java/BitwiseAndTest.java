import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;

import static org.opencv.core.Core.bitwise_and;
import static org.opencv.highgui.HighGui.*;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.*;

public class BitwiseAndTest {

    @Test
    public void testBitwiseAnd() {
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");

        String img_path = "./data/self_driving/images/solidWhiteCurve.jpg";

        Mat src = imread(img_path, IMREAD_COLOR);

        Mat src_gray = new Mat();
        cvtColor(src, src_gray, COLOR_BGR2GRAY);
        Mat dst = new Mat();
        threshold(src_gray, dst, 150, 255, 1);
        Mat  res = new Mat();
        //bitwise_and(src, dst, res);
        bitwise_and(src_gray, dst, res);
        imshow("BW", res);
        waitKey(0);

        // need color image
        bitwise_and( src, src, res, dst );
        imshow("Color", res);
        waitKey(0);

        destroyAllWindows();
    }
}

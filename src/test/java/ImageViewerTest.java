import org.junit.jupiter.api.Test;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import com.github.jiamny.Utils.ImageViewer;
import org.opencv.core.Mat;

public class ImageViewerTest {

    @Test
    public void testImageViewer() throws InterruptedException {
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");

        String img_path = "./data/self_driving/images/solidWhiteCurve.jpg";

        Mat in_image = imread(img_path, IMREAD_COLOR);
        //cvtColor(in_image, in_image, COLOR_BGR2RGB);

        ImageViewer.show(in_image);
        Thread.sleep(500);
    }
}

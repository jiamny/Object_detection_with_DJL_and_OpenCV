import com.github.jiamny.Utils.ImageHelper;
import com.github.jiamny.Utils.ImageUtils;
import org.opencv.core.Core;
import org.testng.annotations.Test;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import com.github.jiamny.Utils.ImageViewer;
import org.opencv.core.Mat;

import java.awt.image.BufferedImage;

public class ImageViewerTest {

    @Test
    public void testImageViewer() throws InterruptedException {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");

        String img_path = "./data/self_driving/images/solidWhiteCurve.jpg";

        Mat in_image = imread(img_path, IMREAD_COLOR);
        //cvtColor(in_image, in_image, COLOR_BGR2RGB);

        ImageViewer.show(in_image);
        Thread.sleep(1500);
    }

    @Test
    public void testBufferedImageViewer() throws InterruptedException {
        //System.load("C:\\Program Files\\Opencv4\\java\\x64\\opencv_java454.dll");
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");

        try {
            String[] labels = {"solidYellowLeft", "solidWhiteRight"};

            Mat img1 = imread("./data/self_driving/images/solidYellowLeft.jpg", IMREAD_COLOR);
            Mat img2 = imread("./data/self_driving/images/solidWhiteRight.jpg", IMREAD_COLOR);
            //cvtColor(in_image, in_image, COLOR_BGR2RGB);

            BufferedImage[] imgs = {ImageHelper.matToBufferedImage(img1), ImageHelper.matToBufferedImage(img2)};
            BufferedImage img = ImageUtils.showImages(imgs, labels, 400, 300);

            ImageViewer.show(ImageHelper.bufferedImage2Mat(img), "");
            Thread.sleep(5000);
        }catch(Exception e) {
            e.printStackTrace();
        }
    }
}

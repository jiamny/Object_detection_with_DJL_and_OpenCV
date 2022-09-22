

import com.github.jiamny.Utils.VideoPlay;
import org.junit.jupiter.api.Test;

import static org.opencv.highgui.HighGui.destroyAllWindows;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

public class ViedoPlayTest {

    @Test
    public void testVideoPlay() {
        System.load("/usr/local/share/java/opencv4/libopencv_java460.so");

        String f = "./data/self_driving/videos/solidWhiteRight.mp4";
        boolean useImshow = false;
        String tlt = "Traffic lane detection";
        VideoPlay vp =new VideoPlay();
        if( ! useImshow )
            vp.initGUI(tlt);

        Mat currentImage = new Mat();
        try {
            VideoCapture capture = new VideoCapture();
            capture.open(f);

            if (capture.isOpened()) {
                while (true) {
                    capture.read(currentImage);
                    if( ! currentImage.empty() )
                        vp.displayImage(currentImage, tlt, useImshow);
                    else
                        break;
                }
            }
            if( useImshow ) destroyAllWindows();
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("Failed start the grabber.");
        }
        System.exit(0);
    }
}

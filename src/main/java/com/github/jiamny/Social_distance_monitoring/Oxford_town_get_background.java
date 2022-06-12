package com.github.jiamny.Social_distance_monitoring;

import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

import static org.opencv.core.Core.convertScaleAbs;
import static org.opencv.core.CvType.CV_32FC3;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.accumulateWeighted;

public class Oxford_town_get_background {
    static {
        // load the OpenCV native library
        // System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");
    }

    public static void main(String[] args) {
        String imgPath = "data/videos/OxfordTownCentreDataset.avi";

        String current_dir = System.getProperty("user.dir");
        System.out.println(current_dir);

        // load video
        VideoCapture capture = new VideoCapture(imgPath);
        if (!capture.isOpened()) {
            System.out.println("Unable to open this file");
            System.exit(-1);
        }

        Mat res2 = new Mat();
        while (true) {
            Mat frame = new Mat();
            capture.read(frame);
            if (frame.empty())
                break;
            Mat img_bkgd = new Mat();

            frame.convertTo(img_bkgd, CV_32FC3);

            accumulateWeighted(frame, img_bkgd, 0.01);
            convertScaleAbs(img_bkgd, res2);

            HighGui.imshow("When you feel the background is good enough, press ESC to terminate and save the background.", res2);
            int keyboard = HighGui.waitKey(20);
            System.out.println(keyboard);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }
        }

        imwrite("output/oxford_town_background.png", res2);
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}

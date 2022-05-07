package com.github.jiamny.YOLO_object_detection;

import org.opencv.core.*;
import org.opencv.dnn.DetectionModel;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import static com.github.jiamny.Utils.ImageHelper.addTextToBox;

public class YOLOv4ObjectDetection {

    static {
        // no opencv_java455 in java.library.path: [/usr/java/packages/lib, /usr/lib/x86_64-linux-gnu/jni,
        // /lib/x86_64-linux-gnu, /usr/lib/x86_64-linux-gnu, /usr/lib/jni, /lib, /usr/lib]
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");
    }

    public static void main(String[] args) throws IOException {
        Mat img = Imgcodecs.imread("data/images/test.jpg");

        List<String> classes = Files.readAllLines(Paths.get("data/coco.names"));

        Net net = Dnn.readNetFromDarknet("data/models/yolov4.cfg", "data/models/yolov4.weights");

        DetectionModel model = new DetectionModel(net);
        model.setInputParams(1 / 255.0, new Size(416, 416), new Scalar(0), true);

        MatOfInt classIds = new MatOfInt();
        MatOfFloat scores = new MatOfFloat();
        MatOfRect boxes = new MatOfRect();
        model.detect(img, classIds, scores, boxes, 0.6f, 0.4f);

        for (int i = 0; i < classIds.rows(); i++) {
            Rect box = new Rect(boxes.get(i, 0));
            Imgproc.rectangle(img, box, new Scalar(0, 0, 255), 1);

            int classId = (int) classIds.get(i, 0)[0];
            double score = scores.get(i, 0)[0];
            String text = String.format("%s: %.2f", classes.get(classId), score);
            addTextToBox(img, text, new Scalar(255, 255, 255), Imgproc.FONT_HERSHEY_SIMPLEX, new Scalar(0, 0, 255),
                    new Point(box.x, box.y - 1), 0.5, 1);
        }

        HighGui.imshow("Image", img);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
        System.exit(0);
    }
}

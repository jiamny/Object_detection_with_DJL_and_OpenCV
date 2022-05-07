package com.github.jiamny.Utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;

import static org.opencv.core.Core.FILLED;
import static org.opencv.imgproc.Imgproc.*;

public class ImageHelper {
    /**
     * Convert a Mat object (OpenCV) in the corresponding Image
     *
     * @param frame
     *            the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    public static Image mat2Image(Mat frame) {
        try {
            return ImageFactory.getInstance().fromImage (HighGui.toBufferedImage(frame));
        } catch (Exception e) {
            System.err.println("Cannot convert the Mat obejct: " + e);
            return null;
        }
    }

    /**
     * put object label to the detected object box
     *
     * @param image
     *            the {@link Mat} representing the current frame
     * @param label label text color
     * @param txtcolor label text color
     * @param fontface  label text font face
     * @param bkcolor  label text background color
     * @param leftpt   Point to box top left point
     * @param txtscale label text scale
     * @param thickness line thinckness
     */
    public static void addTextToBox(Mat image, String label, Scalar txtcolor, int fontface, Scalar bkcolor, Point leftpt,
                                   double txtscale, int thickness) {
        int baseline = 0;
        int [] baselines = {0};
        Size text = getTextSize(label, fontface, txtscale, thickness, baselines);
        rectangle(image, new Point(leftpt.x +0, leftpt.y + baseline),
                         new Point( leftpt.x + text.width + 5, leftpt .y - text.height), bkcolor, FILLED);
        putText(image, label, leftpt, fontface, txtscale, txtcolor, thickness, LINE_AA);
    }
}

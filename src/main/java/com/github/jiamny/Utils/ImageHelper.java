package com.github.jiamny.Utils;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDArray;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.minMaxLoc;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.imgproc.Imgproc.*;

public class ImageHelper {
    /**
     * Convert a Mat object (OpenCV) in the corresponding Image
     *
     * @param frame the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    public static Image mat2DjlImage(Mat frame) {
        try {
            return ImageFactory.getInstance().fromImage(HighGui.toBufferedImage(frame));
        } catch (Exception e) {
            System.err.println("Cannot convert the Mat obejct: " + e);
            return null;
        }
    }

    /**
     * put object label to the detected object box
     *
     * @param image     the {@link Mat} representing the current frame
     * @param label     label text color
     * @param txtcolor  label text color
     * @param fontface  label text font face
     * @param bkcolor   label text background color
     * @param leftpt    Point to box top left point
     * @param txtscale  label text scale
     * @param thickness line thinckness
     */
    public static void addTextToBox(Mat image, String label, Scalar txtcolor, int fontface, Scalar bkcolor, Point leftpt,
                                    double txtscale, int thickness) {
        int baseline = 0;
        int[] baselines = {0};
        Size text = getTextSize(label, fontface, txtscale, thickness, baselines);
        rectangle(image, new Point(leftpt.x + 0, leftpt.y + baseline),
                new Point(leftpt.x + text.width + 5, leftpt.y - text.height), bkcolor, FILLED);
        putText(image, label, leftpt, fontface, txtscale, txtcolor, thickness, LINE_AA);
    }

    public static Mat bufferedImage2Mat(BufferedImage image) throws IOException {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ImageIO.write(image, "jpg", byteArrayOutputStream);
        byteArrayOutputStream.flush();
        return Imgcodecs.imdecode(new MatOfByte(byteArrayOutputStream.toByteArray()), Imgcodecs.IMREAD_UNCHANGED);
    }

    public static Mat ndarrayToMat(NDArray img_cal) {
        long[] shape = img_cal.getShape().getShape();
        Mat mat = null;
        if (shape.length > 2) {
            int h = (int) (img_cal.getShape().getShape()[0]);
            int w = (int) (img_cal.getShape().getShape()[1]);
            int channel = (int) (img_cal.getShape().getShape()[2]);

            if(channel > 3)
                mat = new Mat(h, w, CvType.CV_8UC4);
            else
                mat = new Mat(h, w, CvType.CV_8UC3);

            byte[] data = img_cal.toByteArray();
            mat.put(0, 0, data);
        } else {
            int h = (int) (img_cal.getShape().getShape()[0]);
            int w = (int) (img_cal.getShape().getShape()[1]);
            mat = new Mat(h, w, CvType.CV_8UC1);

            byte[] data = img_cal.toByteArray();
            mat.put(0, 0, data);
        }
        return mat;
    }

    public static BufferedImage matToBufferedImage(Mat matrix) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (matrix.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = matrix.channels() * matrix.cols() * matrix.rows();
        byte[] buffer = new byte[bufferSize];
        matrix.get(0, 0, buffer); // get all the pixels
        BufferedImage image = new BufferedImage(matrix.cols(), matrix.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);
        return image;
    }

    /**
     * Convert `Mat` to one where pixels are represented as 8 bit unsigned integers (`CV_8U`). It
     * creates a copy of the input image.
     *
     * @param src input image.
     * @return copy of the input with pixels values represented as 8 bit unsigned integers.
     */
    public static Mat toMat8U(Mat src, boolean doScaling) {
        //double[] min = {Double.MAX_VALUE};
        //double[] max = {Double.MIN_VALUE};
        Core.MinMaxLocResult minMaxLoc = minMaxLoc(src, null);
        double scale = 1d;
        double offset = 0d;
        if (doScaling) {
            double s = 255d / (minMaxLoc.maxVal - minMaxLoc.minVal);
            scale = s;
            offset = -1 * minMaxLoc.minVal * s;
        }

        Mat dest = new Mat();
        src.convertTo(dest, CV_8U, scale, offset);
        return (dest);
    }
}

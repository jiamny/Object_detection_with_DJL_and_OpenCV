package com.github.jiamny.YOLO_object_detection;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import static com.github.jiamny.Utils.ImageHelper.addTextToBox;

public class YOLOv3ObjectDetection {
    private final String model_weights;
    private final String model_config;
    private final String class_file_name_dir;
    private final String output_path;
    private final List<String> classes;
    private final List<String> output_layers;
    private final String input_path;
    private List<String> layer_names;
    private Net network;
    private final Size size;
    private Integer height;
    private Integer width;
    //   private Integer channels;
    private final Scalar mean;
    private Mat image;
    //   private Mat blob;
    private List<Mat> outputs;
    private final List<Rect2d> boxes;
    private final List<Float> confidences;
    private final List<Integer> class_ids;
    private final String outputFileName;
    private final boolean save;
    private boolean errors;

    static {
        System.load("/usr/local/share/java/opencv4/libopencv_java460.so");
    }

    public YOLOv3ObjectDetection(String inputPath, String outputPath, Integer image_size, String outputFileName, String current_dir) {

        this.input_path = inputPath;
        this.output_path = outputPath;
        this.outputFileName = outputFileName;
        boxes = new ArrayList<>();
        classes = new ArrayList<>();
        class_ids = new ArrayList<>();
        layer_names = new ArrayList<>();
        confidences = new ArrayList<>();
        double[] means = {0.0, 0.0, 0.0};
        mean = new Scalar(means);
        output_layers = new ArrayList<>();
        size = new Size(image_size, image_size);
        // get yolov3-608.weights and yolov3-608.cfg form https://chowdera.com/2021/08/20210810105701286e.html#google_vignette
        model_weights = current_dir + "/data/models/yolov3_608.weights";
        model_config = current_dir + "/data/models/yolov3_608.cfg";
        class_file_name_dir = current_dir + "/data/coco.names";
        save = true;
        errors = false;
    }

    static int argmax(List<Float> array) {
        float max = array.get(0);
        int re = 0;
        for (int i = 1; i < array.size(); i++) {
            if (array.get(i) > max) {
                max = array.get(i);
                re = i;
            }
        }
        return re;
    }

    private void setClasses() {
        try {
            File f = new File(class_file_name_dir);
            Scanner reader = new Scanner(f);
            while (reader.hasNextLine()) {
                String class_name = reader.nextLine();
                classes.add(class_name);
            }
        } catch (FileNotFoundException e) {
            errors = true;
        }
    }

    private void setNetwork() {
        network = Dnn.readNet(model_weights, model_config);
    }

    private void setUnconnectedLayers() {

        for (Integer i : network.getUnconnectedOutLayers().toList()) {
            output_layers.add(layer_names.get(i - 1));
        }
    }

    private void setLayerNames() {
        layer_names = network.getLayerNames();
    }

    private void loadImage() {
        System.out.println(input_path);
        Mat img = Imgcodecs.imread(input_path);
        Mat resizedImage = new Mat();
        Imgproc.resize(img, resizedImage, size, 0.9, 0.9);
        height = resizedImage.height();
        width = resizedImage.width();
        //       channels = resizedImage.channels();
        image = resizedImage;
    }

    private void detectObject() {
        Mat blob_from_image = Dnn.blobFromImage(image, 0.00392, size, mean, true, false);
        network.setInput(blob_from_image);
        outputs = new ArrayList<>();
        network.forward(outputs, output_layers);
        //blob = blob_from_image;
    }

    private void getBoxDimensions() {
        for (Mat output : outputs) {

            for (int i = 0; i < output.height(); i++) {
                Mat row = output.row(i);
                MatOfFloat temp = new MatOfFloat(row);
                List<Float> detect = temp.toList();
                List<Float> score = detect.subList(5, 85);
                int class_id = argmax(score);
                float conf = score.get(class_id);
                if (conf >= 0.4) {
                    int center_x = (int) (detect.get(0) * width);
                    int center_y = (int) (detect.get(1) * height);
                    int w = (int) (detect.get(2) * width);
                    int h = (int) (detect.get(3) * height);
                    int x = (center_x - w / 2);
                    int y = (center_y - h / 2);
                    Rect2d box = new Rect2d(x, y, w, h);
                    boxes.add(box);
                    confidences.add(conf);
                    class_ids.add(class_id);
                }
            }
        }
    }

    private void drawLabels() {
        double[] rgb = new double[]{0, 0, 255};
        Scalar color = new Scalar(rgb);
        MatOfRect2d mat = new MatOfRect2d();
        mat.fromList(boxes);
        MatOfFloat confidence = new MatOfFloat();
        confidence.fromList(confidences);
        MatOfInt indices = new MatOfInt();
        int font = Imgproc.FONT_HERSHEY_PLAIN;
        Dnn.NMSBoxes(mat, confidence, (float) (0.4), (float) (0.4), indices);
        List<Integer> indices_list = indices.toList();

        for (int i = 0; i < boxes.size(); i++) {
            if (indices_list.contains(i)) {
                if (save) {
                    Rect2d box = boxes.get(i);
                    Point x_y = new Point(box.x, box.y);
                    Point w_h = new Point(box.x + box.width, box.y + box.height);
                    Point text_point = new Point(box.x, box.y - 1);
                    Imgproc.rectangle(image, w_h, x_y, color);
                    String label = classes.get(class_ids.get(i));

                    //Imgproc.putText(image, label, text_point, font, 1, color);
                    addTextToBox(image, label, new Scalar(255, 255, 255), font, color,
                            text_point, 1, 1);
                }
            }
        }
        if (save) {
            Imgcodecs.imwrite(output_path + "/" + outputFileName + ".png", image);
        }
    }

    public boolean loadPipeline() {
        try {
            setNetwork();
            setClasses();
            setLayerNames();
            setUnconnectedLayers();
            loadImage();
            detectObject();
            getBoxDimensions();
            drawLabels();
        } catch (Exception e) {
            errors = true;
        }
        return errors;
    }

    public Mat getImage() {
        return image;
    }

    public static void main(String[] args) {

        String current_dir = System.getProperty("user.dir");
        System.out.println(current_dir);
        String inputPath = "data/images/test.jpg";
        String outputPath = "output";
        Integer image_size = 608;
        String outputFileName = "detected_test.jpg";
        YOLOv3ObjectDetection ydt = new YOLOv3ObjectDetection(inputPath, outputPath, image_size, outputFileName, current_dir);

        if (!ydt.loadPipeline()) {
            Mat predImg = ydt.getImage();

            HighGui.imshow("Detected image", predImg);
            HighGui.waitKey(0);
            HighGui.destroyAllWindows();
        }

        System.exit(0);
    }
}

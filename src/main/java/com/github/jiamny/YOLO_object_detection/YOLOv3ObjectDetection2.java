package com.github.jiamny.YOLO_object_detection;

import org.opencv.core.*;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.opencv.videoio.VideoCapture;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

class PredictedBBox {
    private final ArrayList<Rect2d> bboxes = new ArrayList<>();
    private final ArrayList<Float> confidences = new ArrayList<>();
    private final ArrayList<Integer> class_ids = new ArrayList<>();

    public void setBbox(Rect2d rect) {
        bboxes.add(rect);
    }

    public void setConfidence(float cf) {
        confidences.add(cf);
    }

    public void setClass_id(int id) {
        class_ids.add(id);
    }

    public ArrayList<Rect2d> getBbox() {
        return bboxes;
    }

    public ArrayList<Float> getConfidence() {
        return confidences;
    }

    public ArrayList<Integer> getClass_id() {
        return class_ids;
    }
}

public class YOLOv3ObjectDetection2 {

    static {
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");
    }

    private final List<String> cocoLabels;

    private final Net net;

    public YOLOv3ObjectDetection2(List<String> cocoCls, Net network) {
        cocoLabels = cocoCls;
        net = network;
    }

    private int argmax(List<Float> array) {
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

    public Mat detectObjectOnImage(Mat img) {

        // generate radnom color in order to draw bounding boxes
        Random random = new Random();
        ArrayList<Scalar> colors = new ArrayList<>();
        for (int i = 0; i < cocoLabels.size(); i++) {
            colors.add(new Scalar(new double[]{random.nextInt(255), random.nextInt(255), random.nextInt(255)}));
        }

        //  -- determine  the output layer names that we need from YOLO
        List<String> layerNames = net.getLayerNames();
        List<String> outputLayers = new ArrayList<>();
        for (Integer i : net.getUnconnectedOutLayers().toList()) {
            outputLayers.add(layerNames.get(i - 1));
        }

        PredictedBBox result = forwardImageOverNetwork(img, net, outputLayers);

        ArrayList<Rect2d> boxes = result.getBbox();
        ArrayList<Float> confidences = result.getConfidence();
        ArrayList<Integer> class_ids = result.getClass_id();

        // -- Now , do so-called “non-maxima suppression”
        //Non-maximum suppression is performed on the boxes whose confidence is equal to or greater than the threshold.
        // This will reduce the number of overlapping boxes:
        MatOfInt indices = getBBoxIndicesFromNonMaximumSuppression(boxes, confidences);

        //-- Finally, go over indices in order to draw bounding boxes on the image:
        return drawBoxesOnTheImage(img,
                indices,
                boxes,
                cocoLabels,
                class_ids,
                colors);
    }

    private PredictedBBox forwardImageOverNetwork(Mat img, Net dnnNet, List<String> outputLayers) {
        // --We need to prepare some data structure  in order to store the data returned by the network  (ie, after Net.forward() call))
        // So, Initialize our lists of detected bounding boxes, confidences, and  class IDs, respectively
        // This is what this method will return:

        PredictedBBox result = new PredictedBBox();

        // -- The input image to a neural network needs to be in a certain format called a blob.
        //  In this process, it scales the image pixel values to a target range of 0 to 1 using a scale factor of 1/255.
        // It also resizes the image to the given size of (416, 416) without cropping
        // Construct a blob from the input image and then perform a forward  pass of the YOLO object detector,
        // giving us our bounding boxes and  associated probabilities:

        Mat blob_from_image = Dnn.blobFromImage(img, 1 / 255.0, new Size(416, 416), // Here we supply the spatial size that the Convolutional Neural Network expects.
                new Scalar(new double[]{0.0, 0.0, 0.0}), true, false);
        dnnNet.setInput(blob_from_image);

        // -- the output from network's forward() method will contain a List of OpenCV Mat object, so lets prepare one
        List<Mat> outputs = new ArrayList<>();

        // -- Finally, let pass forward throught network. The main work is done here:
        dnnNet.forward(outputs, outputLayers);

        // --Each output of the network outs (ie, each row of the Mat from 'outputs') is represented by a vector of the number
        // of classes + 5 elements.  The first 4 elements represent center_x, center_y, width and height.
        // The fifth element represents the confidence that the bounding box encloses the object.
        // The remaining elements are the confidence levels (ie object types) associated with each class.
        // The box is assigned to the category corresponding to the highest score of the box:

        for (Mat output : outputs) {
            // loop over each of the detections. Each row is a candidate detection,
            // System.out.println("Output.rows(): " + output.rows() + ", Output.cols(): " + output.cols());
            for (int i = 0; i < output.rows(); i++) {
                Mat row = output.row(i);
                List<Float> detect = new MatOfFloat(row).toList();
                List<Float> score = detect.subList(5, output.cols());
                int class_id = argmax(score); // index maximalnog elementa liste
                float conf = score.get(class_id);
                if (conf >= 0.5) {
                    int center_x = (int) (detect.get(0) * img.cols());
                    int center_y = (int) (detect.get(1) * img.rows());
                    int width = (int) (detect.get(2) * img.cols());
                    int height = (int) (detect.get(3) * img.rows());
                    int x = (center_x - width / 2);
                    int y = (center_y - height / 2);
                    Rect2d box = new Rect2d(x, y, width, height);

                    result.setBbox(box);
                    result.setConfidence(conf);
                    result.setClass_id(class_id);
                }
            }
        }
        return result;
    }

    private MatOfInt getBBoxIndicesFromNonMaximumSuppression(ArrayList<Rect2d> boxes, ArrayList<Float> confidences) {
        MatOfRect2d mOfRect = new MatOfRect2d();
        mOfRect.fromList(boxes);
        MatOfFloat mfConfs = new MatOfFloat(Converters.vector_float_to_Mat(confidences));
        MatOfInt result = new MatOfInt();
        Dnn.NMSBoxes(mOfRect, mfConfs, (float) (0.6), (float) (0.5), result);
        return result;
    }

    private Mat drawBoxesOnTheImage(Mat img,
                                    MatOfInt indices,
                                    ArrayList<Rect2d> boxes,
                                    List<String> cocoLabels,
                                    ArrayList<Integer> class_ids,
                                    ArrayList<Scalar> colors) {
        //Scalar color = new Scalar( new double[]{255, 255, 0});
        List<Integer> indices_list = indices.toList();
        for (int i = 0; i < boxes.size(); i++) {
            if (indices_list.contains(i)) {
                Rect2d box = boxes.get(i);
                Point x_y = new Point(box.x, box.y);
                Point w_h = new Point(box.x + box.width, box.y + box.height);
                Point text_point = new Point(box.x, box.y - 5);
                Imgproc.rectangle(img, w_h, x_y, colors.get(class_ids.get(i)), 1);
                String label = cocoLabels.get(class_ids.get(i));
                Imgproc.putText(img, label, text_point, Imgproc.FONT_HERSHEY_SIMPLEX, 1, colors.get(class_ids.get(i)), 2);
            }
        }
        return img;
    }

    public static void main(String[] args) {

        String current_dir = System.getProperty("user.dir");
        System.out.println(current_dir);
        String imgPath = "data/videos/OxfordTownCentreDataset.avi";
        String cfgPath = "data/models/yolov3_608.cfg";
        String wgtPath = "data/models/yolov3_608.weights";
        String clsPath = "data/coco.names";
        YOLOv3ObjectDetection2 tgt2 = null;

        try {
            List<String> cocoCls = new ArrayList<>();
            //  load the COCO class labels
            Scanner scan = new Scanner(new FileReader(clsPath));
            while (scan.hasNextLine()) {
                cocoCls.add(scan.nextLine());
            }

            //  load our YOLO object detector trained on COCO dataset
            Net net = Dnn.readNetFromDarknet(cfgPath, wgtPath);

            // YOLO on GPU:
            // dnnNet.setPreferableBackend(Dnn.DNN_BACKEND_CUDA);
            // dnnNet.setPreferableTarget(Dnn.DNN_TARGET_CUDA);
            tgt2 = new YOLOv3ObjectDetection2(cocoCls, net);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        }

        // load video
        VideoCapture capture = new VideoCapture(imgPath);
        if (!capture.isOpened()) {
            System.out.println("Unable to open this file");
            System.exit(-1);
        }

        while (true) {
            Mat frame = new Mat();
            capture.read(frame);
            if (frame.empty())
                break;

            // detect object with YOLO3
            Mat img = tgt2.detectObjectOnImage(frame);
            HighGui.imshow("Frame", img);

            int keyboard = HighGui.waitKey(5);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }
        }

        HighGui.destroyAllWindows();

        System.exit(0);
    }
}

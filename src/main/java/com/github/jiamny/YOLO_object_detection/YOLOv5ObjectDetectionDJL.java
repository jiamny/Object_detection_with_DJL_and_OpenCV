package com.github.jiamny.YOLO_object_detection;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.YoloV5Translator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.io.IOException;
import java.nio.file.Paths;

import static com.github.jiamny.Utils.ImageHelper.*;
import static org.opencv.imgproc.Imgproc.resize;

public class YOLOv5ObjectDetectionDJL {

    static {
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
        //System.load("C:\\Program Files\\Opencv4\\java\\x64\\opencv_java454.dll");
    }

    static Rect rect = new Rect();
    static Scalar boxcolor = new Scalar(0, 0, 255);
    static Scalar txtcolor = new Scalar(255, 255, 255);

    static void detect(Mat frame, ZooModel<Image, DetectedObjects> model) throws IOException,
            ModelNotFoundException, MalformedModelException, TranslateException {

        Image img = mat2DjlImage(frame);
        if (img == null) {
            System.out.println("Error: convert CV frame to Image");
            System.exit(-1);
        }

        long startTime = System.currentTimeMillis();
        try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {

            DetectedObjects results = predictor.predict(img);
            for (DetectedObject obj : results.<DetectedObject>items()) {
                BoundingBox bbox = obj.getBoundingBox();
                Rectangle rectangle = bbox.getBounds();
                String showText = String.format("%s: %.2f", obj.getClassName(), obj.getProbability());
                rect.x = (int) rectangle.getX();
                rect.y = (int) rectangle.getY();
                rect.width = (int) rectangle.getWidth();
                rect.height = (int) rectangle.getHeight();
                // add rectangle box
                Imgproc.rectangle(frame, rect, boxcolor, 1);
                // put object label
                addTextToBox(frame, showText, txtcolor, Imgproc.FONT_HERSHEY_COMPLEX, boxcolor,
                        new Point(rect.x, rect.y - 1), 0.4, 1);
            }
        }
        System.out.printf("%.2f%n", 1000.0 / (System.currentTimeMillis() - startTime));
    }

    public static void main(String[] args) {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        boolean useOnnx = false;

        Translator<Image, DetectedObjects> translator = YoloV5Translator.builder().optSynsetArtifactName("coco_classes.txt").build();
        Criteria<Image, DetectedObjects> criteria;

        if( ! useOnnx ) {
            criteria = Criteria.builder()
                    .setTypes(Image.class, DetectedObjects.class)
                    .optDevice(Device.cpu())
                    .optModelPath(Paths.get("/media/hhj/localssd/DL_data/weights/yolo5"))
                    //.optModelUrls(YOLOv5ObjectDetectionDJL.class.getResource("/yolov5s").getPath())
                    .optModelName("yolov5s.torchscript.pt")
                    .optTranslator(translator)
                    .optEngine("PyTorch")
                    .build();
        } else {
            criteria = Criteria.builder()
                    .setTypes(Image.class, DetectedObjects.class)
                    .optDevice(Device.cpu())
                    .optModelPath(Paths.get("/media/hhj/localssd/DL_data/weights/yolo5"))
                    //.optModelUrls(YOLOv5ObjectDetectionDJL.class.getResource("/yolov5").getPath())
                    .optModelName("yolov5s.onnx")
                    .optTranslator(translator)
                    .optEngine("OnnxRuntime")
                    .build();
        }

        try(ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            // camera
            //VideoCapture cap = new VideoCapture(CAP_ANY);
            VideoCapture cap = new VideoCapture("/media/hhj/localssd/DL_data/videos/usa-street.mp4"); //car_chase_01.mp4");
            if(!cap.isOpened()) {
                System.err.println("Error opening video file");
                cap.release();
                System.exit(-1);
            } else {

                Mat frame = new Mat();          // output mat
                boolean flag = cap.read(frame); // read current frame

                while (flag) {
                    resize(frame, frame, new Size(640, 640));
                    //System.out.println("r: " + frame.rows() + ", c: " + frame.cols());
                    detect(frame, model);
                    HighGui.imshow("yolov5_djl", frame);
                    int r = HighGui.waitKey(20);

                    if( r == 27 || r == 81 || r == 113 )  // ESC, Q/q
                        break;
                    flag = cap.read(frame);
                }
            }

        } catch (RuntimeException | ModelException | TranslateException | IOException e) {
            e.printStackTrace();
        }
        HighGui.destroyAllWindows();
        System.out.println("Done!");

        System.exit(0);
    }
}


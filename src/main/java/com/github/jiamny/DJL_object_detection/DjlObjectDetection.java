package com.github.jiamny.DJL_object_detection;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static com.github.jiamny.Utils.ImageHelper.addTextToBox;
import static com.github.jiamny.Utils.ImageHelper.mat2Image;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.resize;

/**
 * An example of inference using an object detection model.
 */
public class DjlObjectDetection {

    static {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // Make sure that you loaded your corresponding opencv java .so file.
        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");
    }

    private static Mat image = null;
    private static String backbone = "";

    public DjlObjectDetection() {}

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        DetectedObjects detection = DjlObjectDetection.predict();

        int width = image.width();
        int height = image.height();

        Rect rect = new Rect();
        Scalar boxcolor = new Scalar(255, 0, 0);
        Scalar txtcolor = new Scalar(255, 255, 255);
        System.out.println(">>> " + detection.<DetectedObjects.DetectedObject>items());

        for (DetectedObjects.DetectedObject obj : detection.<DetectedObjects.DetectedObject>items()) {
            BoundingBox bbox = obj.getBoundingBox();

            Rectangle rectangle = bbox.getBounds();
            String showText = String.format("%s: %.2f", obj.getClassName(), obj.getProbability());
            rect.x = (int)(rectangle.getX() * width);
            rect.y = (int)(rectangle.getY() * height);
            rect.width = (int)(rectangle.getWidth() * width);
            rect.height = (int)(rectangle.getHeight() * height);
            // add rectangle box
            Imgproc.rectangle(image, rect, boxcolor, 1);
            // put object label
            addTextToBox(image, showText, txtcolor, Imgproc.FONT_HERSHEY_COMPLEX, boxcolor,
                    new Point(rect.x, rect.y - 1), 0.4, 1);
        }
        // show predicted results
        HighGui.imshow(backbone, image);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();

        System.exit(0);
    }

    public static DetectedObjects predict() throws IOException, ModelException, TranslateException {
        Path imageFile = Paths.get("data/images/dog.jpg");
        image = imread(imageFile.toString());
        //Image img = ImageFactory.getInstance().fromFile(imageFile);
        Image img = mat2Image(image);

        if ("TensorFlow".equals(Engine.getInstance().getEngineName())) {
            backbone = "mobilenet_v2";
        } else {
            backbone = "resnet50";
        }

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", backbone)
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                if( img != null )
                    saveBoundingBoxImage(img, detection);
                return detection;
            }
        }
    }
    private static void saveBoundingBoxImage(Image img, DetectedObjects detection)
            throws IOException {

        Path outputDir = Paths.get("output");
        Files.createDirectories(outputDir);

        // Make image copy with alpha channel because original image was jpg
        Image newImage = img.duplicate(); //Image.Type.TYPE_INT_ARGB);
        newImage.drawBoundingBoxes(detection);

        Path imagePath = outputDir.resolve("detected-dog_bike_car.png");
        // OpenJDK can't save jpg with alpha channel
        newImage.save(Files.newOutputStream(imagePath), "png");
        System.out.println("Detected objects image has been saved in: " + imagePath);

    }
}

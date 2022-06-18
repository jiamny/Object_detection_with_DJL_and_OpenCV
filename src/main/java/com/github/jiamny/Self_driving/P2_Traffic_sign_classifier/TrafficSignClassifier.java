package com.github.jiamny.Self_driving.P2_Traffic_sign_classifier;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SequentialBlock;
import ai.djl.translate.Translator;
import com.github.jiamny.Utils.ImageHelper;
import com.github.jiamny.Utils.ImageViewer;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import static com.github.jiamny.Self_driving.P2_Traffic_sign_classifier.TrafficSignLeNet.createLeNet;
import static com.github.jiamny.Self_driving.P2_Traffic_sign_classifier.TrafficSignLeNet.getClassMap;

public class TrafficSignClassifier {

    public static void main(String[] args) {

        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");

        try {
            Engine.getInstance().setRandomSeed(1111);

            HashMap<Integer, String> imgclasses = getClassMap();

            NDManager manager = NDManager.newBaseManager();
            SequentialBlock block = new SequentialBlock();

            block = createLeNet(block, manager);

            Model model = Model.newInstance("cnn");
            model.setBlock(block);

            // Assume you have run TrafficSignLeNet.java example, and saved model in ./data/self_driving/model folder.
            Path modelDir = Paths.get("./data/self_driving/model");
            model.load(modelDir);

            List<String> classes =  imgclasses.keySet().stream().mapToInt(i -> i).mapToObj(String::valueOf).collect(Collectors.toList());

            Translator<Image, Classifications> translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new Resize(32, 32))
                            .addTransform(new ToTensor())
                            .optSynset(classes)
                            .build();

            Predictor<Image, Classifications> predictor = model.newPredictor(translator);

            String dirPath = "./data/self_driving/images/new_images_test";

            if( Files.exists(Paths.get(dirPath)) ) {
                File f = new File(dirPath);
                String[] pathnames = f.list();

                for (String pathname : pathnames) {
                    System.out.println(pathname);
                    Path imageFile = Paths.get( dirPath + "/" + pathname);
                    Image img = ImageFactory.getInstance().fromFile(imageFile);

                    Classifications prd = predictor.predict(img);
                    String cls = prd.best().getClassName();
                    System.out.println(imgclasses.get(Integer.valueOf(cls)));

                    Mat iim = ImageHelper.ndarrayToMat(img.toNDArray(manager));

                    // ImageFactory.getInstance().fromFile() read image in BGR format
                    Imgproc.cvtColor(iim, iim, Imgproc.COLOR_BGR2RGB);
                    //HighGui.imshow(imgclasses.get(Integer.valueOf(cls)), iim);
                    //HighGui.waitKey(3000);

                    Mat dst = new Mat();
                    Imgproc.resize(iim, dst, new Size(300, 200));
                    ImageViewer.show(dst, imgclasses.get(Integer.valueOf(cls)));
                    Thread.sleep(3000);
                }
                //HighGui.destroyAllWindows();
            }
        } catch(Exception e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}
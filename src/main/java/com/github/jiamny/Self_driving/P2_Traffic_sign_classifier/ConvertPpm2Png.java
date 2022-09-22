package com.github.jiamny.Self_driving.P2_Traffic_sign_classifier;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Blocks;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.loss.Loss;
import com.github.jiamny.Utils.ImageHelper;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class ConvertPpm2Png {

    public static void main(String[] args) {
        System.load("/usr/local/share/java/opencv4/libopencv_java460.so");

        Repository repository = Repository.newInstance("train", "./data/GTSRB/Final_Training/Images");
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        try(Model model = Model.newInstance("model")) {
            model.setBlock(Blocks.identityBlock());

            ImageFolder dataset =
                    ImageFolder.builder()
                            .setRepository(repository)
                            .addTransform(new Resize(100, 100))
                            .addTransform(new ToTensor())
                            .setSampling(1, false)
                            .build();

            List<String> synsets = Arrays.asList(
                    "00000", "00001", "00002", "00003", "00004", "00005", "00006", "00007", "00008", "00009",
                    "00010", "00011", "00012", "00013", "00014", "00015", "00016", "00017", "00018", "00019",
                    "00020", "00021", "00022", "00023", "00024", "00025", "00026", "00027", "00028", "00029",
                    "00030", "00031", "00032", "00033", "00034", "00035", "00036", "00037", "00038", "00039",
                    "00040", "00041", "00042");

            Trainer trainer = model.newTrainer(config);
            NDManager manager = trainer.getManager();

            // convert ppm to png
            for(String cls : synsets) {
                String dirPath = "./data/GTSRB/Final_Training/Images/" + cls;
                System.out.println(dirPath);
                String outDir = "./data/GTSRB/train/" + cls;

                if( ! Files.exists(Paths.get(outDir)) )
                    Files.createDirectory(Paths.get(outDir));

                if( Files.exists(Paths.get(dirPath)) ) {
                    File f = new File(dirPath);
                    String [] pathnames = f.list();

                    // For each pathname in the pathnames array
                    for (String pathname : pathnames) {
                        // Print the names of files and directories
                        if( pathname.contains(".ppm") ) {
                            System.out.println(pathname);
                            String fpath = dirPath + "/" + pathname;
                            String opath = outDir + "/" + pathname.replace(".ppm", ".png");
                            /*
                            NDArray img = ImageFactory.getInstance()
                                            .fromFile(Paths.get(fpath))
                                            .toNDArray(manager);

                            Mat iim = ImageHelper.ndarrayToMat(img);
                             */
                            System.out.println(opath);
                            // -------------------------------------------
                            // ImageIO.read() image in RGB format!!!
                            // -------------------------------------------
                            BufferedImage img2 = ImageIO.read(new File(fpath));
                            Mat iim = ImageHelper.bufferedImage2Mat(img2);
                            //Imgproc.cvtColor(iim, iim, Imgproc.COLOR_BGR2RGB);
                            Imgcodecs.imwrite(opath, iim);
                        }
                    }
                }
            }
            String dirPath = "./data/GTSRB/Final_Test/Images";
            String outDir = "./data/GTSRB/test";

            if( ! Files.exists(Paths.get(outDir)) )
                Files.createDirectory(Paths.get(outDir));

            if( Files.exists(Paths.get(dirPath)) ) {
                File f = new File(dirPath);
                String[] pathnames = f.list();

                for (String pathname : pathnames) {
                    // Print the names of files and directories
                    if( pathname.contains(".ppm") ) {
                        System.out.println(pathname);
                        String fpath = dirPath + "/" + pathname;
                        String opath = outDir + "/" + pathname.replace(".ppm", ".png");
                        /*
                        NDArray img = ImageFactory.getInstance()
                                            .fromFile(Paths.get(fpath))
                                            .toNDArray(manager);

                        Mat iim = ImageHelper.ndarrayToMat(img);
                        */
                        System.out.println(opath);
                        // -------------------------------------------
                        // ImageIO.read() image in RGB format!!!
                        // -------------------------------------------
                        BufferedImage img2 = ImageIO.read(new File(fpath));
                        Mat iim = ImageHelper.bufferedImage2Mat(img2);
                        //Imgproc.cvtColor(iim, iim, Imgproc.COLOR_BGR2RGB);
                        Imgcodecs.imwrite(opath, iim);
                    }
                }
            }

        } catch(Exception e) {
            e.printStackTrace();
        }
    }
}
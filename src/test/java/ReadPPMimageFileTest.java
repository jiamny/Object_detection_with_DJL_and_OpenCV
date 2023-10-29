import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageFolder;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDResource;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;

import com.github.jiamny.Utils.ImageHelper;
import com.github.jiamny.Utils.ImageViewer;
//import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

//import static org.junit.jupiter.api.Assertions.assertEquals;

import org.testng.Assert;
import org.testng.annotations.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;


import static com.github.jiamny.Utils.ImageHelper.bufferedImage2Mat;
import static org.testng.Assert.assertEquals;

public class ReadPPMimageFileTest {
    @Test
    public void testReadPpmFile() {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");

        try {
            System.out.println(Arrays.toString(ImageIO.getReaderFormatNames()));

            BufferedImage in_image = ImageIO.read(new File("./data/IMG/Test/00000.ppm"));

            ImageViewer.show(bufferedImage2Mat(in_image));
            Thread.sleep(1500);
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testReadPpmFileFolder() {
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");

        Repository repository = Repository.newInstance("Train", "./data/IMG");
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
            //assertEquals(synsets, dataset.getSynset());

            Trainer trainer = model.newTrainer(config);
            NDManager manager = trainer.getManager();

            BufferedImage img2 = ImageIO.read(new File("./data/IMG/Train/00000/00000_00001.ppm"));

            NDArray img =
                    ImageFactory.getInstance()
                            .fromFile(Paths.get("./data/IMG/Train/00000/00000_00001.ppm"))
                            .toNDArray(manager);

            Mat iim = ImageHelper.bufferedImage2Mat(img2);
            ImageViewer.show(iim);
            //Imgproc.cvtColor(iim, iim, Imgproc.COLOR_RGB2BGR);

            ImageViewer.show(ImageHelper.bufferedImage2Mat(img2), "img2");
            Thread.sleep(3500);
            Image jimg = ImageFactory.getInstance().fromImage(img2);

            //Mat iim = ImageHelper.ndarrayToMat(img);
            //Imgproc.cvtColor(iim, iim, Imgproc.COLOR_BGR2RGB);
            //ImageViewer.show(iim, "iim");
            //Thread.sleep(3500);
            //Imgcodecs.imwrite("./data/PPM/00000_00001.png", iim);
/*
            int cnt = 0;
            // convert ppm to png
            for(String cls : synsets) {
                String dirPath = "./data/IMG/Train/" + cls;
                //String dirPath = "./data/GTSRB/Final_Test/Images";
                System.out.println(dirPath);
                String outDir = "./data/GTSRB/train/" + cls;
                //String outDir = "./data/GTSRB/test";
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

                            //NDArray img = ImageFactory.getInstance()
                            //                .fromFile(Paths.get(fpath))
                            //                .toNDArray(manager);

                            //Mat iim = ImageHelper.ndarrayToMat(img);

                            System.out.println(opath);
                            // -------------------------------------------
                            // ImageIO.read() image in RGB format!!!
                            // -------------------------------------------
                            BufferedImage img2 = ImageIO.read(new File(fpath));
                            Mat iim = ImageHelper.bufferedImage2Mat(img2);
                            //Imgproc.cvtColor(iim, iim, Imgproc.COLOR_BGR2RGB);
                            if( Math.random() < 0.5 )
                                Imgcodecs.imwrite(opath, iim);
                            cnt++;
                        }
                    }
                }
            }
*/
        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testRandomSplit() throws IOException, TranslateException {
        //System.load("C:\\Program Files\\Opencv4\\java\\x64\\opencv_java454.dll");
        System.load("/usr/local/share/java/opencv4/libopencv_java470.so");

        try {
            NDManager manager = NDManager.newBaseManager();
            NDArray img =
                    ImageFactory.getInstance()
                            .fromFile(Paths.get("./data/IMG/Train/00000/00003_00028.ppm"))
                            .toNDArray(manager);

            Mat iim = ImageHelper.ndarrayToMat(img);
            Imgproc.cvtColor(iim, iim, Imgproc.COLOR_BGR2RGB);
            ImageViewer.show(iim);
            Thread.sleep(3500);
            Imgcodecs.imwrite("./data/IMG/00003_00028.png", iim);

            NDArray dog =
                    ImageFactory.getInstance()
                            .fromFile(
                                    Paths.get(
                                            "./data/images/dog.jpg"))
                            .toNDArray(manager);

            iim = ImageHelper.ndarrayToMat(dog);
            ImageViewer.show(iim);
            Thread.sleep(3500);
            Imgproc.cvtColor(iim, iim, Imgproc.COLOR_BGR2RGB);
            ImageViewer.show(iim);
            Thread.sleep(3500);
        }catch(Exception e) {
            e.printStackTrace();
        }
    }
}

package com.github.jiamny.Semantic_Segmentation;
import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslatorFactory;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Color;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static com.github.jiamny.Utils.ImageHelper.ndarrayToMat;

/**
 * An example of inference using a semantic segmentation model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/semantic_segmentation.md">doc</a>
 * for information about this example.
 */
public final class SemanticSegmentation {
  static {
    //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
  }

  private static final Logger logger = LoggerFactory.getLogger(SemanticSegmentation.class);

  private SemanticSegmentation() {}

  public static void main(String[] args) throws IOException, ModelException, TranslateException {
    // ----------------------------------------------------------------------
    // set specific version of torch & CUDA
    // ----------------------------------------------------------------------
    System.setProperty("PYTORCH_VERSION", "1.13.1");
    System.setProperty("PYTORCH_FLAVOR", "cu117");
    System.out.println(Engine.getDefaultEngineName());
    System.out.println(Engine.getInstance().defaultDevice());

    java.util.Set<java.lang.String> egs = Engine.getAllEngines();
    for( String s : egs )
      System.out.println(s);

    Engine eg = Engine.getEngine("PyTorch");
    System.out.println("Engine: " + eg.getEngineName());

    ai.djl.Device [] ds = eg.getDevices();
    for(int i = 0; i < ds.length; i++)
      System.out.println(ds[i].toString());

    System.out.println("Default device: " + eg.defaultDevice().toString());

    SemanticSegmentation.predict();
  }

  public static void predict() throws IOException, ModelException, TranslateException {
    Path imageFile = Paths.get("./data/images/person.jpg");
    ImageFactory factory = ImageFactory.getInstance();
    Image img = factory.fromFile(imageFile);

    String url =
            "https://mlrepo.djl.ai/model/cv/semantic_segmentation/ai/djl/pytorch/deeplabv3/0.0.1/deeplabv3.zip";

    Criteria<Image, CategoryMask> criteria =
            Criteria.builder()
                    .setTypes(Image.class, CategoryMask.class)
                    .optModelUrls(url)
                    .optTranslatorFactory(new SemanticSegmentationTranslatorFactory())
                    .optEngine("PyTorch")
                    .optDevice(Device.cpu()) //use CPU !!! 'prepacked::conv2d_clamp_run' only available for CPU
                    .optProgress(new ProgressBar())
                    .build();
    Image bg = factory.fromFile(Paths.get("data/images/stars-in-the-night-sky.jpg"));
    try (ZooModel<Image, CategoryMask> model = criteria.loadModel();
         Predictor<Image, CategoryMask> predictor = model.newPredictor()) {
      CategoryMask mask = predictor.predict(img);

      // Highlights the detected object on the image with random opaque colors.
      Image img1 = img.duplicate();
      mask.drawMask(img1, 255);
      saveSemanticImage(img1, "semantic_instances1.png");

      NDManager manager = NDManager.newBaseManager();
      Mat im = ndarrayToMat(img1.toNDArray(manager));
      HighGui.imshow("semantic_instances1", im);

      if( HighGui.waitKey(0) == 27) {
        System.out.println("UU");
        HighGui.destroyAllWindows();
      }

      // Highlights the detected object on the image with random colors.
      Image img2 = img.duplicate();
      mask.drawMask(img2, 180, 0);
      saveSemanticImage(img2, "semantic_instances2.png");
      im = ndarrayToMat(img2.toNDArray(manager));
      HighGui.imshow("semantic_instances2", im);
      int k = HighGui.waitKey(0);
      if(k == 27)
        HighGui.destroyAllWindows();

      // Highlights only the dog with blue color.
      Image img3 = img.duplicate();
      mask.drawMask(img3, 12, Color.BLUE.brighter().getRGB(), 180);
      saveSemanticImage(img3, "semantic_instances3.png");
      im = ndarrayToMat(img3.toNDArray(manager));
      HighGui.imshow("semantic_instances3", im);
      k = HighGui.waitKey(0);
      if(k == 27)
        HighGui.destroyAllWindows();

      // Extract dog from the image
      Image dog = mask.getMaskImage(img, 12);
      dog = dog.resize(img.getWidth(), img.getHeight(), true);
      saveSemanticImage(dog, "semantic_instances4.png");
      im = ndarrayToMat(dog.toNDArray(manager));
      HighGui.imshow("semantic_instances4", im);
      k = HighGui.waitKey(0);
      if(k == 27)
        HighGui.destroyAllWindows();

      // Replace background with an image
      bg = bg.resize(img.getWidth(), img.getHeight(), true);
      bg.drawImage(dog, true);
      saveSemanticImage(bg, "semantic_instances5.png");
      im = ndarrayToMat(bg.toNDArray(manager));
      HighGui.imshow("semantic_instances5", im);
      k = HighGui.waitKey(0);
      if(k == 27)
        HighGui.destroyAllWindows();

      System.exit(0);
    }
  }

  private static void saveSemanticImage(Image img, String fileName) throws IOException {
    Path outputDir = Paths.get("output");
    Files.createDirectories(outputDir);

    Path imagePath = outputDir.resolve(fileName);
    img.save(Files.newOutputStream(imagePath), "png");
    logger.info("Segmentation result image has been saved in: {}", imagePath.toAbsolutePath());
  }
}

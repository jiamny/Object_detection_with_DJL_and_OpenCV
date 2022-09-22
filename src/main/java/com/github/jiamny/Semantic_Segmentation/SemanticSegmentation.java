package com.github.jiamny.Semantic_Segmentation;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import org.opencv.core.Mat;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static com.github.jiamny.Utils.ImageHelper.bufferedImage2Mat;
import static com.github.jiamny.Utils.ImageHelper.mat2DjlImage;
import static com.github.jiamny.Utils.ImageViewer.show;
import static org.opencv.imgcodecs.Imgcodecs.imread;

public final class SemanticSegmentation {

  static {
    //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    // Make sure that you loaded your corresponding opencv java .so file.
    System.load("/usr/local/share/java/opencv4/libopencv_java460.so");
  }

  private SemanticSegmentation() {}

  public static void main(String[] args) throws IOException, ModelException, TranslateException {
    Path imageFile = Paths.get("./data/images/person.jpg");
    //Image img = ImageFactory.getInstance().fromFile(imageFile);
    Mat image = imread(imageFile.toString());
    //Image img = ImageFactory.getInstance().fromFile(imageFile);
    Image img = mat2DjlImage(image);

    BufferedImage detection = SemanticSegmentation.predict(img);
    Mat segimg = bufferedImage2Mat(detection);
    show(segimg, "Segmentation image");

    Path outputDir = Paths.get("./output/images");
    Files.createDirectories(outputDir);
    Path imagePath = outputDir.resolve("Segmentation_image.png");
    mat2DjlImage(segimg).save(Files.newOutputStream(imagePath), "png");
    System.out.println("Segmentation image has been saved");
  }

  public static BufferedImage predict(Image img) throws IOException, ModelException, TranslateException {

    String url =
            "https://mlrepo.djl.ai/model/cv/semantic_segmentation/ai/djl/pytorch/deeplabv3/0.0.1/deeplabv3.zip";

    String modelPath = "./models/deeplabv3.zip";

    Map<String, String> arguments = new ConcurrentHashMap<>();
    arguments.put("toTensor", "true");
    arguments.put("normalize", "true");
    SemanticSegmentationTranslator translator =
            SemanticSegmentationTranslator.builder(arguments).build();

    Criteria<Image, Image> criteria =
            Criteria.builder()
                    .setTypes(Image.class, Image.class)
                    .optModelUrls(url)
                    //.optModelPath(Paths.get(modelPath)) // search models in specified path
                    .optTranslator(translator)
                    .optEngine("PyTorch")
                    .build();

    try (ZooModel<Image, Image>  model = criteria.loadModel()) {
      try ( Predictor<Image, Image> predictor = model.newPredictor()) {
        BufferedImage img2 = (BufferedImage) predictor.predict(img).getWrappedImage();
        return img2;
      }
    }
  }
}

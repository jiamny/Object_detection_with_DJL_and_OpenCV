package com.github.jiamny.FeatureExtraction;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * 图片特征提取 （512维特征）
 * Image feature extraction (512-dimensional features)
 * @author Calvin
 * @date 2021-07-10
 * @email 179209347@qq.com
 */
public final class FeatureExtractionExample {

    private static final Logger logger = LoggerFactory.getLogger(FeatureExtractionExample.class);

    private FeatureExtractionExample() {
    }

    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        // ---------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        Path imageFile = Paths.get("data/images/car1.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);
        Criteria<Image, float[]> criteria = new ImageEncoderModel().criteria();
        try (ZooModel model = ModelZoo.loadModel(criteria);
             Predictor<Image, float[]> predictor = model.newPredictor()) {
            float[] feature = predictor.predict(img);
            logger.info(Arrays.toString(feature));
            System.out.println("feature: " + Arrays.toString(feature));
        }
    }
}

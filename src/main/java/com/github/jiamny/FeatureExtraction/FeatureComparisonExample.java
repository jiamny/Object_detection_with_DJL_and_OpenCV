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

/**
 * Feature Comparison - 1:1.
 * 特征比对 - 1:1.
 * @author Calvin
 * @date 2021-07-10
 * @email 179209347@qq.com
 **/
public final class FeatureComparisonExample {

    private static final Logger logger = LoggerFactory.getLogger(FeatureComparisonExample.class);
    private FeatureComparisonExample() {
    }

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        Path imageFile1 = Paths.get("data/images/car1.png");
        Image img1 = ImageFactory.getInstance().fromFile(imageFile1);
        Path imageFile2 = Paths.get("data/images/car2.png");
        Image img2 = ImageFactory.getInstance().fromFile(imageFile2);
        Criteria<Image, float[]> criteria = new ImageEncoderModel().criteria();
        try (ZooModel model = ModelZoo.loadModel(criteria);
             Predictor<Image, float[]> predictor = model.newPredictor()) {
            float[] feature1 = predictor.predict(img1);
            float[] feature2 = predictor.predict(img2);

            // 欧式距离
            // Euclidean distance
            float dis = FeatureComparison.dis(feature1, feature2);
            logger.info(Float.toString(dis));
            System.out.println("Euclidean distance: " + dis);

            // 余弦相似度
            // Cosine similarity
            float cos = FeatureComparison.cosineSim(feature1, feature2);
            logger.info(Float.toString(cos));
            System.out.println("余弦相似度: " + cos);
        }
        System.exit(0);
    }

}

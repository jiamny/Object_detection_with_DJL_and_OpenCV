package com.github.jiamny.Self_driving.P2_Traffic_sign_classifier;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.*;
import ai.djl.engine.Engine;
import ai.djl.metric.*;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.NumericColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;

import java.nio.file.Paths;
import java.util.*;

import static ai.djl.training.EasyTrain.evaluateDataset;
import static ai.djl.training.EasyTrain.trainBatch;
import static com.github.jiamny.Utils.Utils.toDoubleArray;

import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

public class TrafficSignLeNet {

    public static SequentialBlock createLeNet(SequentialBlock block, NDManager manager) {
        block.add(Conv2d.builder()
                    .setKernelShape(new Shape(5, 5))
                    .optPadding(new Shape(2, 2))
                    .optBias(false)
                    .setFilters(6)
                    .build())
             .add(Activation::sigmoid)
             .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
             .add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .setFilters(16).build())
             .add(Activation::sigmoid)
             .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                // Blocks.batchFlattenBlock() will transform the input of the shape (batch size, channel,
                // height, width) into the input of the shape (batch size,
                // channel * height * width)
             .add(Blocks.batchFlattenBlock())
             .add(Linear
                        .builder()
                        .setUnits(120)
                        .build())
             .add(Activation::sigmoid)
             .add(Linear
                        .builder()
                        .setUnits(84)
                        .build())
             .add(Activation::sigmoid)
             .add(Linear
                        .builder()
                        .setUnits(43)
                        .build());
        return block;
    }

    public static HashMap<Integer, String> getClassMap() {
        HashMap<Integer, String> imgclasses = new HashMap<>();
        imgclasses.put(0,"catBatch.getSize()");
        imgclasses.put(1,"Speed limit (30km/h)");
        imgclasses.put(2,"Speed limit (50km/h)");
        imgclasses.put(3,"Speed limit (60km/h)");
        imgclasses.put(4,"Speed limit (70km/h)");
        imgclasses.put(5,"Speed limit (80km/h)");
        imgclasses.put(6,"End of speed limit (80km/h)");
        imgclasses.put(7,"Speed limit (100km/h)");
        imgclasses.put(8,"Speed limit (120km/h)");
        imgclasses.put(9,"No passing");
        imgclasses.put(10,"No passing for vehicles over 3.5 metric tons");
        imgclasses.put(11,"Right-of-way at the next intersection");
        imgclasses.put(12,"Priority road");
        imgclasses.put(13,"Yield");
        imgclasses.put(14,"Stop");
        imgclasses.put(15,"No vehicles");
        imgclasses.put(16,"Vehicles over 3.5 metric tons prohibited");
        imgclasses.put(17,"No entry");
        imgclasses.put(18,"General caution");
        imgclasses.put(19,"Dangerous curve to the left");
        imgclasses.put(20,"Dangerous curve to the right");
        imgclasses.put(21,"Double curve");
        imgclasses.put(22,"Bumpy road");
        imgclasses.put(23,"Slippery road");
        imgclasses.put(24,"Road narrows on the right");
        imgclasses.put(25,"Road work");
        imgclasses.put(26,"Traffic signals");
        imgclasses.put(27,"Pedestrians");
        imgclasses.put(28,"Children crossing");
        imgclasses.put(29,"Bicycles crossing");
        imgclasses.put(30,"Beware of ice/snow");
        imgclasses.put(31,"Wild animals crossing");
        imgclasses.put(32,"End of all speed and passing limits");
        imgclasses.put(33,"Turn right ahead");
        imgclasses.put(34,"Turn left ahead");
        imgclasses.put(35,"Ahead only");
        imgclasses.put(36,"Go straight or right");
        imgclasses.put(37,"Go straight or left");
        imgclasses.put(38,"Keep right");
        imgclasses.put(39,"Keep left");
        imgclasses.put(40,"Roundabout mandatory");
        imgclasses.put(41,"End of no passing");
        imgclasses.put(42,"End of no passing by vehicles over 3.5 metric tons");
        return imgclasses;
    }

    public static void main(String [] args) {

        System.load("/usr/local/share/java/opencv4/libopencv_java455.so");

        try {
            Repository repository = Repository.newInstance("train", "./data/GTSRB/train");
            Engine.getInstance().setRandomSeed(1111);

            HashMap<Integer, String> imgclasses = getClassMap();

            NDManager manager = NDManager.newBaseManager();
            SequentialBlock block = new SequentialBlock();

            block = createLeNet(block, manager);
            float lr = 0.9f;
            Model model = Model.newInstance("cnn");
            model.setBlock(block);

            Loss loss = Loss.softmaxCrossEntropyLoss();

            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
                    //.optDevices(Engine.getInstance().getDevices(1)) // Single GPU
                    .addEvaluator(new Accuracy()) // Model Accuracy
                    .addTrainingListeners(TrainingListener.Defaults.basic());

            Trainer trainer = model.newTrainer(config);

            NDArray X = manager.randomUniform(0f, 1.0f, new Shape(1, 3, 32, 32));
            trainer.initialize(X.getShape());

            Shape currentShape = X.getShape();

            for (int i = 0; i < block.getChildren().size(); i++) {
                Shape[] newShape = block.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
                currentShape = newShape[0];
                System.out.println(block.getChildren().get(i).getKey() + " layer output : " + currentShape);
            }

            int batchSize = 32;
            int numEpochs = Integer.getInteger("MAX_EPOCH", 100);
            ArrayList<Double> trainLoss = new ArrayList<>();
            ArrayList<Double> testAccuracy = new ArrayList<>();
            ArrayList<Double> epochCount = new ArrayList<>();
            ArrayList<Double> trainAccuracy = new ArrayList<>();
            double valAccuracy = 0.96;

            // GTSRB data
            ImageFolder dataset =
                    ImageFolder.builder()
                            .setRepository(repository)
                            .addTransform(new Resize(32, 32))
                            .addTransform(new ToTensor())
                            .setSampling(batchSize, true)
                            .build();

            System.out.println(dataset.getSynset());

            RandomAccessDataset[] sets = dataset.randomSplit(80, 20);
            System.out.println(sets[0].size());
            System.out.println(sets[1].size());

            //Iterator<Batch> trainIter = trainer.iterateDataset(sets[0]).iterator();
            //Iterator<Batch> testIter = trainer.iterateDataset(sets[1]).iterator();
            //Batch batch = ds.next();
            //System.out.println(batch.getSize());

            //double avgTrainTimePerEpoch = 0;
            //Map<String, double[]> evaluatorMetrics = new HashMap<>();
            trainer.setMetrics(new Metrics());

            System.out.println("numEpochs: " + numEpochs);

            for (int epoch = 0; epoch < numEpochs; epoch++) {

                // We iterate through the dataset once during each epoch
                for (Batch batch : trainer.iterateDataset(sets[0])) {

                    // During trainBatch, we update the loss and evaluators with the results for the
                    // training batch
                    trainBatch(trainer, batch);

                    // Now, we update the model parameters based on the results of the latest trainBatch
                    trainer.step();

                    // We must make sure to close the batch to ensure all the memory associated with the
                    // batch is cleared.
                    // If the memory isn't closed after each batch, you will very quickly run out of
                    // memory on your GPU
                    batch.close();
                }

                // After each epoch, test against the validation dataset if we have one
                evaluateDataset(trainer, sets[1]);

                // reset training and validation evaluators at end of epoch
                trainer.notifyListeners(listener -> listener.onEpoch(trainer));

                //Metrics metrics = trainer.getMetrics();
                Map<String, Float> trlt = trainer.getTrainingResult().getEvaluations();

                //System.out.println(trlt.keySet());
                System.out.printf( "epoch: %d; train_loss: %.4f; train_Accuracy %.4f; validate_loss: %.4f; validate_Accuracy: %.4f\n",
                        (epoch+1), trlt.get("train_loss").floatValue(), trlt.get("train_Accuracy").floatValue(),
                        trlt.get("validate_loss").floatValue(), trlt.get("validate_Accuracy").floatValue());

                //System.out.println(trainer.getTrainingResult().getValidateEvaluation("Accuracy"));
                trainLoss.add(trlt.get("train_loss").floatValue()*1.0);
                testAccuracy.add(trlt.get("validate_Accuracy").floatValue()*1.0);
                epochCount.add((epoch+1)*1.0);
                trainAccuracy.add(trlt.get("train_Accuracy").floatValue()*1.0);

                if( trainer.getTrainingResult().getValidateEvaluation("Accuracy")*1.0 > valAccuracy ){
                    model.setProperty("Epoch", String.valueOf((epoch + 1)));
                    model.setProperty(
                            "Accuracy", String.format("%.5f", trainer.getTrainingResult().getValidateEvaluation("Accuracy")));
                    model.setProperty("Loss", String.format("%.5f", trainer.getTrainingResult().getValidateLoss()));
                    model.save(Paths.get("./data/self_driving/model"), "cnn");
                    valAccuracy = trainer.getTrainingResult().getValidateEvaluation("Accuracy");
                }
            }

            String[] lossLabel = new String[trainLoss.size() + testAccuracy.size() + trainAccuracy.size()];

            Arrays.fill(lossLabel, 0, trainLoss.size(), "train loss");
            Arrays.fill(lossLabel, trainAccuracy.size(), trainLoss.size() + trainAccuracy.size(), "train acc");
            Arrays.fill(lossLabel, trainLoss.size() + trainAccuracy.size(),
                    trainLoss.size() + testAccuracy.size() + trainAccuracy.size(), "test acc");

            Table data = Table.create("Data").addColumns(
                    DoubleColumn.create("epoch", ArrayUtils.addAll(toDoubleArray(epochCount),
                            ArrayUtils.addAll(toDoubleArray(epochCount), toDoubleArray(epochCount)))),
                    DoubleColumn.create("metrics", ArrayUtils.addAll(toDoubleArray(trainLoss),
                            ArrayUtils.addAll(toDoubleArray(trainAccuracy), toDoubleArray(testAccuracy)))),
                    StringColumn.create("lossLabel", lossLabel)
            );
            //LinePlot.create("", data, "epoch", "metrics", "lossLabel");

            Table trLoss = data.where(data.stringColumn("lossLabel").isEqualTo("train loss"));
            Table trAcc = data.where(data.stringColumn("lossLabel").isEqualTo("train acc"));
            Table tsAcc = data.where(data.stringColumn("lossLabel").isEqualTo("test acc"));

            Layout layout =
                    Layout.builder().title("Train GTSRB data").build();

            NumericColumn<?> x1 = trLoss.nCol("epoch");
            NumericColumn<?> y1 = trLoss.nCol("metrics");
            ScatterTrace trace1 = ScatterTrace.builder(x1, y1)
                    .name("train loss")
                    .mode(ScatterTrace.Mode.LINE).build();

            NumericColumn<?> x2 = trAcc.nCol("epoch");
            NumericColumn<?> y2 = trAcc.nCol("metrics");
            ScatterTrace trace2 = ScatterTrace.builder(x2, y2)
                    .name("train acc")
                    .mode(ScatterTrace.Mode.LINE).build();

            NumericColumn<?> x3 = tsAcc.nCol("epoch");
            NumericColumn<?> y3 = tsAcc.nCol("metrics");
            ScatterTrace trace3 = ScatterTrace.builder(x3, y3)
                    .name("test acc")
                    .mode(ScatterTrace.Mode.LINE).build();

            Plot.show(new Figure(layout, trace1, trace2, trace3));
        }catch(Exception e) {
            e.printStackTrace();
        }
    }
}

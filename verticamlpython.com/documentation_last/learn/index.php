<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <div><?php include('../../include/logo.php'); ?></div>
        <h1>Machine Learning</h1>
        <center><ul id="advantages">
          <a href="classification.php"><li>
            <img src="../../img/icons/Business Hierarcy.png">
            <h3>Classification</h3>
            <div>Predict a Categorical Response</div>
          </li></a>
           <a href="regression.php"><li>
            <img src="../../img/icons/Analysis.png">
            <h3>Regression</h3>
            <div>Predict a Numerical Response</div>
          </li></a>
          <a href="unsupervised.php"><li>
            <img src="../../img/icons/Team Leader Search.png">
            <h3>Clustering & Anomaly Detection</h3>
            <div>Find Clusters and Anomalies</div>
          </li></a>
          <a href="decomposition-preprocessing.php"><li>
            <img src="../../img/icons/Management.png">
            <h3>Decomposition & Preprocessing</h3>
            <div>Decompose and Preprocess</div>
          </li></a>
          <a href="examples/index.php"><li>
            <img src="../../img/icons/Manual.png" height="200px">
            <h3>Examples</h3>
            <div>Models Usage Examples</div>
          </li></a>
        </ul></center>
        <h1>Model Evaluation & Selection</h1>
        <center><ul id="advantages">
          <a href="model-selection.php"><li>
            <img src="../../img/icons/Campaign Tweaking.png">
            <h3>Model Selection</h3>
            <div>Cross Validate, Train/Test Split...</div>
          </li></a>
          <a href="metrics/index.php"><li>
            <img src="../../img/icons/Target.png">
            <h3>Metrics</h3>
            <div>F1-Score, AUC, MSE...</div>
          </li></a>
          <a href="plot.php"><li>
            <img src="../../img/icons/Graphics.png">
            <h3>Graphics</h3>
            <div>ROC/PRC Curve, Elbow...</div>
          </li></a>
        </ul></center>
        <h1>Datasets</h1>
        <center><ul id="advantages">
          <a href="datasets/index.php"><li>
            <img src="../../img/icons/Game Develop.png">
            <h3>Datasets</h3>
            <div>Titanic, Iris, Amazon...</div>
          </li></a>
        </ul></center>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
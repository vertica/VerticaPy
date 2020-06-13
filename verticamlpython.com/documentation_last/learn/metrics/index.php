<!DOCTYPE html>
<html>
    <?php include('../../../include/head.php'); ?>
    <body>
      <div><?php include('../../../include/header.php'); ?></div>
      <div id="content">
        <h2>Metrics for Regression</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="explained_variance/index.php">explained_variance</td> <td>Computes the Explained Variance.</td> </tr>
          <tr> <td><a href="max_error/index.php">max_error</td> <td>Computes the Max Error.</td> </tr>
          <tr> <td><a href="mean_absolute_error/index.php">mean_absolute_error</td> <td>Computes the Mean Absolute Error.</td> </tr>
          <tr> <td><a href="mean_squared_error/index.php">mean_squared_error</td> <td>Computes the Mean Squared Error.</td> </tr>
          <tr> <td><a href="mean_squared_log_error/index.php">mean_squared_log_error</td> <td>Computes the Mean Squared Log Error.</td> </tr>
          <tr> <td><a href="median_absolute_error/index.php">median_absolute_error</td> <td>Computes the Median Absolute Error.</td> </tr>
          <tr> <td><a href="r2_score/index.php">r2_score</td> <td>Computes the R2 Score.</td> </tr>
          <tr> <td><a href="regression_report/index.php">regression_report</td> <td>Computes a regression report using multiple metrics (r2, mse, max error...).</td> </tr>
        </table>
        <h2>Metrics for Classification</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="accuracy_score/index.php">accuracy_score</td> <td>Computes the Accuracy Score.</td> </tr>
          <tr> <td><a href="auc">auc</td> <td>Computes the ROC AUC (Area Under Curve).</td> </tr>
          <tr> <td><a href="classification_report/index.php">classification_report</td> <td>Computes a classification report using multiple metrics (AUC, accuracy, PRC AUC, F1...).</td> </tr>
          <tr> <td><a href="confusion_matrix/index.php">confusion_matrix</td> <td>Computes the Confusion Matrix.</td> </tr>
          <tr> <td><a href="critical_success_index/index.php">critical_success_index</td> <td>Computes the Critical Success Index.</td> </tr>
          <tr> <td><a href="f1_score/index.php">f1_score</td> <td>Computes the F1 Score.</td> </tr>
          <tr> <td><a href="informedness/index.php">informedness</td> <td>Computes the Informedness.</td> </tr>
          <tr> <td><a href="log_loss/index.php">log_loss</td> <td>Computes the Log Loss.</td> </tr>
          <tr> <td><a href="markedness/index.php">markedness</td> <td>Computes the Markedness.</td> </tr>
          <tr> <td><a href="matthews_corrcoef/index.php">matthews_corrcoef</td> <td>Computes the Matthews Correlation Coefficient.</td> </tr>
          <tr> <td><a href="multilabel_confusion_matrix/index.php">multilabel_confusion_matrix</td> <td>Computes the Multi Label Confusion Matrix.</td> </tr>
          <tr> <td><a href="negative_predictive_score/index.php">negative_predictive_score</td> <td>Computes the Negative Predictive Score.</td> </tr>
          <tr> <td><a href="prc_auc/index.php">prc_auc</td> <td>Computes the PRC AUC (Area Under Curve).</td> </tr>
          <tr> <td><a href="precision_score/index.php">precision_score</td> <td>Computes the Precision Score.</td> </tr>
          <tr> <td><a href="recall_score/index.php">recall_score</td> <td>Computes the Recall Score.</td> </tr>
          <tr> <td><a href="specificity_score/index.php">specificity_score</td> <td>Computes the Specificity Score.</td> </tr>
        </table>
      <div><?php include('../../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Model Selection</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="elbow/index.php">elbow</td> <td>Draws the an elbow curve.</td> </tr>
            <tr> <td><a href="lift_chart/index.php">lift_chart</td> <td>Draws a lift chart.</td> </tr>
          <tr> <td><a href="prc_curve/index.php">prc_curve</td> <td>Draws a precision-recall curve.</td> </tr>
          <tr> <td><a href="roc_curve/index.php">roc_curve</td> <td>Draws a receiver operating characteristic (ROC) curve.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>

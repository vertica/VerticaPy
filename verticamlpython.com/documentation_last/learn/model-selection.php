<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Model Selection</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="best_k/index.php">best_k</td> <td>Finds the KMeans K based on a score.</td> </tr>
          <tr> <td><a href="cross_validate/index.php">cross_validate</td> <td>Computes the K-Fold cross validation of an estimator.</td> </tr>
          <tr> <td><a href="train_test_split/index.php">train_test_split</td> <td>Creates a temporary table and 2 views which can be used to evaluate a model.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
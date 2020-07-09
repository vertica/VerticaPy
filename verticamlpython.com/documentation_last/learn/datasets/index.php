<!DOCTYPE html>
<html>
    <?php include('../../../include/head.php'); ?>
    <body>
      <div><?php include('../../../include/header.php'); ?></div>
      <div id="content">
        <h2>Datasets</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="load_amazon/index.php">load_amazon</td> <td>Ingests the Amazon dataset in the Vertica DB (Dataset ideal for TS and Regression).</td> </tr>
          <tr> <td><a href="load_iris/index.php">load_iris</td> <td>Ingests the Iris dataset in the Vertica DB (Dataset ideal for Classification and Clustering).</td> </tr>
          <tr> <td><a href="load_market/index.php">load_market</td> <td>Ingests the market dataset in the Vertica DB (Dataset ideal for easy exploration).</td> </tr>
          <tr> <td><a href="load_smart_meters/index.php">load_smart_meters</td> <td>Ingests the smart meters dataset in the Vertica DB (Dataset ideal for TS and Regression).</td> </tr>
          <tr> <td><a href="load_titanic/index.php">load_titanic</td> <td>Ingests the Titanic dataset in the Vertica DB (Dataset ideal for Classification).</td> </tr>
          <tr> <td><a href="load_winequality/index.php">load_winequality</td> <td>Ingests the winequality dataset in the Vertica DB (Dataset ideal for Regression and Classification).</td> </tr>
        </table>
      <div><?php include('../../../include/footer.php'); ?></div>
      </div>
    </body>
</html>

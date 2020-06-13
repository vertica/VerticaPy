<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Statistical Tests</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/testdf/index.php">testdf</td> <td>Dickey Fuller test (Time Series stationarity).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/testdw/index.php">testdw</td> <td>Durbin Watson test (residuals autocorrelation).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/testjb/index.php">testjb</td> <td>Jarque Bera test (Distribution Normality).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/testmk/index.php">testmk</td> <td>Mann Kendall test (Time Series trend).</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
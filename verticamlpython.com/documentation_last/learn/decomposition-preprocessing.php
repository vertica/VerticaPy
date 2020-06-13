<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Decomposition</h2>
        <table class="functions_description">
          <tr> <th>Class</th> <th>Definition</th> </tr>
          <tr> <td><a href="PCA/index.php">PCA</td> <td>Creates a PCA (Principal Component Analysis) object by using the Vertica Highly Distributed and Scalable PCA on the data.</td> </tr>
          <tr> <td><a href="SVD/index.php">SVD</td> <td>Creates a SVD (Singular Value Decomposition) object by using the Vertica Highly Distributed and Scalable SVD on the data.</td> </tr>
        </table>
        <h2>Preprocessing</h2>
        <table class="functions_description">
          <tr> <th>Class / Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="Balance/index.php">Balance</td> <td>Creates a view with an equal distribution of the input data based on the response_column.</td> </tr>
          <tr> <td><a href="CountVectorizer/index.php">CountVectorizer</td> <td>Creates a Text Index which will count the occurences of each word in the 
data.</td> </tr>
          <tr> <td><a href="Normalizer/index.php">Normalizer</td> <td>Creates a Vertica Normalizer object.</td> </tr>
          <tr> <td><a href="OneHotEncoder/index.php">OneHotEncoder</td> <td>Create a Vertica One Hot Encoder object.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
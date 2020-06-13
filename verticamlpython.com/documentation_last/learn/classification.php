<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Binary Classification</h2>
        <table class="functions_description">
          <tr> <th>Class</th> <th>Definition</th> </tr>
          <tr> <td><a href="LinearSVC/index.php">LinearSVC</td> <td>Creates a LinearSVC object by using the Vertica Highly Distributed and Scalable SVM on the data.</td> </tr>
          <tr> <td><a href="LogisticRegression/index.php">LogisticRegression</td> <td>Creates a LogisticRegression object by using the Vertica Highly Distributed and Scalable Logistic Regression on the data.</td> </tr>
        </table>
        <h2>Multiclass Classification</h2>
        <table class="functions_description">
          <tr> <th>Class</th> <th>Definition</th> </tr>
          <tr> <td><a href="DecisionTreeClassifier/index.php">DecisionTreeClassifier</td> <td>Single Decision Tree Classifier.</td> </tr>
          <tr> <td><a href="DummyTreeClassifier/index.php">DummyTreeClassifier</td> <td>This classifier learns by heart the training data.</td> </tr>
          <tr> <td><a href="KNeighborsClassifier/index.php">KNeighborsClassifier</td> <td>Creates a KNeighborsClassifier object by using the K Nearest Neighbors Algorithm.</td> </tr>
          <tr> <td><a href="MultinomialNB/index.php">MultinomialNB</td> <td>Creates a MultinomialNB object by using the Vertica Highly Distributed and Scalable Naive Bayes on the data.</td> </tr>
          <tr> <td><a href="NearestCentroid/index.php">NearestCentroid</td> <td>Creates a NearestCentroid object by using the K Nearest Centroid Algorithm.</td> </tr>
          <tr> <td><a href="RandomForestClassifier/index.php">RandomForestClassifier</td> <td>Creates a RandomForestClassifier object by using the Vertica Highly Distributed and Scalable Random Forest on the data.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
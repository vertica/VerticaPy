<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Encoding</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/decode/index.php">decode</td> <td>Encodes the vcolumn using a User Defined Encoding.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/discretize/index.php">discretize</td> <td>Discretizes the vcolumn using the input method.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/get_dummies/index.php">get_dummies</td> <td>Encodes the vcolumns using the One Hot Encoding algorithm.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/get_dummies/index.php">get_dummies</td> <td>Encodes the vcolumn using the One Hot Encoding algorithm.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/label_encode/index.php">label_encode</td> <td>Encodes the vcolumn using a bijection from the different categories to [0, n - 1]</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/mean_encode/index.php">mean_encode</td> <td>Encode the vcolumn using the average of the response partitioned by the different vcolumn categories.</td> </tr>
        </table>
        <h2>Dealing with Missing Values</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/dropna/index.php">dropna</td> <td>Filters the vDataFrame where the input vcolumns are missing.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/dropna/index.php">dropna</td> <td>Filters the vDataFrame where the vcolumn is missing.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/fillna/index.php">fillna</td> <td>Fills the vcolumns missing elements using specific rules.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/fillna/index.php">fillna</td> <td>Fills the vcolumn missing elements using specific rules.</td> </tr>
        </table>
        <h2>Normalization and Global Outliers</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/clip/index.php">clip</td> <td>Clips the vcolumn.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/fill_outliers/index.php">fill_outliers</td> <td>Fills the vcolumns outliers using the input method.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/normalize/index.php">normalize</td> <td>Normalizes the input vcolumns using the input method.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/normalize/index.php">normalize</td> <td>Normalizes the input vcolumns using the input method.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/outliers/index.php">outliers</td> <td>Adds a new vcolumn labeled with 0 and 1. 1 means that the record is a global outlier.</td> </tr>
        </table>
        <h2>Data Types Conversion</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/astype/index.php">astype</td> <td>Converts the vcolumns to the input types.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/astype/index.php">astype</td> <td>Converts the vcolumn to the input type.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/bool_to_int/index.php">bool_to_int</td> <td>Converts all the booleans vcolumns to integers.</td> </tr>
        </table>
        <h2>Renaming</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/rename/index.php">rename</td> <td>Renames the vcolumn.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
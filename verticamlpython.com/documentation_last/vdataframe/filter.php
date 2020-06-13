<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Search</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/search/index.php">search</td> <td>Searches the elements which matches with the input conditions.</td> </tr>
        </table>
        <h2>Filter Records</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/at_time/index.php">at_time</td> <td>Filters the vDataFrame by only keeping the records at the input time.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/between_time/index.php">between_time</td> <td>Filters the vDataFrame by only keeping the records between two input times.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/filter/index.php">filter</td> <td>Filters the vDataFrame using the input expressions.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/first/index.php">first</td> <td>Filters the vDataFrame by only keeping the first records.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/last/index.php">last</td> <td>Filters the vDataFrame by only keeping the last records.</td> </tr>
        </table>
        <h2>Filter Columns</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/drop/index.php">drop</td> <td>Drops the input vcolumns from the vDataFrame.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/drop/index.php">drop</td> <td>Drops the vcolumn from the vDataFrame.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/drop_duplicates/index.php">drop_duplicates</td> <td>Filters the duplicated using a partition by the input vcolumns.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/drop_outliers/index.php">drop_outliers</td> <td>Drops the vcolumns outliers.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/select/index.php">select</td> <td>Returns a copy of the vDataFrame with only the selected vcolumns.</td> </tr>
        </table>
        <h2>Sample</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/sample/index.php">sample</td> <td>Downsamples the vDataFrame by filtering using a random vcolumn.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
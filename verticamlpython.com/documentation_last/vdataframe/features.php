<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Moving Windows</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/cummax/index.php">cummax</td> <td>Adds a new vcolumn to the vDataFrame by computing the cumulative maximum of the input vcolumn.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/cummin/index.php">cummin</td> <td>Adds a new vcolumn to the vDataFrame by computing the cumulative minimum of the input vcolumn.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/cumprod/index.php">cumprod</td> <td>Adds a new vcolumn to the vDataFrame by computing the cumulative product of the input vcolumn.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/cumsum/index.php">cumsum</td> <td>Adds a new vcolumn to the vDataFrame by computing the cumulative sum of the input vcolumn.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/rolling/index.php">rolling</td> <td>Adds a new vcolumn to the vDataFrame by using an advanced analytical window function on one or two specific vcolumns.</td> </tr>
        </table>
        <h2>Analytic Functions</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/analytic/index.php">analytic</td> <td>Adds a new vcolumn to the vDataFrame by using an advanced analytical function on one or two specific vcolumns.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/asfreq/index.php">asfreq</td> <td>Computes a regular time interval vDataFrame by interpolating the missing values using different techniques.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/sessionize/index.php">sessionize</td> <td>Adds a new vcolumn to the vDataFrame which will correspond to sessions.</td> </tr>
        </table>
        <h2>Customized Features Creation</h2>
        <table class="functions_description">
          <tr> <td>vDataFrame.<a href="main-methods/case_when/index.php">case_when</td> <td>Creates a new feature by evaluating some conditions.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/eval/index.php">eval</td> <td>Evaluates a customized expression.</td> </tr>
        </table>
        <h2>Features Transformations</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/abs/index.php">abs</td> <td>Applies the absolute value function to the input vcolumns.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/abs/index.php">abs</td> <td>Applies the absolute value function to the input vcolumn.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/apply/index.php">apply</td> <td>Applies each function of the dictionary to the input vcolumns.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/apply/index.php">apply</td> <td>Applies a function to the vcolumn.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/apply_fun/index.php">apply_fun</td> <td>Applies a default function to the vcolumn.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/applymap/index.php">applymap</td> <td>Applies a function to all the vcolumns.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/date_part/index.php">date_part</td> <td>Extracts a specific TS field from the vcolumn.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/round/index.php">round</td> <td>Rounds the vcolumn by keeping only the input number of digits after comma.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/slice/index.php">slice</td> <td>Slices the vcolumn using a TS rule. The vcolumn will be transformed.</td> </tr>
        </table>
        <h2>Working with Text</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/regexp/index.php">regexp</td> <td>Computes a new vcolumn based on regular expressions.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/str_contains/index.php">str_contains</td> <td>Verifies if the regular expression is in each of the vcolumn records. The vcolumn will be transformed.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/str_count/index.php">str_count</td> <td>Computes the regular expression count match in each record of the vcolumn. The vcolumn will be transformed.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/str_extract/index.php">str_extract</td> <td>Extracts the regular expression in each record of the vcolumn. The vcolumn will be transformed.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/str_replace/index.php">str_replace</td> <td>Replaces the regular expression matches in each of the vcolumn record by an input value. The vcolumn will be transformed.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/str_slice/index.php">str_slice</td> <td>Slices the vcolumn. The vcolumn will be transformed.</td> </tr>
        </table>
        <h2>Binary Operator Functions</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/add/index.php">add</td> <td>Adds the input element to the vcolumn.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/div/index.php">div</td> <td>Divides the vcolumn by the input element.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/mul/index.php">mul</td> <td>Multiplies the vcolumn by the input element.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/sub/index.php">sub</td> <td>Substracts the input element to the vcolumn.</td> </tr>
        </table>
        <h2>Copy</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/add_copy/index.php">add_copy</td> <td>Adds a copy vcolumn to the parent vDataFrame.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/copy/index.php">copy</td> <td>Returns a copy of the vDataFrame.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>

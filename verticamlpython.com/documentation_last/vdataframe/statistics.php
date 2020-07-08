<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Descriptive Statistics</h2>
        <table class="functions_description">
          <tr> <th>Method</th> <th>Definition</th> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/agg/index.php">agg / aggregate</td> <td>Aggregates the vDataFrame using the input functions.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/agg/index.php">agg / aggregate</td> <td>Aggregates the vcolumn using the input functions.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/all/index.php">all</td> <td>Aggregates the vDataFrame using 'bool_and'.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/any/index.php">any</td> <td>Aggregates the vDataFrame using 'bool_or'.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/avg/index.php">avg / mean</td> <td>Aggregates the vDataFrame using 'avg' (Average).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/avg/index.php">avg / mean</td> <td>Aggregates the vcolumn using 'avg' (Average).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/count/index.php">count</td> <td>Aggregates the vDataFrame using a list of 'count' (Number of missing values).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/count/index.php">count</td> <td>Aggregates the vcolumn using 'count' (Number of Missing elements).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/describe/index.php">describe</td> <td>Aggregates the vDataFrame using multiple statistical aggregations.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/describe/index.php">describe</td> <td>Aggregates the vcolumn using multiple statistical aggregations.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/distinct/index.php">distinct</td> <td>Returns the vcolumn distinct categories.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/duplicated/index.php">duplicated</td> <td>Returns the duplicated values.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/groupby/index.php">groupby</td> <td>Aggregates the vDataFrame by grouping the elements.</td> 
          <tr> <td>vDataFrame.<a href="main-methods/isin/index.php">isin</td> <td>Looks if some specific records are in the vDataFrame.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/isin/index.php">isin</td> <td>Looks if some specific records are in the vcolumn.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/kurt/index.php">kurt / kurtosis</td> <td>Aggregates the vDataFrame using 'kurtosis'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/kurt/index.php">kurt / kurtosis</td> <td>Aggregates the vcolumn using 'kurtosis'.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/mad/index.php">mad</td> <td>Aggregates the vDataFrame using 'mad' (Median Absolute Deviation).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/mad/index.php">mad</td> <td>Aggregates the vcolumn using 'mad' (Median Absolute Deviation).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/mae/index.php">mae</td> <td>Aggregates the vDataFrame using 'mae' (Mean Absolute Error).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/mae/index.php">mae</td> <td>Aggregates the vcolumn using 'mae' (Mean Absolute Error).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/max/index.php">max</td> <td>Aggregates the vDataFrame using 'max' (Maximum).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/max/index.php">max</td> <td>Aggregates the vcolumn using 'max' (Maximum).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/median/index.php">median</td> <td>Aggregates the vDataFrame using 'median'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/median/index.php">median</td> <td>Aggregates the vcolumn using 'median'.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/min/index.php">min</td> <td>Aggregates the vDataFrame using 'min' (Minimum).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/min/index.php">min</td> <td>Aggregates the vcolumn using 'min' (Minimum).</td> </tr>
            <tr> <td>vDataFrame[].<a href="vcolumn-methods/mode/index.php">mode</td> <td>Returns the nth most occurent element.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/nlargest/index.php">nlargest</td> <td>Returns the n largest vcolumn elements.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/nsmallest/index.php">nsmallest</td> <td>Returns the n smallest vcolumn elements.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/nunique/index.php">nunique</td> <td>Aggregates the vDataFrame using 'unique' (cardinality).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/numh/index.php">numh</td> <td>Computes the optimal vcolumn bar width.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/nunique/index.php">nunique</td> <td>Aggregates the vcolumn using 'unique' (cardinality).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/prod/index.php">prod /product</td> <td>Aggregates the vDataFrame using 'product'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/prod/index.php">prod /product</td> <td>Aggregates the vcolumn using 'product'.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/quantile/index.php">quantile</td> <td>Aggregates the vDataFrame using a list of 'quantiles'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/quantile/index.php">quantile</td> <td>Aggregates the vcolumn using an input 'quantile'.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/score/index.php">score</td> <td>Computes the score using the input columns and the input method.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/sem/index.php">sem</td> <td>Aggregates the vDataFrame using 'sem' (Standard Error of the Mean).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/sem/index.php">sem</td> <td>Aggregates the vcolumn using 'sem' (Standard Error of the Mean).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/shape/index.php">shape</td> <td>Returns the number of rows and columns of the vDataFrame.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/skew/index.php">skew / skewness</td> <td>Aggregates the vDataFrame using 'skewness'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/skew/index.php">skew / skewness</td> <td>Aggregates the vcolumn using 'skewness'.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/statistics/index.php">statistics</td> <td>Aggregates the vDataFrame using multiple statistical aggregations.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/std/index.php">std</td> <td>Aggregates the vDataFrame using 'std' (Standard Deviation).</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/std/index.php">std</td> <td>Aggregates the vcolumn using 'std' (Standard Deviation).</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/sum/index.php">sum</td> <td>Aggregates the vDataFrame using 'sum'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/sum/index.php">sum</td> <td>Aggregates the vcolumn using 'sum'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/topk/index.php">topk</td> <td>Returns the top-k most occurent elements and their percentages of the distribution.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/value_counts/index.php">value_counts</td> <td>Returns the top-k most frequent elements and how often they appear.</td> </tr>
          <tr> <td>vDataFrame.<a href="main-methods/var/index.php">var</td> <td>Aggregates the vDataFrame using 'variance'.</td> </tr>
          <tr> <td>vDataFrame[].<a href="vcolumn-methods/var/index.php">var</td> <td>Aggregates the vcolumn using 'variance'.</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>

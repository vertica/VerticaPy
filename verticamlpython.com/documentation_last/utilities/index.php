<!DOCTYPE html>
<html>
    <?php include('../../include/head.php'); ?>
    <body>
      <div><?php include('../../include/header.php'); ?></div>
      <div id="content">
        <h2>Ingest Data</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="pandas_to_vertica/index.php">pandas_to_vertica</a></td> <td>Ingests a pandas DataFrame to Vertica DB by creating a CSV file first and then using flex tables.</td> </tr>
          <tr> <td><a href="pcsv/index.php">pcsv</a></td> <td>Parses a CSV file using flex tables and identifies the columns and their respective types.</td> </tr>
          <tr> <td><a href="pjson/index.php">pjson</a></td> <td>Parses a JSON file using flex tables and identifies the columns and their respective types.</td> </tr>
          <tr> <td><a href="read_csv/index.php">read_csv</a></td> <td>Ingests a CSV file using flex tables.</td> </tr>
          <tr> <td><a href="read_json/index.php">read_json</a></td> <td>Ingests a JSON file using flex tables.</td> </tr>
          <tr> <td><a href="read_vdf/index.php">read_vdf</a></td> <td>Reads a VDF file and create the associated vDataFrame.</td> </tr>
        </table>
        <h2>Read Data</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="get_data_types/index.php">get_data_types</a></td> <td>Returns customized relation columns and the respective data types. Creates a temporary table during the process.</td></tr>
          <tr> <td><a href="load_model/index.php">load_model</a></td> <td>Loads a Vertica model and returns the associated object.</td> </tr>
          <tr> <td><a href="readSQL/index.php">readSQL</a></td> <td>Returns the result of a SQL query as a tablesample object.</td> </tr>
          <tr> <td><a href="tablesample/index.php">tablesample</a></td><td>Displays query results. This object does not have dependencies with any other module.</td> </tr>
          <tr> <td><a href="to_tablesample/index.php">to_tablesample</a></td> <td>Returns the result of a SQL query as a tablesample object.</td> </tr>
          <tr> <td><a href="vdf_from_relation/index.php">vdf_from_relation</a></td> <td>Creates a vDataFrame based on a customized relation.</td> </tr>
        </table>
        <h2>Drop Data</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="drop_model/index.php">drop_model</a></td> <td>Drops the input model.</td> </tr>
          <tr> <td><a href="drop_table/index.php">drop_table</a></td> <td>Drops the input table.</td> </tr>
          <tr> <td><a href="drop_text_index/index.php">drop_text_index</a></td> <td>Drops the input text index.</td> </tr>
          <tr> <td><a href="drop_view/index.php">drop_view</a></td> <td>Drops the input view.</td> </tr>
        </table>
        <h2>Help</h2>
        <table class="functions_description">
          <tr> <th>Function</th> <th>Definition</th> </tr>
          <tr> <td><a href="vHelp/index.php">vHelp</a></td> <td>Help module (FAQ).</td> </tr>
        </table>
      <div><?php include('../../include/footer.php'); ?></div>
      </div>
    </body>
</html>
